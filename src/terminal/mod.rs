use crate::kitty_graphics::KittyGraphicsState;
use base64::Engine;
use jterm_core::click_cursor;
use smallvec::SmallVec;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Character class for word selection boundaries.
#[derive(PartialEq)]
enum CharClass {
    Word,
    Whitespace,
    Symbol,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_whitespace_char(c: char) -> bool {
    c == ' ' || c == '\t' || c == '\0'
}

fn char_class(c: char) -> CharClass {
    if is_word_char(c) {
        CharClass::Word
    } else if is_whitespace_char(c) {
        CharClass::Whitespace
    } else {
        CharClass::Symbol
    }
}

fn is_extended_token_separator(c: char) -> bool {
    matches!(
        c,
        '/' | '\\' | '.' | ':' | '-' | '~' | '?' | '&' | '=' | '#' | '%' | '+' | '@'
    )
}

fn is_extended_token_char(c: char) -> bool {
    is_word_char(c) || is_extended_token_separator(c)
}

fn is_token_prefix_wrapper(c: char) -> bool {
    matches!(c, '"' | '\'' | '`' | '(' | '[' | '{' | '<')
}

fn is_token_suffix_wrapper(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '`' | ')' | ']' | '}' | '>' | ',' | ';' | '!'
    )
}

const PRIMARY_DEVICE_ATTRIBUTES_RESPONSE: &[u8] = b"\x1b[?65;1;9c";
const SECONDARY_DEVICE_ATTRIBUTES_RESPONSE: &[u8] = b"\x1b[>1;7802;0c";
const XTERM_VERSION_RESPONSE: &[u8] = b"\x1bP>|VTE(7802)\x1b\\";
pub const MAX_TERMINAL_COLS: usize = 1024;
pub const MAX_TERMINAL_ROWS: usize = 512;
/// Hard cap on bytes carried across PTY read batches inside an unfinished
/// OSC/DCS/CSI escape. Any well-formed sequence is far below this; a
/// runaway/binary stream that never sends a terminator (BEL/ST/final byte)
/// would otherwise grow the pending escape buffers without bound.
pub const MAX_PENDING_ESCAPE: usize = 4 * 1024 * 1024;

/// Clipboard reads triggered by Kitty paste events are capabilities, not a
/// general permission to inspect the host clipboard. Keep the capability
/// short-lived and bound all protocol-controlled collections.
const OSC_5522_PASTE_GRANT_TTL: std::time::Duration = std::time::Duration::from_secs(10);
const MAX_PENDING_CLIPBOARD_REQUESTS: usize = 8;
const MAX_OSC_5522_MIME_TYPES: usize = 64;
const MAX_OSC_5522_MIME_LEN: usize = 256;

type VisibleCellsCache = (u64, usize, std::sync::Arc<Vec<Vec<TerminalCell>>>);

#[derive(Clone, Copy)]
struct ViewportMappingExactCache {
    cols: usize,
    rows: usize,
    scroll_offset: usize,
    scrollback_len: usize,
    total_lines_scrolled: u64,
    exact: bool,
}

/// Hard cap on tracked OSC 133 command records. Protocol strings are bounded
/// separately, so even an untrusted process attached to the PTY cannot grow
/// terminal state without limit.
pub const MAX_COMMAND_MARKS: usize = 1024;
/// Window titles are bounded where they enter terminal state, not only where
/// they are drawn: `app::window::safe_window_title` caps the *displayed*
/// title at the same number, but it can only cap the string it is handed, and
/// an OSC 0/2 payload arrives from a pending-escape buffer that tolerates
/// megabytes.
pub(crate) const MAX_WINDOW_TITLE_CHARS: usize = 200;
/// Depth of the XTWINOPS title stack (ops 22/23), matching frost.
pub(crate) const MAX_TITLE_STACK_DEPTH: usize = 16;
/// OSC 133 per-field byte budgets are the shared protocol's constants, not a
/// fourth per-app set. Ember's decoder and `jterm_core::parser::CommandMeta`
/// read the same packets, so a cap that differs makes the two disagree about
/// whether a packet carried a command, a cwd or an id at all — and the journal
/// lifecycle token minted from ember's decode would then bind durable output to
/// identity the core writer re-derives differently. Ember's own numbers were
/// looser than anything jsh emits (64 KiB command, 16 KiB cwd).
///
/// The id budget is core's generic-FinalTerm one, deliberately wider than the
/// 192-byte `is_valid_jsh_execution_id` grammar: an opaque id from a non-jsh
/// shell still names a block in the timeline, and only the journal path is
/// gated on the narrower jsh token.
const MAX_OSC_133_COMMAND_BYTES: usize = jterm_core::parser::MAX_OSC133_COMMAND_BYTES;
const MAX_OSC_133_ID_BYTES: usize = jterm_core::parser::MAX_OSC133_ID_BYTES;
const MAX_OSC_133_CWD_BYTES: usize = jterm_core::parser::MAX_OSC133_CWD_BYTES;
const MAX_OSC_133_SESSION_ID_BYTES: usize = jterm_core::parser::MAX_OSC133_SESSION_ID_BYTES;
const MAX_PENDING_COMPLETED_COMMANDS: usize = 32;
const MAX_CONSUMED_COMMAND_IDS: usize = MAX_COMMAND_MARKS;
pub const MAX_COMPLETED_COMMAND_OUTPUT_BYTES: usize = 256 * 1024;
pub const MAX_CAPTURED_COMMAND_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

/// One entry recorded by an OSC 133-aware shell. `line_id` is the monotonic
/// id of the row where the prompt began; we resolve it to a current
/// scrollback/grid row at navigation time, since scrollback indices shift
/// when old lines get evicted.
#[derive(Clone, Copy, Debug)]
pub struct CommandMark {
    /// Monotonic line id of the prompt row, equal to total_lines_scrolled
    /// at record time plus the prompt's viewport row.
    pub line_id: u64,
    /// Exit code reported by `OSC 133;D;<n>`. None until the command exits.
    pub exit_code: Option<i32>,
}

/// Stable terminal-buffer coordinate used by semantic command records.
///
/// `line_id` is monotonic for the lifetime of the primary screen. Unlike a
/// scrollback index it does not change when old scrollback rows are evicted.
/// An anchor can nevertheless become unavailable once its row is evicted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferAnchor {
    pub line_id: u64,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandState {
    Prompt,
    Editing,
    Running,
    Complete,
}

/// A complete OSC 133 A/B/C/D lifecycle. The legacy [`CommandMark`] list is
/// retained for source compatibility, while this is the canonical semantic
/// representation used by command-history UI and AI context capture.
#[derive(Clone, Debug)]
pub struct CommandRecord {
    /// Shell supplied `jsh_id`/`id`, or a terminal-local stable fallback.
    pub id: String,
    pub sequence: u64,
    /// Exact command supplied as `cmdline_url`, when available. For generic
    /// shells this may be reconstructed from the final displayed input.
    pub command: Option<String>,
    /// True only when the shell supplied exact command metadata. Screen
    /// reconstruction is useful for display, but must not authorize rerun.
    pub command_exact: bool,
    /// The producer explicitly omitted or shortened an oversized command.
    pub command_truncated: bool,
    /// Working directory in which the command started (OSC 133 C).
    pub cwd: Option<String>,
    /// Working directory reported after the command finished (OSC 133 D).
    /// Kept separate so a stateful `cd` cannot rewrite execution provenance.
    pub cwd_after: Option<String>,
    /// jsh's persistent session identity, observed on the OSC 133 `C` mark.
    ///
    /// This and the two fields below are the Start-identity triple that
    /// [`jterm_core::execution_journal::ExecutionLifecycle`] demands alongside
    /// `id` before durable output may be written. jsh emits all three on `C`
    /// and none of them on `D`, and the shared parser accepts them only there,
    /// so they are captured at output start and never minted or replaced at
    /// completion: a token assembled from metadata seen at two different times
    /// would bind captured output to a Start generation nobody observed.
    pub session_id: Option<String>,
    /// Shell-local sequence number of this exact Start generation.
    pub seq: Option<u64>,
    /// Wall-clock start identity copied from jsh's own journal Start.
    pub started_at_ms: Option<u64>,
    pub prompt_start: BufferAnchor,
    pub command_start: Option<BufferAnchor>,
    pub output_start: Option<BufferAnchor>,
    pub output_end: Option<BufferAnchor>,
    pub end: Option<BufferAnchor>,
    pub exit_code: Option<i32>,
    pub duration_ms: Option<u64>,
    pub state: CommandState,
    pub complete: bool,
    /// Whether the matching OSC 133 `C` execution-start mark was observed.
    /// This survives scrollback eviction independently of row anchors.
    pub start_mark_seen: bool,
    /// Evidence that closed the lifecycle. Outcome (`exit_code`) remains a
    /// separate fact and is never synthesized from this value.
    pub completion_provenance: crate::block_mode::CompletionProvenance,
    pub started_at: Option<std::time::SystemTime>,
    pub finished_at: Option<std::time::SystemTime>,
    /// Locally armed Agent approval associated with this exact command
    /// lifecycle. This value never comes from PTY-controlled OSC metadata.
    pub agent_generation: Option<u64>,
    /// Bounded normalized output captured at D. This survives scrollback
    /// eviction; `truncated`/`total_bytes` describe the original range.
    pub captured_output: Option<ExtractedText>,
    started_instant: Option<std::time::Instant>,
}

/// Plain-text terminal extraction with explicit capacity/truncation metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtractedText {
    pub text: String,
    pub truncated: bool,
    /// UTF-8 byte length the normalized text would have without the cap.
    pub total_bytes: usize,
}

/// Completed-command event drained by the app/session pump. Capturing happens
/// at OSC 133;D while the output anchors are still likely to be retained; file
/// IO and journal persistence intentionally stay outside `TerminalState`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedCommandOutput {
    pub id: String,
    /// Start identity observed on this execution's OSC 133 `C` mark. All three
    /// must be present, together with a jsh-shaped `id`, before the execution
    /// journal will accept the captured output; see [`CommandRecord::session_id`].
    pub session_id: Option<String>,
    pub seq: Option<u64>,
    pub started_at_ms: Option<u64>,
    pub command: Option<String>,
    pub cwd: Option<String>,
    pub exit_code: Option<i32>,
    pub duration_ms: Option<u64>,
    pub output: String,
    pub output_available: bool,
    pub truncated: bool,
    pub total_bytes: usize,
    /// One-shot local Agent approval generation. `None` for every command
    /// that was not armed by the application before its bytes were queued.
    pub agent_generation: Option<u64>,
}

/// Additive lifecycle envelope around the source-compatible completed-output
/// payload. Existing integrations may keep constructing and draining
/// [`CompletedCommandOutput`]; provenance-aware consumers use this event so a
/// boundary-inferred termination cannot masquerade as shell-reported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedCommandEvent {
    pub completed: CompletedCommandOutput,
    pub start_mark_seen: bool,
    pub completion_provenance: crate::block_mode::CompletionProvenance,
}

impl CompletedCommandEvent {
    pub fn lifecycle_health(&self) -> crate::block_mode::BlockLifecycleHealth {
        crate::block_mode::assess_lifecycle(self.start_mark_seen, self.completion_provenance)
    }

    pub fn is_trusted_completion(&self) -> bool {
        matches!(
            self.lifecycle_health(),
            crate::block_mode::BlockLifecycleHealth::Healthy
                | crate::block_mode::BlockLifecycleHealth::Recovered
        )
    }
}

impl std::ops::Deref for CompletedCommandEvent {
    type Target = CompletedCommandOutput;

    fn deref(&self) -> &Self::Target {
        &self.completed
    }
}

impl std::ops::DerefMut for CompletedCommandEvent {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.completed
    }
}

#[derive(Clone, Debug)]
struct ArmedAgentExecution {
    generation: u64,
    command_sequence: u64,
    command: String,
}

pub fn clamp_terminal_dimensions(cols: usize, rows: usize) -> (usize, usize) {
    (
        cols.clamp(1, MAX_TERMINAL_COLS),
        rows.clamp(1, MAX_TERMINAL_ROWS),
    )
}

mod grid;
mod hyperlink;
mod parser;
#[allow(dead_code)] // Public in the library; the binary mirrors modules privately.
mod projection;
mod state;
#[cfg(test)]
mod tests;

pub use grid::*;
pub(crate) use hyperlink::is_supported_hyperlink_uri;
pub use hyperlink::HyperlinkId;
#[allow(unused_imports)] // Public P1 contract; UI consumers land in the next slice.
pub use projection::{
    DisplayPoint, FinishedOutputRange, HistoryProjection, ProjectedBufferAnchorLocation,
    ProjectedRowKind, ProjectedViewport, ProjectionCacheKey, ProjectionLayoutKey, ProjectionPolicy,
    ProjectionViewState, RawCellAnchor, RawCellBoundary, SyntheticRowKey,
};

#[derive(Clone, Debug)]
#[allow(dead_code)] // Consumed by the transformed projection slice.
struct FinishedOutputProvenance {
    range: FinishedOutputRange,
    start_line_id: u64,
    rows: Vec<FinishedOutputRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinishedOutputRow {
    row_id: RawRowId,
    start_col: usize,
    end_col: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinishedOutputOwner {
    zone_id: u64,
    start_col: usize,
    end_col: usize,
}

#[cfg(test)]
thread_local! {
    static FINISHED_OUTPUT_EVICTION_ROW_CHECKS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[derive(Clone, Copy, Debug)]
struct ActiveOutputProvenance {
    zone_id: u64,
    start: BufferAnchor,
    start_row_id: RawRowId,
    extent: Option<BufferAnchor>,
    extent_row_id: Option<RawRowId>,
    last_write_end: Option<BufferAnchor>,
    /// Boundary produced by real terminal output events. Printable writes use
    /// their cell end; LF/IND use the following row at column zero. Explicit
    /// cursor positioning never updates this value.
    semantic_end: Option<BufferAnchor>,
    /// CR/BS/tab and other horizontal cursor controls are only exact when a
    /// subsequent output event resolves them. A D immediately after the move
    /// must fail closed rather than infer cells from cursor position alone.
    cursor_moved_since_output: bool,
    /// Unsupported non-linear output (for example a write above OSC 133 C)
    /// must fail closed instead of claiming prompt/header cells.
    invalid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Normal,
    Block,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub anchor: (usize, usize),
    pub active: (usize, usize),
    pub mode: SelectionMode,
}

/// Selection coordinates in a transformed projected document. These cannot
/// be mixed with raw scrollback indices because collapse summaries introduce
/// holes and synthetic rows.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectedSelection {
    plan_revision: u64,
    /// Width at which endpoint columns were minted. Reflowing to another
    /// width can change the characters between two otherwise-live anchors.
    plan_cols: usize,
    /// Effective hidden set, not merely requested collapse policy. A collapse
    /// can become ineffective after history churn without a policy revision.
    hidden: BTreeSet<u64>,
    anchor: ProjectedSelectionEndpoint,
    active: ProjectedSelectionEndpoint,
    mode: SelectionMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectedSelectionAnchor {
    Cell(RawCellAnchor),
    /// A blank row or a column beyond retained text has row identity but no
    /// exact raw-cell origin. The column is valid only at `plan_cols`.
    Row {
        row: RawRowId,
        column: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProjectedSelectionEndpoint {
    /// `(document row, projected column)` in `plan_revision`.
    point: (usize, usize),
    anchor: ProjectedSelectionAnchor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Charset {
    #[default]
    Ascii,
    DecSpecialGraphics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClipboardReadKind {
    MimeData(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipboardReadRequest {
    pub kind: ClipboardReadKind,
}

#[derive(Debug)]
struct PendingPasteGrant {
    token: String,
    offered_mimes: HashSet<String>,
    expires_at: std::time::Instant,
}

#[derive(Clone, Debug, Default)]
struct TerminalModes {
    bits: u32,
}

impl TerminalModes {
    const fn bit_index(mode: u16) -> Option<u32> {
        match mode {
            // DECCKM — application cursor keys. Keep this independent from
            // alternate-screen modes: full-screen applications explicitly
            // opt into the SS3 cursor-key sequences with CSI ? 1 h.
            1 => Some(13),
            7 => Some(0),
            25 => Some(1),
            1000 => Some(2),
            1001 => Some(3),
            1002 => Some(4),
            1003 => Some(5),
            1004 => Some(6),
            1006 => Some(7),
            1049 => Some(8),
            2004 => Some(9),
            2026 => Some(10),
            2031 => Some(11),
            5522 => Some(12),
            _ => None,
        }
    }

    #[inline]
    fn contains(&self, mode: &u16) -> bool {
        match Self::bit_index(*mode) {
            Some(bit) => self.bits & (1 << bit) != 0,
            None => false,
        }
    }

    #[inline]
    fn insert(&mut self, mode: u16) {
        if let Some(bit) = Self::bit_index(mode) {
            self.bits |= 1 << bit;
        }
    }

    #[inline]
    fn remove(&mut self, mode: &u16) {
        if let Some(bit) = Self::bit_index(*mode) {
            self.bits &= !(1 << bit);
        }
    }
}

/// DECSC/DECRC 保存的完整光标状态(含 SGR 属性、字符集、模式),
/// 与备用屏幕缓冲使用的 saved_cursor_row/col 解耦。
#[derive(Clone)]
struct SavedCursorState {
    row: usize,
    col: usize,
    fg: Color,
    bg: Color,
    flags: StyleFlags,
    g0: Charset,
    g1: Charset,
    active: Charset,
    origin_mode: bool,
    autowrap: bool,
    // VT510: DECSC 须保存 Last Column Flag(延迟换行标志),DECRC 须恢复。
    // 否则右 prompt(如 starship cmd_duration)写到末列后 pending_wrap=true,
    // 经 CSI u/ESC 8 恢复光标到 prompt 之后,下一字符(如 zsh-autosuggestions 的
    // ghost 文本)会立刻触发换行,在屏底触发滚动,光标看似下移一行。
    pending_wrap: bool,
}

/// 256-slot dynamic palette: `None` slots fall through to the theme/default.
pub type DynamicColorPalette = [Option<(u8, u8, u8)>; 256];

pub struct TerminalState {
    pub grid: TerminalGrid,
    alt_grid: TerminalGrid,
    pub scrollback: VecDeque<ScrollbackLine>,
    pub selection: Option<Selection>,
    projected_selection: Option<ProjectedSelection>,
    selection_revision: u64,
    pub scroll_offset: usize,
    max_scrollback: usize,
    use_alt_buffer: bool,

    pub cursor_row: usize,
    pub cursor_col: usize,
    saved_cursor_row: usize,
    saved_cursor_col: usize,
    alt_cursor_row: usize,
    alt_cursor_col: usize,
    pub cursor_shape: CursorShape,

    // DECSC/DECRC 完整保存状态
    saved_state: Option<SavedCursorState>,
    /// The primary screen's drawing state, snapshotted on the way into the
    /// alternate buffer so leaving it restores what the shell had rather than
    /// zeroing SGR. `printf '\e[31m'; vim` must come back red, as it does in
    /// VTE and in frost.
    saved_primary_screen_state: Option<SavedCursorState>,
    // IRM 插入模式 (ANSI mode 4):写字符时右移而非覆盖
    insert_mode: bool,
    /// The last graphic character written, kept pre-translation so REP
    /// (`CSI Pn b`) can reprint it through the charset in force at repeat time.
    last_printed_char: Option<char>,
    // DECOM 原点模式 (DEC private ?6):光标寻址相对滚动区域顶端
    origin_mode: bool,
    // 自定义制表位 (HTS/TBC),index 为列,true 表示该列是制表位
    tab_stops: Vec<bool>,
    // DEC 延迟换行 (Last Column Flag):写满最后一列后置位,下一个可打印字符才换行
    pending_wrap: bool,

    current_fg: Color,
    current_bg: Color,
    current_flags: StyleFlags,
    /// Window title from OSC 0/2, bounded at ingest to
    /// [`MAX_WINDOW_TITLE_CHARS`]. Still untrusted text — the window-manager
    /// sanitiser owns bidi/control filtering — but no longer unbounded.
    pub window_title: String,
    /// The icon title, kept separately from the window title because XTWINOPS
    /// reports and saves them independently (ops 20/21, 22/23). OSC 0 sets
    /// both, OSC 1 only this one, OSC 2 only the window title.
    pub icon_title: String,
    /// xterm's title save/restore stack, depth-capped so a program that only
    /// ever pushes cannot grow it without bound.
    title_stack: Vec<(Option<String>, Option<String>)>,
    /// jsh's own session identity as announced over OSC 7770, when the shell
    /// in this pane announced one.
    ///
    /// Ember passes `--session <its own id>` when *it* spawns jsh, so this is
    /// normally the same value. It is not when the shell reached jsh some
    /// other way — a jsh started by hand inside bash, or over ssh — and then
    /// this is the only key under which that pane's execution journal exists.
    pub announced_jsh_session_id: Option<String>,
    /// Working directory reported by the shell via OSC 7
    /// (`ESC ] 7 ; file://host/path ST`). Optional because many shells need
    /// PROMPT_COMMAND wiring to emit it. When absent the session manager
    /// falls back to `/proc/<pid>/cwd`. Survives across shell PWD changes —
    /// each prompt re-emits OSC 7.
    pub current_working_dir: Option<String>,

    // Global background color set by vim (CSI ... m)
    pub global_bg: Color,

    // Scrolling region (DECSTBM)
    scroll_region_top: usize,
    scroll_region_bottom: usize,

    // UTF-8 decoding buffer
    utf8_buf: [u8; 4],
    utf8_len: u8,
    utf8_expected: u8,

    // Incomplete escape sequence buffer across PTY reads
    pending_escape: Vec<u8>,
    // Kitty APCs are streamed separately so fragmented multi-megabyte payloads
    // are appended and scanned once instead of rebuilding/rescanning the whole
    // escape on every PTY read.
    pending_apc: Vec<u8>,
    pending_apc_scan_from: usize,
    discarding_oversized_apc: bool,
    discarding_apc_prev_escape: bool,
    // OSC and DCS/SOS/PM use the same streaming pattern: large string payloads
    // (e.g. OSC 52 clipboard data) would otherwise be merged and rescanned
    // from byte 0 on every PTY read, which is quadratic on the UI thread.
    pending_osc: Vec<u8>,
    pending_osc_scan_from: usize,
    pending_string: Vec<u8>,
    pending_string_scan_from: usize,

    g0_charset: Charset,
    g1_charset: Charset,
    active_charset: Charset,

    // IME support
    pub ime_enabled: bool,
    pub preedit_text: String,
    pub preedit_cursor: usize,

    modes: TerminalModes,

    // Output buffer for DSR/CPR responses to be sent back to PTY
    pub output_buffer: Vec<u8>,

    keyboard_enhancement_flags: u16,
    keyboard_enhancement_stack: Vec<u16>,
    alt_keyboard_enhancement_flags: u16,
    alt_keyboard_enhancement_stack: Vec<u16>,
    xterm_modify_other_keys: u16,
    xterm_format_other_keys: u16,
    pending_clipboard_requests: Vec<ClipboardReadRequest>,
    pending_paste_grant: Option<PendingPasteGrant>,

    // Kitty graphics protocol support
    pub kitty_graphics: KittyGraphicsState,

    // Dirty rectangle tracking for optimized rendering
    pub dirty_region: DirtyRegion,

    // P4 优化：行版本化追踪
    pub grid_version: u64,      // 全局网格版本号
    pub row_versions: Vec<u64>, // 每行的修改版本号

    // Cached visible cells to avoid per-frame cloning
    visible_cells_cache: Option<VisibleCellsCache>,
    /// Projection metadata is versioned independently from the terminal grid.
    /// P0 is an identity projection over `visible_cells_cache`, so the cell
    /// allocation is shared and only the stable-origin index is cached here.
    projected_viewport_cache: Option<ProjectedViewport>,
    /// Cell-free full-document plan and late-materialized transformed slice.
    /// Their exact keys intentionally have different invalidation domains.
    projection_plan_cache: Option<projection::ProjectionPlanCache>,
    transformed_viewport_cache: Option<projection::TransformedViewportCache>,
    next_projection_plan_revision: u64,
    /// Cached answer for whether raw buffer coordinates exactly match the
    /// lazily reflowed viewport. Interior mutability keeps per-match lookup
    /// O(1) while the first lookup for a new viewport performs one scan.
    viewport_mapping_exact_cache: std::cell::Cell<Option<ViewportMappingExactCache>>,

    // OSC 8 hyperlink tracking. Cells retain only the compact id; the URI is
    // stored once in a bounded table rather than cloned into every cell.
    hyperlinks: hyperlink::HyperlinkTable,
    current_hyperlink: HyperlinkId,

    // Synchronized output (mode 2026): suppress rendering until mode is cleared
    pub sync_output_active: bool,
    sync_output_start: Option<std::time::Instant>,
    last_archived_screen_snapshot: Vec<String>,
    last_synced_primary_screen_snapshot: Vec<String>,

    // OSC 52 clipboard set requests (selection_param, decoded_text)
    pub pending_osc52_clipboard_set: Option<String>,
    // OSC 52 clipboard query pending (needs clipboard read + response)
    pub pending_osc52_clipboard_query: bool,

    // OSC 10/11/12 dynamic colors
    pub dynamic_fg: Option<(u8, u8, u8)>,
    pub dynamic_bg: Option<(u8, u8, u8)>,
    pub dynamic_cursor_color: Option<(u8, u8, u8)>,
    /// OSC 4 per-index palette overrides; OSC 104 resets (all or listed).
    pub dynamic_palette: DynamicColorPalette,

    // OSC 9/777 pending notifications
    pub pending_notifications: Vec<(String, String)>,

    /// Total lines ever pushed into `scrollback` (does not decrement on
    /// `pop_front`). Combined with current `scrollback.len()`, lets us
    /// translate a stable `line_id` back to a current scrollback index even
    /// after old lines have been evicted.
    pub total_lines_scrolled: u64,

    /// Checked monotonic allocator for projection-only physical row identity.
    /// Zero is the exhausted sentinel and is never emitted as a tracked id.
    next_raw_row_id: u64,
    /// Independent generation for row moves/fresh-row creation. This is part
    /// of `ProjectionCacheKey` even when PTY cell bytes did not otherwise bump
    /// the terminal grid version.
    row_identity_revision: u64,
    /// Canonical primary full-screen row-transfer counter. Noncanonical row
    /// moves advance only `row_identity_revision`, so the two key deltas fail
    /// closed for that batch and a later full rebuild establishes a recoverable
    /// baseline. Zero permanently disables advancement only after overflow.
    full_screen_scroll_revision: u64,

    /// OSC 133 command boundaries recorded by FinalTerm-aware shells. FIFO,
    /// capped at `MAX_COMMAND_MARKS`. Marks pointing to lines that have been
    /// evicted from scrollback are pruned lazily during navigation.
    pub command_marks: VecDeque<CommandMark>,

    command_records: VecDeque<CommandRecord>,
    next_command_sequence: u64,
    /// Recently closed OSC execution ids. Records can be evicted or cleared
    /// by RIS, so this bounded authority prevents a delayed duplicate D from
    /// being adopted by a later terminal-local placeholder.
    consumed_command_ids: VecDeque<String>,
    finished_output_provenance: HashMap<u64, FinishedOutputProvenance>,
    finished_output_owners: HashMap<RawRowId, Vec<FinishedOutputOwner>>,
    /// Monotonic structural revision for completed-range ownership. This is
    /// independent from ordinary cell paint changes.
    finished_output_revision: u64,
    active_output_provenance: Option<ActiveOutputProvenance>,
    pending_completed_command_outputs: VecDeque<CompletedCommandEvent>,
    captured_command_output_bytes: usize,
    /// Single-level undo stash for `clear_completed_blocks` (see state.rs).
    /// Dropped by `hard_reset` together with the rest of the block history.
    cleared_blocks_stash: Option<ClearedBlocksStash>,
    agent_prompt_input_tainted: bool,
    armed_agent_execution: Option<ArmedAgentExecution>,
}

/// Bounded snapshot of the records removed by one `clear_completed_blocks`,
/// kept as data so `undo_clear_blocks` can reinsert them (anvil/forge
/// semantics). Inherits the live deque's bounds — at most `MAX_COMMAND_MARKS`
/// records and `MAX_CAPTURED_COMMAND_OUTPUT_BYTES` of captured output — so
/// undo data never needs a second budget.
#[derive(Clone, Debug)]
struct ClearedBlocksStash {
    records: Vec<CommandRecord>,
    /// Finished-output sidecars for the stashed records, keyed implicitly by
    /// each record's `sequence`. Restored alongside so exact output ranges
    /// revalidate against the untouched buffer rows.
    provenance: Vec<FinishedOutputProvenance>,
    captured_output_bytes: usize,
}
