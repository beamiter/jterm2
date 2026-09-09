use super::projection::{
    MaterializedProjection, OriginSpan, ProjectedTopAnchor, ProjectionMode, ProjectionPlan,
    ProjectionPlanCacheKey, ProjectionPlanRow, RawRowLayout, ResolvedCollapse,
    TransformedViewportCacheKey,
};
use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static PROJECTION_PLAN_HISTORY_LAYOUT_VISITS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static PROJECTION_PLAN_ORACLE_HISTORY_DECOMPRESSES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static PROJECTION_PLAN_BUILD_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static PROJECTION_VIEW_HISTORY_DECOMPRESSES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static VISIBLE_CELLS_RECYCLE_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

const OUTPUT_TRUNCATION_MARKER: &str = "\n… output truncated …\n";

fn allocate_raw_row_id_from(next: &mut u64) -> RawRowId {
    if *next == 0 {
        return RawRowId::UNTRACKED;
    }
    let id = RawRowId::new(*next);
    *next = next.checked_add(1).unwrap_or(0);
    id
}

/// Whether a cell is styled the way shells paint an inline suggestion.
///
/// There is no protocol for "this text is a preview", only a convention, and
/// the convention is a muted grey: `jsh` prints its suggestion in ANSI colour 8
/// (`ESC[38;5;8m`), zsh-autosuggestions defaults to the same colour, and `dim`
/// (SGR 2) is the other spelling. Being wrong in the permissive direction
/// accepts text the user never typed, so a cell that merely *might* be a
/// suggestion is treated as one — the cost is that a click cannot place the
/// cursor inside genuinely grey text, which is a click that does nothing.
fn is_inline_suggestion_cell(cell: &TerminalCell) -> bool {
    cell.flags.dim() || matches!(cell.foreground, Color::Indexed(8) | Color::BrightBlack)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Osc133DecodeError {
    MalformedPercentEncoding,
    TooLong,
    InvalidUtf8,
}

/// The Start-identity slots an OSC 133 `C` packet may carry.
///
/// Grouped rather than passed as three more loose arguments because they are
/// only ever meaningful together: `jterm_core::execution_journal::ExecutionLifecycle`
/// refuses to exist unless all of them, plus a jsh-shaped execution id, arrived
/// on the same `C`. Each field is independently `None` when the slot was
/// absent, repeated, or failed its own validation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Osc133StartIdentity {
    session_id: Option<String>,
    seq: Option<u64>,
    started_at_ms: Option<u64>,
}

/// Streaming UTF-8 head+tail collector. It retains the full value while it
/// fits; only after the first overflow does it repartition into bounded head
/// and rolling tail storage.
struct BoundedTextBuilder {
    max_bytes: usize,
    total_bytes: usize,
    head: String,
    tail: std::collections::VecDeque<char>,
    tail_bytes: usize,
    tail_budget: usize,
    marker: &'static str,
    truncated: bool,
}

impl BoundedTextBuilder {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            total_bytes: 0,
            head: String::with_capacity(max_bytes.min(4096)),
            tail: std::collections::VecDeque::new(),
            tail_bytes: 0,
            tail_budget: 0,
            marker: "",
            truncated: false,
        }
    }

    fn push(&mut self, ch: char) {
        let ch_bytes = ch.len_utf8();
        self.total_bytes = self.total_bytes.saturating_add(ch_bytes);
        if !self.truncated && self.head.len().saturating_add(ch_bytes) <= self.max_bytes {
            self.head.push(ch);
            return;
        }

        if !self.truncated {
            self.truncated = true;
            self.marker = if self.max_bytes >= OUTPUT_TRUNCATION_MARKER.len() + 2 {
                OUTPUT_TRUNCATION_MARKER
            } else {
                ""
            };
            let payload_budget = self.max_bytes.saturating_sub(self.marker.len());
            let requested_head_bytes = payload_budget / 2;
            let mut split = requested_head_bytes.min(self.head.len());
            while !self.head.is_char_boundary(split) {
                split -= 1;
            }

            let previous = std::mem::take(&mut self.head);
            self.head.push_str(&previous[..split]);
            self.tail_budget = self
                .max_bytes
                .saturating_sub(self.marker.len())
                .saturating_sub(self.head.len());
            for previous_ch in previous[split..].chars() {
                self.push_tail(previous_ch);
            }
        }
        self.push_tail(ch);
    }

    fn push_tail(&mut self, ch: char) {
        let ch_bytes = ch.len_utf8();
        while self.tail_bytes.saturating_add(ch_bytes) > self.tail_budget {
            let Some(removed) = self.tail.pop_front() else {
                break;
            };
            self.tail_bytes = self.tail_bytes.saturating_sub(removed.len_utf8());
        }
        if ch_bytes <= self.tail_budget {
            self.tail.push_back(ch);
            self.tail_bytes = self.tail_bytes.saturating_add(ch_bytes);
        }
    }

    fn finish(mut self) -> ExtractedText {
        if self.truncated {
            self.head.push_str(self.marker);
            self.head.extend(self.tail);
        }
        debug_assert!(self.head.len() <= self.max_bytes);
        ExtractedText {
            text: self.head,
            truncated: self.truncated,
            total_bytes: self.total_bytes,
        }
    }
}

impl super::TerminalState {
    /// 解析 CSI 参数字节。
    ///
    /// 返回 `(params, colon_flags)`,其中 `colon_flags[k]` 表示参数 k 之前的
    /// 分隔符是否为冒号(子参数语法,如 `4:3`)。这样调用方可区分 `4:3`
    /// (扩展下划线样式)与 `4;3`(下划线 + 斜体两个独立 SGR)。
    ///
    /// 与 VT 规范一致:空字段默认为 0(`;5`→`[0,5]`、`5;`→`[5,0]`),
    /// 完全为空的参数串返回空向量(由各处理器使用各自默认值)。
    pub(super) fn parse_csi_params(
        param_bytes: &[u8],
    ) -> (SmallVec<[u16; 8]>, SmallVec<[bool; 8]>) {
        let mut params: SmallVec<[u16; 8]> = SmallVec::new();
        let mut colon_flags: SmallVec<[bool; 8]> = SmallVec::new();
        if param_bytes.is_empty() {
            return (params, colon_flags);
        }

        let mut current: u16 = 0;
        // 当前正在累积的参数之前的分隔符是否为冒号(首个参数无前导分隔符)
        let mut current_is_colon = false;

        for &byte in param_bytes {
            match byte {
                b'0'..=b'9' => {
                    current = current
                        .saturating_mul(10)
                        .saturating_add((byte - b'0') as u16);
                }
                b';' | b':' => {
                    params.push(current);
                    colon_flags.push(current_is_colon);
                    current = 0;
                    current_is_colon = byte == b':';
                }
                _ => {}
            }
        }
        params.push(current);
        colon_flags.push(current_is_colon);

        (params, colon_flags)
    }

    /// 默认每 8 列一个制表位。
    pub(super) fn default_tab_stops(cols: usize) -> Vec<bool> {
        (0..cols).map(|c| c % 8 == 0).collect()
    }

    /// 从给定列出发,返回下一个制表位的列(无则停在最后一列)。
    pub(super) fn next_tab_stop(&self, col: usize) -> usize {
        let cols = self.grid.row_len();
        let mut c = col + 1;
        while c < cols {
            if self.tab_stops.get(c).copied().unwrap_or(false) {
                return c;
            }
            c += 1;
        }
        cols.saturating_sub(1)
    }

    /// 从给定列出发,返回严格左侧的上一个制表位(无则停在第 0 列)。
    pub(super) fn prev_tab_stop(&self, col: usize) -> usize {
        let mut c = col;
        while c > 0 {
            c -= 1;
            if self.tab_stops.get(c).copied().unwrap_or(false) {
                return c;
            }
        }
        0
    }

    /// DECSC / CSI s:保存完整光标状态(含 SGR、字符集、模式)。
    pub(super) fn save_cursor_state(&mut self) {
        self.saved_state = Some(SavedCursorState {
            row: self.cursor_row,
            col: self.cursor_col,
            fg: self.current_fg,
            bg: self.current_bg,
            flags: self.current_flags,
            g0: self.g0_charset,
            g1: self.g1_charset,
            active: self.active_charset,
            origin_mode: self.origin_mode,
            autowrap: self.modes.contains(&7),
            pending_wrap: self.pending_wrap,
        });
    }

    /// DECRC / CSI u:恢复 save_cursor_state 保存的完整状态;
    /// 若从未保存过,按规范复位到原点。
    pub(super) fn restore_cursor_state(&mut self) {
        if let Some(s) = self.saved_state.clone() {
            self.cursor_row = s.row.min(self.grid.rows().saturating_sub(1));
            self.cursor_col = s.col.min(self.grid.row_len().saturating_sub(1));
            self.current_fg = s.fg;
            self.current_bg = s.bg;
            self.current_flags = s.flags;
            self.g0_charset = s.g0;
            self.g1_charset = s.g1;
            self.active_charset = s.active;
            self.origin_mode = s.origin_mode;
            if s.autowrap {
                self.modes.insert(7);
            } else {
                self.modes.remove(&7);
            }
            self.pending_wrap = s.pending_wrap;
        } else {
            self.cursor_row = 0;
            self.cursor_col = 0;
            self.pending_wrap = false;
        }
    }

    /// CUP/HVP 光标定位(1 基参数)。原点模式下行相对滚动区域顶端并限制在区域内。
    pub(super) fn set_cursor_position(&mut self, row_param: usize, col_param: usize) {
        self.pending_wrap = false;
        let col = col_param
            .saturating_sub(1)
            .min(self.grid.row_len().saturating_sub(1));
        let row0 = row_param.saturating_sub(1);
        self.cursor_row = if self.origin_mode {
            (self.scroll_region_top + row0).min(self.scroll_region_bottom)
        } else {
            row0.min(self.grid.rows().saturating_sub(1))
        };
        self.cursor_col = col;
    }

    pub fn new(cols: usize, rows: usize) -> Self {
        let (cols, rows) = clamp_terminal_dimensions(cols, rows);
        let mut grid = TerminalGrid::new(rows, cols);
        let mut alt_grid = TerminalGrid::new(rows, cols);
        let mut next_raw_row_id = 1u64;
        for id in grid.row_ids.iter_mut().chain(alt_grid.row_ids.iter_mut()) {
            *id = allocate_raw_row_id_from(&mut next_raw_row_id);
        }

        let mut modes = TerminalModes::default();
        modes.insert(25);
        modes.insert(7);

        let mut dirty_region = DirtyRegion::new();
        // Mark all rows as dirty on initialization to ensure first frame renders correctly
        dirty_region.mark_all(rows);
        let mut kitty_graphics = KittyGraphicsState::new();
        kitty_graphics.resize(cols, rows);

        TerminalState {
            grid,
            alt_grid,
            scrollback: VecDeque::new(),
            selection: None,
            projected_selection: None,
            selection_revision: 1,
            scroll_offset: 0,
            max_scrollback: 10000,
            use_alt_buffer: false,
            cursor_row: 0,
            cursor_col: 0,
            saved_cursor_row: 0,
            saved_cursor_col: 0,
            alt_cursor_row: 0,
            alt_cursor_col: 0,
            cursor_shape: CursorShape::default(),
            saved_state: None,
            saved_primary_screen_state: None,
            insert_mode: false,
            last_printed_char: None,
            origin_mode: false,
            tab_stops: Self::default_tab_stops(cols),
            pending_wrap: false,
            current_fg: Color::Default,
            current_bg: Color::Default,
            current_flags: StyleFlags::default(),
            window_title: String::new(),
            announced_jsh_session_id: None,
            current_working_dir: None,
            global_bg: Color::Default,
            utf8_buf: [0; 4],
            utf8_len: 0,
            utf8_expected: 0,
            pending_escape: Vec::new(),
            pending_apc: Vec::new(),
            pending_apc_scan_from: 0,
            discarding_oversized_apc: false,
            discarding_apc_prev_escape: false,
            pending_osc: Vec::new(),
            pending_osc_scan_from: 0,
            pending_string: Vec::new(),
            pending_string_scan_from: 0,
            g0_charset: Charset::Ascii,
            g1_charset: Charset::Ascii,
            active_charset: Charset::Ascii,
            ime_enabled: false,
            preedit_text: String::new(),
            preedit_cursor: 0,
            scroll_region_top: 0,
            scroll_region_bottom: rows.saturating_sub(1),
            modes,
            output_buffer: Vec::new(),
            keyboard_enhancement_flags: 0,
            keyboard_enhancement_stack: Vec::new(),
            alt_keyboard_enhancement_flags: 0,
            alt_keyboard_enhancement_stack: Vec::new(),
            xterm_modify_other_keys: 0,
            xterm_format_other_keys: 0,
            pending_clipboard_requests: Vec::new(),
            pending_paste_grant: None,
            kitty_graphics,
            dirty_region,
            grid_version: 1,
            // IMPORTANT: row_versions must match grid.rows(), not the parameter 'rows'
            // This ensures dirty tracking works correctly even with scrollback
            row_versions: vec![1; rows], // Use 'rows' here since grid.rows() == rows at init
            visible_cells_cache: None,
            projected_viewport_cache: None,
            projection_plan_cache: None,
            transformed_viewport_cache: None,
            next_projection_plan_revision: 1,
            viewport_mapping_exact_cache: std::cell::Cell::new(None),
            hyperlinks: hyperlink::HyperlinkTable::default(),
            current_hyperlink: HyperlinkId::NONE,
            sync_output_active: false,
            sync_output_start: None,
            last_archived_screen_snapshot: Vec::new(),
            last_synced_primary_screen_snapshot: Vec::new(),
            pending_osc52_clipboard_set: None,
            pending_osc52_clipboard_query: false,
            dynamic_fg: None,
            dynamic_bg: None,
            dynamic_cursor_color: None,
            dynamic_palette: [None; 256],
            pending_notifications: Vec::new(),
            total_lines_scrolled: 0,
            next_raw_row_id,
            row_identity_revision: 1,
            full_screen_scroll_revision: 1,
            command_marks: VecDeque::new(),
            command_records: VecDeque::new(),
            next_command_sequence: 1,
            consumed_command_ids: VecDeque::new(),
            finished_output_provenance: HashMap::new(),
            finished_output_owners: HashMap::new(),
            finished_output_revision: 1,
            active_output_provenance: None,
            pending_completed_command_outputs: VecDeque::new(),
            captured_command_output_bytes: 0,
            cleared_blocks_stash: None,
            agent_prompt_input_tainted: false,
            armed_agent_execution: None,
        }
    }

    pub(super) fn decode_base64(value: &str) -> Option<String> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(value)
            .ok()?;
        String::from_utf8(bytes).ok()
    }

    pub(super) fn osc_terminator() -> &'static [u8] {
        b"\x1b\\"
    }

    pub(super) fn append_osc_5522_status(&mut self, metadata: &str, payload: Option<&str>) {
        self.output_buffer.extend_from_slice(b"\x1b]5522;");
        self.output_buffer.extend_from_slice(metadata.as_bytes());
        if let Some(payload) = payload {
            self.output_buffer.extend_from_slice(b";");
            self.output_buffer.extend_from_slice(payload.as_bytes());
        }
        self.output_buffer.extend_from_slice(Self::osc_terminator());
    }

    pub(super) fn handle_osc_color(&mut self, command: &str, value: &str) {
        if value == "?" {
            // Query: respond with current color
            let color = match command {
                "10" => self.dynamic_fg.unwrap_or((255, 255, 255)),
                "11" => self.dynamic_bg.unwrap_or((0, 0, 0)),
                "12" => self.dynamic_cursor_color.unwrap_or((255, 255, 255)),
                _ => return,
            };
            let response = format!(
                "\x1b]{};rgb:{:04x}/{:04x}/{:04x}\x1b\\",
                command,
                (color.0 as u16) * 257,
                (color.1 as u16) * 257,
                (color.2 as u16) * 257,
            );
            self.output_buffer.extend_from_slice(response.as_bytes());
        } else if let Some(rgb) = Self::parse_color_spec(value) {
            match command {
                "10" => self.dynamic_fg = Some(rgb),
                "11" => self.dynamic_bg = Some(rgb),
                "12" => self.dynamic_cursor_color = Some(rgb),
                _ => {}
            }
        }
    }

    /// OSC 110/111/112: reset one dynamic color back to the theme default.
    pub(super) fn reset_osc_color(&mut self, command: &str) {
        match command {
            "110" => self.dynamic_fg = None,
            "111" => self.dynamic_bg = None,
            "112" => self.dynamic_cursor_color = None,
            _ => {}
        }
    }

    /// OSC 4: set or query 256-palette entries (`idx;spec` pairs; `?` queries).
    pub(super) fn handle_osc_palette(&mut self, value: &str) {
        let mut parts = value.split(';');
        while let Some(idx_s) = parts.next() {
            let Some(color_s) = parts.next() else {
                break;
            };
            let Ok(idx) = idx_s.parse::<u8>() else {
                continue;
            };
            if color_s == "?" {
                let color = self.dynamic_palette[idx as usize]
                    .unwrap_or_else(|| Self::default_256_color(idx));
                self.append_osc_palette_response(idx, color);
            } else if let Some(rgb) = Self::parse_color_spec(color_s) {
                self.dynamic_palette[idx as usize] = Some(rgb);
            }
        }
    }

    /// OSC 104: reset the whole palette (empty payload) or the listed indices.
    pub(super) fn reset_osc_palette(&mut self, value: &str) {
        if value.is_empty() {
            self.dynamic_palette = [None; 256];
            return;
        }
        for idx_s in value.split(';').filter(|s| !s.is_empty()) {
            if let Ok(idx) = idx_s.parse::<u8>() {
                self.dynamic_palette[idx as usize] = None;
            }
        }
    }

    fn append_osc_palette_response(&mut self, idx: u8, color: (u8, u8, u8)) {
        let response = format!(
            "\x1b]4;{};rgb:{:04x}/{:04x}/{:04x}\x1b\\",
            idx,
            (color.0 as u16) * 257,
            (color.1 as u16) * 257,
            (color.2 as u16) * 257,
        );
        self.output_buffer.extend_from_slice(response.as_bytes());
    }

    /// Standard xterm defaults for palette queries when no override is set.
    fn default_256_color(idx: u8) -> (u8, u8, u8) {
        const ANSI: [(u8, u8, u8); 16] = [
            (0, 0, 0),
            (205, 0, 0),
            (0, 205, 0),
            (205, 205, 0),
            (0, 0, 238),
            (205, 0, 205),
            (0, 205, 205),
            (229, 229, 229),
            (127, 127, 127),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (92, 92, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
        ];
        match idx {
            0..=15 => ANSI[idx as usize],
            16..=231 => {
                let idx = idx - 16;
                let r_idx = idx / 36;
                let g_idx = (idx % 36) / 6;
                let b_idx = idx % 6;
                let scale = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
                (scale(r_idx), scale(g_idx), scale(b_idx))
            }
            232..=255 => {
                let gray = 8 + (idx - 232) * 10;
                (gray, gray, gray)
            }
        }
    }

    /// An OSC 7 payload longer than this cannot percent-decode to a directory
    /// the shared cwd validator would accept: `%XX` is the densest encoding at
    /// three raw bytes per decoded byte, so three times the 4 KiB cwd budget
    /// plus a `file://<host>` prefix already exceeds any acceptable path. The
    /// gate is checked before decoding, so a pending-escape payload of several
    /// megabytes cannot buy an allocation of the same size on the PTY thread.
    /// Same number as the shared parser's own OSC 7 budget.
    pub(super) const MAX_OSC7_URI_BYTES: usize = 16 * 1024;

    /// Decode one OSC 7 working-directory announcement to a local filesystem
    /// path. Accepts either `file://host/path` (whose path is percent-encoded)
    /// or a raw path, and returns `None` when the payload is empty or
    /// malformed. A non-local hostname is rejected: persisting an SSH server's
    /// `/etc` as a local cwd would restore the next session in the wrong host
    /// directory.
    ///
    /// The result is not display text: it is cloned onto every command record,
    /// shaped into the pane header and bottom bar each frame, and handed to
    /// `Pty::new_with_pinned_cwd` as the working directory of a session split
    /// from this pane. It was stored with no length bound and no character
    /// rule at all, so a PTY could park megabytes in terminal state and could
    /// name a directory whose displayed spelling — via a bidi override or a
    /// zero-width joiner — is not the directory a new shell would start in.
    /// [`jterm_core::execution_journal::is_valid_jsh_cwd`] is the family's one
    /// answer to "a cwd this terminal will record": 4 KiB, non-empty, no
    /// controls, no visual spoofing. The OSC 133 cwd path already uses it.
    pub(super) fn decode_osc7_cwd(value: &str) -> Option<String> {
        if value.len() > Self::MAX_OSC7_URI_BYTES {
            return None;
        }
        let path_part = if let Some(rest) = value.strip_prefix("file://") {
            let slash = rest.find('/')?;
            let host = &rest[..slash];
            if !Self::osc7_host_is_local(host) {
                return None;
            }
            &rest[slash..]
        } else if value.starts_with('/') {
            value
        } else {
            return None;
        };
        // Percent-decode. We don't pull in a url crate — the alphabet is small.
        let bytes = path_part.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                let byte = u8::from_str_radix(hex, 16).ok()?;
                out.push(byte);
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out)
            .ok()
            .filter(|cwd| jterm_core::execution_journal::is_valid_jsh_cwd(cwd))
    }

    fn osc7_host_is_local(host: &str) -> bool {
        if host.is_empty() || host.eq_ignore_ascii_case("localhost") {
            return true;
        }
        // Resolved once per process, not once per packet. This runs on the PTY
        // parse loop, and every OSC 7 naming any other host — which a remote
        // shell emits on every prompt — opened and read `/etc/hostname` again.
        // The machine's own name does not change under a running window, and a
        // failed read stays failed rather than being retried per keystroke.
        static LOCAL_HOSTNAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let local_hostname = LOCAL_HOSTNAME.get_or_init(|| {
            std::env::var("HOSTNAME").ok().or_else(|| {
                std::fs::read_to_string("/etc/hostname")
                    .ok()
                    .map(|hostname| hostname.trim().to_string())
            })
        });
        local_hostname
            .as_deref()
            .is_some_and(|local| host.eq_ignore_ascii_case(local))
    }

    pub(super) fn parse_color_spec(spec: &str) -> Option<(u8, u8, u8)> {
        // Parse rgb:RR/GG/BB / rgb:RRRR/GGGG/BBBB / rgb:R/G/B / rgb:RRR/GGG/BBB / #RRGGBB
        // Per XParseColor, each component is 1..=4 hex digits and is left-aligned
        // into a 16-bit field (i.e. component value * (2^16-1) / (2^bits-1)),
        // then truncated to 8 bits. The previous scale=1 fallback for 1/3-digit
        // components produced a u8 cast of e.g. 0xFFF=4095, which wraps to 255
        // for full-on but is wrong for any partial value.
        fn scale_to_u8(value: u16, hex_digits: usize) -> u8 {
            // Range of n hex digits is [0, 16^n - 1].
            let max_n: u32 = (1u32 << (hex_digits * 4)).saturating_sub(1).max(1);
            // Scale value to 0..=255, rounding to nearest.
            (((value as u32) * 255 + max_n / 2) / max_n) as u8
        }
        if let Some(hex) = spec.strip_prefix('#') {
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                return Some((r, g, b));
            }
        } else if let Some(rgb) = spec.strip_prefix("rgb:") {
            let parts: Vec<&str> = rgb.split('/').collect();
            if parts.len() == 3
                && (1..=4).contains(&parts[0].len())
                && parts[0].len() == parts[1].len()
                && parts[1].len() == parts[2].len()
            {
                let digits = parts[0].len();
                let r = u16::from_str_radix(parts[0], 16).ok()?;
                let g = u16::from_str_radix(parts[1], 16).ok()?;
                let b = u16::from_str_radix(parts[2], 16).ok()?;
                return Some((
                    scale_to_u8(r, digits),
                    scale_to_u8(g, digits),
                    scale_to_u8(b, digits),
                ));
            }
        }
        None
    }

    pub(super) fn handle_osc_52(&mut self, value: &str) {
        // OSC 52 format: <selection>;<base64-data>
        // selection: c=clipboard, p=primary, s=select (we treat all as clipboard)
        // data: ? means query, base64 means set.
        //
        // Cap on payload size: a remote process should not be able to push
        // arbitrary multi-MB blobs into the host clipboard. xterm uses 100 KB
        // by default; we match that.
        const OSC52_MAX_BYTES: usize = 100 * 1024;
        if let Some((_sel, data)) = value.split_once(';') {
            if data == "?" {
                self.pending_osc52_clipboard_query = true;
            } else if !data.is_empty() {
                if data.len() > OSC52_MAX_BYTES.saturating_mul(4) / 3 + 8 {
                    // Reject before even attempting to decode.
                    crate::debug_log!(
                        "[OSC52] rejecting clipboard set: encoded {} bytes exceeds limit",
                        data.len()
                    );
                    return;
                }
                if let Some(decoded) = Self::decode_base64(data) {
                    if decoded.len() <= OSC52_MAX_BYTES {
                        self.pending_osc52_clipboard_set = Some(decoded);
                    } else {
                        crate::debug_log!(
                            "[OSC52] rejecting clipboard set: decoded {} bytes exceeds {}",
                            decoded.len(),
                            OSC52_MAX_BYTES
                        );
                    }
                }
            }
        }
    }

    pub(super) fn handle_osc_5522(&mut self, metadata: &str, payload: Option<&str>) {
        crate::debug_log!("[OSC5522] metadata={} payload={:?}", metadata, payload);

        let mut message_type = None;
        let mut password = None;
        let mut human_name = None;

        for part in metadata.split(':') {
            if let Some(value) = part.strip_prefix("type=") {
                message_type = Some(value);
            } else if let Some(value) = part.strip_prefix("pw=") {
                password = Self::decode_base64(value);
            } else if let Some(value) = part.strip_prefix("name=") {
                human_name = Self::decode_base64(value);
            }
        }

        if message_type != Some("read") {
            return;
        }

        // Per the OSC 5522 protocol, the third field is base64-encoded and is
        // either "." (list types) or a space-separated MIME request. This
        // implementation intentionally supports one MIME per paste grant for
        // now; accepting arbitrary direct reads would require a permission UI.
        let Some(request) = payload.and_then(Self::decode_base64) else {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        };

        if request == "." {
            // Clipboard type enumeration is itself a host read. ember sends
            // the sanitized MIME list proactively only after an actual user
            // paste, together with a short-lived capability. A PTY-originated
            // discovery request must not spawn host clipboard helpers.
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }

        let mut requested_mimes = request.split_ascii_whitespace();
        let Some(mime_type) = requested_mimes.next() else {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        };
        if requested_mimes.next().is_some() || !Self::is_valid_osc_5522_mime(mime_type) {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }

        if !self.is_paste_events_enabled() {
            self.pending_paste_grant = None;
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }

        let Some(grant) = self.pending_paste_grant.as_ref() else {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        };
        if std::time::Instant::now() >= grant.expires_at {
            self.pending_paste_grant = None;
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }
        if password.as_deref() != Some(grant.token.as_str())
            || human_name.as_deref() != Some("Paste event")
        {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }

        // Consume the capability before any asynchronous clipboard I/O is
        // queued. A valid token is single-use and cannot be replayed or raced.
        let Some(grant) = self.pending_paste_grant.take() else {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        };
        if !grant.offered_mimes.contains(mime_type) {
            self.append_osc_5522_status("type=read:status=EPERM", None);
            return;
        }

        self.queue_clipboard_request(ClipboardReadKind::MimeData(mime_type.to_string()));
    }

    fn queue_clipboard_request(&mut self, kind: ClipboardReadKind) {
        if self.pending_clipboard_requests.len() >= MAX_PENDING_CLIPBOARD_REQUESTS {
            self.append_osc_5522_status("type=read:status=EBUSY", None);
            return;
        }
        self.pending_clipboard_requests
            .push(ClipboardReadRequest { kind });
    }

    fn is_valid_osc_5522_mime(mime: &str) -> bool {
        !mime.is_empty()
            && mime.len() <= MAX_OSC_5522_MIME_LEN
            && mime.is_ascii()
            && !mime
                .bytes()
                .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    }

    pub(super) fn set_keyboard_enhancement_flags(&mut self, flags: u16, mode: u16) {
        match mode {
            1 => self.keyboard_enhancement_flags = flags,
            2 => self.keyboard_enhancement_flags |= flags,
            3 => self.keyboard_enhancement_flags &= !flags,
            _ => {}
        }
    }

    pub(super) fn push_keyboard_enhancement_flags(&mut self, flags: u16) {
        if self.keyboard_enhancement_stack.len() >= 32 {
            self.keyboard_enhancement_stack.remove(0);
        }
        self.keyboard_enhancement_stack
            .push(self.keyboard_enhancement_flags);
        self.keyboard_enhancement_flags = flags;
    }

    pub(super) fn pop_keyboard_enhancement_flags(&mut self, count: usize) {
        for _ in 0..count.max(1) {
            match self.keyboard_enhancement_stack.pop() {
                Some(flags) => self.keyboard_enhancement_flags = flags,
                None => {
                    self.keyboard_enhancement_flags = 0;
                    break;
                }
            }
        }
    }

    /// Compose a combining mark onto the most recently written cell using NFC.
    /// Only single-codepoint compositions are applied (the cell stores one char).
    pub(super) fn apply_combining_mark(&mut self, mark: char) {
        // After a char fills the last column, the wrap is deferred: the cursor
        // stays *on* the last column with pending_wrap set, so the base glyph is
        // at cursor_col itself rather than to its left.
        let mut base_col = if self.pending_wrap {
            self.cursor_col
        } else if self.cursor_col == 0 {
            return;
        } else {
            self.cursor_col - 1
        };
        if self
            .grid
            .get(self.cursor_row, base_col)
            .flags
            .wide_continuation()
        {
            if base_col == 0 {
                return;
            }
            base_col -= 1;
        }

        let base = self.grid.get(self.cursor_row, base_col).character;
        if base == ' ' || base == '\0' {
            return;
        }

        let mut composed = String::with_capacity(2);
        composed.push(base);
        composed.push(mark);
        let nfc: String =
            unicode_normalization::UnicodeNormalization::nfc(composed.as_str()).collect();
        let mut chars = nfc.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            if c != base {
                self.invalidate_finished_output_span(
                    self.cursor_row,
                    base_col,
                    base_col.saturating_add(1),
                );
                self.grid.get_mut(self.cursor_row, base_col).character = c;
                self.dirty_region.mark_row(self.cursor_row);
                self.mark_row_dirty(self.cursor_row);
                let row_id = self.grid.row_id(self.cursor_row);
                self.note_output_write(
                    self.cursor_row,
                    row_id,
                    base_col,
                    base_col.saturating_add(1),
                );
            }
        }
    }

    pub(super) fn put_char(&mut self, ch: char) {
        let _orig_ch = ch;
        let ch = self.translate_char(ch);
        let width = crate::char_width::cached_char_width(ch);
        if width == 0 {
            // Zero-width characters are combining marks. Try to compose the mark
            // onto the previously written cell (NFC). Marks with no precomposed
            // single-codepoint form (e.g. stacked diacritics) are dropped, as the
            // cell can only hold one char.
            self.apply_combining_mark(ch);
            return;
        }

        let cols = self.grid.row_len();
        let blank_cell = self.create_blank_cell();
        let autowrap = self.modes.contains(&7);

        // Resolve a wrap deferred by the previous character (DEC Last Column Flag).
        // The actual line break happens here, when the next printable char arrives.
        if self.pending_wrap {
            self.pending_wrap = false;
            if autowrap {
                self.wrap_to_next_line();
            }
        }

        // A wide character that does not fit in the columns left on this line wraps
        // immediately (it cannot be split across the line boundary).
        if self.cursor_col + width > cols {
            if autowrap {
                self.wrap_to_next_line();
            } else {
                // Autowrap disabled: clamp cursor to last column instead of wrapping
                self.cursor_col = cols.saturating_sub(width);
            }
        }

        // IRM (insert mode, ANSI mode 4): shift existing cells right by `width`
        // before writing, discarding cells pushed past the end of the row.
        let mutation_end = if self.insert_mode {
            cols
        } else {
            self.cursor_col.saturating_add(width)
        };
        let (mutation_start, mutation_end) =
            self.expanded_cell_mutation_span(self.cursor_row, self.cursor_col, mutation_end);
        self.invalidate_finished_output_span(self.cursor_row, mutation_start, mutation_end);

        if self.insert_mode {
            for _ in 0..width {
                if self.cursor_col < cols {
                    self.grid
                        .insert_cell_in_row(self.cursor_row, self.cursor_col, blank_cell);
                }
            }
            self.grid
                .clear_dangling_wide_at_row_end(self.cursor_row, blank_cell);
        }

        // Overwriting either half of a double-width character must clear the
        // other half. `width` covers both the cell written and, for a wide
        // character, the cell claimed as its own continuation below: landing
        // that continuation on somebody else's lead orphans *its* continuation
        // one column further right.
        self.split_wide_pairs_around(self.cursor_row, self.cursor_col, self.cursor_col + width);

        let output_row = self.cursor_row;
        let output_col = self.cursor_col;
        let output_row_id = self.grid.row_id(output_row);

        // Write character
        let cell = self.grid.get_mut(self.cursor_row, self.cursor_col);
        cell.character = ch;
        cell.foreground = self.current_fg;
        cell.background = self.current_bg;
        cell.flags = self.current_flags;
        cell.hyperlink_id = self.current_hyperlink;
        cell.flags.set_wide(width == 2);
        cell.flags.set_wide_continuation(false);

        // Set up wide character continuation cell if needed
        if width == 2 && self.cursor_col + 1 < cols {
            let cont_cell = self.grid.get_mut(self.cursor_row, self.cursor_col + 1);
            *cont_cell = blank_cell;
            cont_cell.hyperlink_id = self.current_hyperlink;
            cont_cell.flags.set_wide_continuation(true);
        }

        self.cursor_col += width;
        // If we just filled the last cell, defer the wrap: keep the cursor on the
        // last column and set the Last Column Flag. The wrap fires on the next char.
        if self.cursor_col >= cols {
            if autowrap {
                self.cursor_col = cols.saturating_sub(width);
                self.pending_wrap = true;
            } else {
                self.cursor_col = cols.saturating_sub(width);
            }
        }
        // Mark the row as dirty after writing character
        self.dirty_region.mark_row(self.cursor_row);
        self.mark_row_dirty(self.cursor_row);
        self.note_output_write(
            output_row,
            output_row_id,
            output_col,
            output_col.saturating_add(width),
        );
        self.last_printed_char = Some(_orig_ch);
    }

    pub(super) fn put_ascii_run(&mut self, bytes: &[u8]) {
        // Insert mode (IRM) needs per-character right-shifting, which the fast
        // overwrite path below does not do. Fall back to put_char for each byte.
        if self.insert_mode {
            for &byte in bytes {
                self.put_char(byte as char);
            }
            return;
        }

        let cols = self.grid.row_len();
        let autowrap = self.modes.contains(&7);
        let mut pos = 0;

        while pos < bytes.len() {
            // 先结算上一次写入遗留的延迟换行(DEC 末列标志)。
            if self.pending_wrap {
                self.pending_wrap = false;
                if autowrap {
                    self.wrap_to_next_line();
                }
            }

            let remaining = cols - self.cursor_col;
            let chunk_len = (bytes.len() - pos).min(remaining);

            // Write chunk to grid directly through a single row slice
            // (avoids recomputing row*cols + bounds-check on every cell)
            let fg = self.current_fg;
            let bg = self.current_bg;
            let hyperlink_id = self.current_hyperlink;
            let mut flags = self.current_flags;
            flags.set_wide(false);
            flags.set_wide_continuation(false);
            let col = self.cursor_col;
            let output_row = self.cursor_row;
            let output_row_id = self.grid.row_id(output_row);
            // This chunk overwrites a whole span at once, so — exactly as
            // put_char does per cell — clear the double-width halves it leaves
            // stranded on either side. Without this, ASCII typed over CJK left
            // an orphaned continuation cell that the renderer skips (the
            // character reads as empty background) and that the next write
            // there blanks its left neighbour to repair.
            let (mutation_start, mutation_end) =
                self.expanded_cell_mutation_span(self.cursor_row, col, col + chunk_len);
            self.invalidate_finished_output_span(self.cursor_row, mutation_start, mutation_end);
            self.split_wide_pairs_around(self.cursor_row, col, col + chunk_len);
            let row = &mut self.grid[self.cursor_row][col..col + chunk_len];
            for (cell, &byte) in row.iter_mut().zip(&bytes[pos..pos + chunk_len]) {
                cell.character = byte as char;
                cell.foreground = fg;
                cell.background = bg;
                cell.flags = flags;
                cell.hyperlink_id = hyperlink_id;
            }

            self.cursor_col += chunk_len;
            pos += chunk_len;

            self.dirty_region.mark_row(self.cursor_row);
            self.mark_row_dirty(self.cursor_row);
            self.note_output_write(
                output_row,
                output_row_id,
                col,
                col.saturating_add(chunk_len),
            );

            // 写满末列时不立即换行,改为置延迟换行标志,
            // 光标停在末列,等待下一个可打印字符再决定是否换行。
            if self.cursor_col >= cols {
                self.cursor_col = cols - 1;
                if autowrap {
                    self.pending_wrap = true;
                }
            }
        }
        // The fast path bypasses put_char, so REP's target is recorded here too.
        if let Some(&last) = bytes.last() {
            self.last_printed_char = Some(last as char);
        }
    }

    /// 自动换行时推进到下一行,受 DECSTBM 滚动区底边距约束(与 LF 行为一致)。
    /// 恰在底边距时区域上滚(全屏区会把顶行压入 scrollback);否则在网格内下移光标。
    /// 此前换行只与 grid.rows() 比较并调用全屏 scroll_down(),会让区内文本溢出到
    /// 底边距下方,破坏 pager/分屏 TUI 布局。
    pub(super) fn wrap_to_next_line(&mut self) {
        let departed_row_id = self.grid.row_id(self.cursor_row);
        self.grid.row_wrapped[self.cursor_row] = true;
        self.cursor_col = 0;
        if self.cursor_row == self.scroll_region_bottom {
            self.scroll_region_up(self.scroll_region_top, self.scroll_region_bottom);
        } else if self.cursor_row + 1 < self.grid.rows() {
            self.cursor_row += 1;
        }
        self.rebase_active_output_start_after_initial_wrap(departed_row_id);
    }

    pub(super) fn create_blank_cell(&self) -> TerminalCell {
        TerminalCell {
            character: ' ',
            foreground: Color::Default,
            background: self.current_bg, // Preserve current background color
            flags: StyleFlags::default(),
            hyperlink_id: HyperlinkId::NONE,
        }
    }

    pub(super) fn blank_line(&self, cols: usize) -> Vec<TerminalCell> {
        vec![self.create_blank_cell(); cols]
    }

    pub(super) fn normalize_line_width(
        &self,
        mut line: Vec<TerminalCell>,
        cols: usize,
    ) -> Vec<TerminalCell> {
        match line.len().cmp(&cols) {
            std::cmp::Ordering::Equal => line,
            std::cmp::Ordering::Greater => {
                line.truncate(cols);
                line
            }
            std::cmp::Ordering::Less => {
                line.resize(cols, self.create_blank_cell());
                line
            }
        }
    }

    pub(super) fn line_is_blank(&self, row: usize) -> bool {
        let blank = self.create_blank_cell();
        self.grid[row].iter().all(|cell| {
            cell.character == blank.character
                && cell.foreground == blank.foreground
                && cell.background == blank.background
                && cell.flags == blank.flags
                && cell.hyperlink_id == blank.hyperlink_id
        })
    }

    pub(super) fn archive_visible_screen_to_scrollback(&mut self) {
        self.archive_visible_screen_to_scrollback_with_options(false, false);
    }

    pub(super) fn visible_screen_snapshot(&self) -> Option<Vec<String>> {
        if self.grid.rows() == 0 {
            return None;
        }

        let first = (0..self.grid.rows()).find(|&row| !self.line_is_blank(row));
        let last = (0..self.grid.rows()).rfind(|&row| !self.line_is_blank(row));
        let (Some(first), Some(last)) = (first, last) else {
            return None;
        };

        Some(
            (first..=last)
                .map(|row| self.grid[row].iter().map(|cell| cell.character).collect())
                .collect(),
        )
    }

    pub(super) fn archive_primary_screen_unless_last_synced_snapshot(&mut self) {
        let Some(snapshot) = self.visible_screen_snapshot() else {
            return;
        };

        if snapshot == self.last_synced_primary_screen_snapshot {
            return;
        }

        self.archive_visible_screen_to_scrollback();
    }

    pub(super) fn archive_visible_screen_to_scrollback_with_options(
        &mut self,
        allow_alt_buffer: bool,
        dedupe_snapshot: bool,
    ) {
        if (self.use_alt_buffer && !allow_alt_buffer) || self.grid.rows() == 0 {
            return;
        }

        let first = (0..self.grid.rows()).find(|&row| !self.line_is_blank(row));
        let last = (0..self.grid.rows()).rfind(|&row| !self.line_is_blank(row));
        let (Some(first), Some(last)) = (first, last) else {
            return;
        };

        if dedupe_snapshot {
            let snapshot = self.visible_screen_snapshot().unwrap_or_default();
            if snapshot == self.last_archived_screen_snapshot {
                return;
            }
            self.last_archived_screen_snapshot = snapshot;
        }

        // A synchronized TUI frame is copied into scrollback while the live
        // grid stays in place. Raw selections use `scrollback + grid` row
        // coordinates, so history growth must move every live-grid anchor by
        // the same amount. Codex redraws this way on every frame.
        let old_grid_base = self.scrollback.len();
        for row in first..=last {
            let line = ScrollbackLine::compress(&self.grid[row], self.grid.row_wrapped[row]);
            self.push_scrollback_compressed_with_options(line, allow_alt_buffer);
        }
        self.rebase_raw_selection_for_grid_base_change(old_grid_base, self.scrollback.len());
    }

    fn rebase_raw_selection_for_grid_base_change(&mut self, old_base: usize, new_base: usize) {
        let Some(selection) = self.selection.as_mut() else {
            return;
        };
        for point in [&mut selection.anchor, &mut selection.active] {
            if point.0 < old_base {
                continue;
            }
            point.0 = if new_base >= old_base {
                point.0.saturating_add(new_base - old_base)
            } else {
                point.0.saturating_sub(old_base - new_base)
            };
        }
        self.bump_selection_revision();
    }

    #[allow(dead_code)] // Test/internal copy entrypoint; production uses options explicitly.
    pub(super) fn push_scrollback_compressed(&mut self, line: ScrollbackLine) {
        self.push_scrollback_compressed_with_options(line, false);
    }

    pub(super) fn fresh_raw_row_id(&mut self) -> RawRowId {
        allocate_raw_row_id_from(&mut self.next_raw_row_id)
    }

    fn mark_row_identity_changed_preserving_scroll_journal(&mut self) {
        self.row_identity_revision = self.row_identity_revision.wrapping_add(1);
        self.visible_cells_cache = None;
        self.projected_viewport_cache = None;
        // Keep the stale full-document plan available for the strictly
        // validated full-screen-scroll append fast path. Other identity
        // changes miss its exact key and replace it with a full rebuild.
        self.transformed_viewport_cache = None;
    }

    pub(super) fn mark_row_identity_changed(&mut self) {
        self.mark_row_identity_changed_preserving_scroll_journal();
    }

    fn fill_untracked_grid_row_ids(&mut self) {
        for row in 0..self.grid.rows() {
            if !self.grid.row_id(row).is_tracked() {
                let id = self.fresh_raw_row_id();
                self.grid.row_ids[row] = id;
            }
        }
        for row in 0..self.alt_grid.rows() {
            if !self.alt_grid.row_id(row).is_tracked() {
                let id = self.fresh_raw_row_id();
                self.alt_grid.row_ids[row] = id;
            }
        }
    }

    pub(super) fn shift_row_id_region_down(&mut self, top: usize, bottom: usize, count: usize) {
        if top > bottom || bottom >= self.grid.rows() || count == 0 {
            return;
        }
        let count = count.min(bottom - top + 1);
        if count < bottom - top + 1 {
            self.grid
                .row_ids
                .copy_within(top..=bottom - count, top + count);
        }
        for row in top..top + count {
            let id = self.fresh_raw_row_id();
            self.grid.row_ids[row] = id;
        }
        self.mark_row_identity_changed();
    }

    pub(super) fn shift_row_id_region_up(&mut self, top: usize, bottom: usize, count: usize) {
        if top > bottom || bottom >= self.grid.rows() || count == 0 {
            return;
        }
        let count = count.min(bottom - top + 1);
        if count < bottom - top + 1 {
            self.grid.row_ids.copy_within(top + count..=bottom, top);
        }
        for row in bottom + 1 - count..=bottom {
            let id = self.fresh_raw_row_id();
            self.grid.row_ids[row] = id;
        }
        self.mark_row_identity_changed_preserving_scroll_journal();
    }

    #[inline]
    pub(super) fn invalidate_scrollback_view_cache(&mut self) {
        self.visible_cells_cache = None;
        self.projected_viewport_cache = None;
        self.transformed_viewport_cache = None;
        self.viewport_mapping_exact_cache.set(None);
    }

    pub(super) fn push_scrollback_compressed_with_options(
        &mut self,
        mut line: ScrollbackLine,
        allow_alt_buffer: bool,
    ) {
        if self.use_alt_buffer && !allow_alt_buffer {
            return;
        }
        // Archival/snapshot callers copy a row rather than move it. Always
        // allocate a distinct identity so the same origin cannot resolve at
        // both the snapshot and its still-live source.
        line.set_raw_row_id(self.fresh_raw_row_id());
        self.push_scrollback_line(line, allow_alt_buffer);
    }

    fn push_scrollback_transferred_with_options(
        &mut self,
        line: ScrollbackLine,
        allow_alt_buffer: bool,
    ) {
        // A real top-margin scroll moves the physical row into scrollback;
        // preserve even UNTRACKED so allocator exhaustion stays fail-closed.
        self.push_scrollback_line(line, allow_alt_buffer);
    }

    fn push_scrollback_line(&mut self, line: ScrollbackLine, allow_alt_buffer: bool) {
        if self.use_alt_buffer && !allow_alt_buffer {
            return;
        }
        if self.scrollback.len() >= self.max_scrollback {
            if let Some(evicted) = self.scrollback.pop_front() {
                self.forget_output_provenance_for_raw_row(evicted.raw_row_id());
            }
        }
        self.scrollback.push_back(line);
        self.total_lines_scrolled = self.total_lines_scrolled.saturating_add(1);
        // Pin the viewport while the user is reading history. `scroll_offset`
        // is a distance from the BOTTOM of scrollback, so every push moves the
        // text under a stationary offset: start_idx = scrollback.len() -
        // (scroll_offset + rows) drifts by one row per line, and a build or a
        // tail walks the reader back to the live edge one line at a time.
        // Past the scrollback cap the increment is still right — eviction
        // shifts content indices down by one while len() stays put — and the
        // clamp only bites when the reader is already at the very top, where
        // the line they were looking at has genuinely been evicted.
        if self.scroll_offset > 0 {
            self.scroll_offset = (self.scroll_offset + 1).min(self.scrollback.len());
        }
        self.mark_row_identity_changed_preserving_scroll_journal();
        self.invalidate_scrollback_view_cache();
    }

    pub(super) fn scroll_region_down(&mut self, top: usize, bottom: usize) {
        if top >= self.grid.rows() || bottom >= self.grid.rows() || top > bottom {
            return;
        }
        // The bottom physical row is discarded, while every other RawRowId is
        // merely moved. Remove ownership before the copy so the reverse index
        // cannot retain a zone whose protected cells no longer exist.
        self.forget_output_provenance_for_grid_rows(bottom, bottom);
        let cols = self.grid.row_len();
        // Shift rows down: move [top..bottom) to [top+1..=bottom]
        let src_start = top * cols;
        let src_end = bottom * cols;
        let dst = (top + 1) * cols;
        self.grid.cells.copy_within(src_start..src_end, dst);
        // Clear top row(保留当前背景色 / BCE)
        let blank = self.create_blank_cell();
        self.grid.cells[src_start..src_start + cols].fill(blank);
        self.grid.row_wrapped.copy_within(top..bottom, top + 1);
        self.grid.row_wrapped[top] = false;
        self.shift_row_id_region_down(top, bottom, 1);
        self.kitty_graphics.scroll_region_down(top, bottom, 1);
        self.dirty_region.mark_rows(top, bottom);
        self.mark_rows_dirty(top, bottom);
    }

    pub(super) fn scroll_region_up(&mut self, top: usize, bottom: usize) {
        if top >= self.grid.rows() || bottom >= self.grid.rows() || top > bottom {
            return;
        }

        let cols = self.grid.row_len();
        // VTE saves lines scrolled off the top margin into scrollback whenever
        // the scrolling region starts at the first screen row. The bottom margin
        // may be above the last row so TUIs can keep prompts/status lines fixed
        // while the history area scrolls.
        let scrolls_off_screen_top = top == 0;

        // Compress the removed line directly from the grid slice before mutating,
        // avoiding a per-line Vec allocation from get_row.
        let allow_alt_scrollback = self.use_alt_buffer && self.sync_output_active;
        let scrollback_line = if scrolls_off_screen_top
            && (!self.use_alt_buffer || allow_alt_scrollback)
        {
            let mut line = ScrollbackLine::compress(&self.grid[top], self.grid.row_wrapped[top]);
            line.set_raw_row_id(self.grid.row_id(top));
            Some(line)
        } else {
            None
        };

        // A top-margin scroll transfers the RawRowId into scrollback. Partial
        // regions (and non-archiving buffers) really discard it instead.
        if scrollback_line.is_none() {
            self.forget_output_provenance_for_grid_rows(top, top);
        }

        let src_start = (top + 1) * cols;
        let src_end = (bottom + 1) * cols;
        let dst_start = top * cols;
        self.grid.cells.copy_within(src_start..src_end, dst_start);
        let blank_start = bottom * cols;
        // 保留当前背景色 / BCE
        let blank = self.create_blank_cell();
        self.grid.cells[blank_start..blank_start + cols].fill(blank);
        self.grid.row_wrapped.copy_within(top + 1..=bottom, top);
        self.grid.row_wrapped[bottom] = false;
        self.shift_row_id_region_up(top, bottom, 1);
        self.kitty_graphics.scroll_region_up(
            top,
            bottom,
            1,
            scrolls_off_screen_top && (!self.use_alt_buffer || allow_alt_scrollback),
        );

        self.dirty_region.mark_rows(top, bottom);
        self.mark_rows_dirty(top, bottom);

        if let Some(line) = scrollback_line {
            self.push_scrollback_transferred_with_options(line, allow_alt_scrollback);
            if !self.use_alt_buffer
                && bottom + 1 == self.grid.rows()
                && self.full_screen_scroll_revision != 0
            {
                self.full_screen_scroll_revision =
                    self.full_screen_scroll_revision.checked_add(1).unwrap_or(0);
            }
        }
    }

    pub(super) fn charset_from_designator(byte: u8) -> Charset {
        match byte {
            b'0' => Charset::DecSpecialGraphics,
            _ => Charset::Ascii,
        }
    }

    pub(super) fn translate_char(&self, ch: char) -> char {
        match self.active_charset {
            Charset::Ascii => ch,
            Charset::DecSpecialGraphics => match ch {
                '`' => '◆',
                'a' => '▒',
                'f' => '°',
                'g' => '±',
                'j' => '┘',
                'k' => '┐',
                'l' => '┌',
                'm' => '└',
                'n' => '┼',
                'o' => '⎺',
                'p' => '⎻',
                'q' => '─',
                'r' => '⎼',
                's' => '⎽',
                't' => '├',
                'u' => '┤',
                'v' => '┴',
                'w' => '┬',
                'x' => '│',
                'y' => '≤',
                'z' => '≥',
                '{' => 'π',
                '|' => '≠',
                '}' => '£',
                '~' => '·',
                _ => ch,
            },
        }
    }

    /// Blank the double-width halves that overwriting `start..end` orphans.
    ///
    /// Only the two edges of the span can have a partner outside it: a lead at
    /// `start - 1` whose continuation is about to go, and a continuation at
    /// `end` whose lead is about to go. A surviving half breaks what the
    /// renderer assumes — it skips continuation cells entirely and paints
    /// `wide()` cells two columns wide — so the orphan would either hide the
    /// character written into it or overpaint its neighbour, and the next write
    /// there would blank a good cell while repairing the pair.
    pub(super) fn split_wide_pairs_around(&mut self, row: usize, start: usize, end: usize) {
        let cols = self.grid.row_len();
        if row >= self.grid.rows() || start >= end || start >= cols {
            return;
        }
        let end = end.min(cols);
        let blank_cell = self.create_blank_cell();
        if start > 0 && self.grid.get(row, start - 1).flags.wide() {
            *self.grid.get_mut(row, start - 1) = blank_cell;
        }
        if end < cols && self.grid.get(row, end).flags.wide_continuation() {
            *self.grid.get_mut(row, end) = blank_cell;
        }
    }

    fn expanded_cell_mutation_span(&self, row: usize, start: usize, end: usize) -> (usize, usize) {
        let cols = self.grid.row_len();
        let mut start = start.min(cols);
        let mut end = end.min(cols);
        if row >= self.grid.rows() || start >= end {
            return (start, end);
        }
        if start > 0 && self.grid.get(row, start - 1).flags.wide() {
            start -= 1;
        }
        if end < cols && self.grid.get(row, end).flags.wide_continuation() {
            end += 1;
        }
        (start, end)
    }

    pub(super) fn clear_cell_unchecked(&mut self, row: usize, col: usize) {
        let cols = self.grid.row_len();
        let bg_color = self.current_bg;
        let blank_cell = TerminalCell {
            character: ' ',
            foreground: Color::Default,
            background: bg_color,
            flags: StyleFlags::default(),
            hyperlink_id: HyperlinkId::NONE,
        };
        // If clearing a continuation cell, also clear the wide character body
        if self.grid.get(row, col).flags.wide_continuation() && col > 0 {
            *self.grid.get_mut(row, col - 1) = blank_cell;
        }
        // If clearing a wide character body, also clear the continuation cell
        if self.grid.get(row, col).flags.wide() && col + 1 < cols {
            *self.grid.get_mut(row, col + 1) = blank_cell;
        }
        *self.grid.get_mut(row, col) = blank_cell;
    }

    /// P3 优化：批量处理输入数据，只在处理完成后触发一次网格版本更新
    /// 相比多次 process_input，这个方法避免了多次网格版本递增
    pub fn process_batch(&mut self, input: &[u8]) {
        self.grid_version = self.grid_version.wrapping_add(1);
        self.process_input(input);
    }

    #[inline]
    pub(super) fn mark_row_dirty(&mut self, row: usize) {
        if row < self.row_versions.len() {
            self.row_versions[row] = self.grid_version;
        }
    }

    #[inline]
    pub(super) fn mark_rows_dirty(&mut self, start: usize, end: usize) {
        for row in start..=end.min(self.row_versions.len().saturating_sub(1)) {
            self.row_versions[row] = self.grid_version;
        }
    }

    /// P4：获取上次渲染后修改过的行索引
    pub fn get_dirty_rows(&self, last_rendered_version: u64, out: &mut Vec<usize>) {
        out.clear();
        for (i, &v) in self.row_versions.iter().enumerate() {
            if v > last_rendered_version {
                out.push(i);
            }
        }
    }

    /// P4：获取网格版本号（用于缓存比较）
    pub fn get_grid_version(&self) -> u64 {
        self.grid_version
    }

    /// Resolve the compact hyperlink reference carried by a cell.
    pub fn hyperlink_uri(&self, id: HyperlinkId) -> Option<&str> {
        self.hyperlinks.resolve(id)
    }

    pub fn take_osc52_clipboard_set(&mut self) -> Option<String> {
        self.pending_osc52_clipboard_set.take()
    }

    pub fn take_osc52_clipboard_query(&mut self) -> bool {
        let q = self.pending_osc52_clipboard_query;
        self.pending_osc52_clipboard_query = false;
        q
    }

    /// Check if sync output timed out (>1s) and auto-clear if so
    pub fn check_sync_output_timeout(&mut self) {
        if self.sync_output_active {
            if let Some(start) = self.sync_output_start {
                if start.elapsed() > std::time::Duration::from_secs(1) {
                    if self.use_alt_buffer {
                        self.archive_visible_screen_to_scrollback_with_options(true, true);
                    } else {
                        self.last_synced_primary_screen_snapshot =
                            self.visible_screen_snapshot().unwrap_or_default();
                    }
                    self.sync_output_active = false;
                    self.sync_output_start = None;
                    self.modes.remove(&2026);
                    self.dirty_region.mark_all(self.grid.rows());
                    self.mark_rows_dirty(0, self.grid.rows().saturating_sub(1));
                }
            }
        }
    }

    /// 处理一个 UTF-8 多字节引导字节。`expected` 是该序列总长度(2/3/4)。
    /// - 缓冲区剩余字节不足:把引导字节暂存,等待下一批输入续接(跨 PTY 读边界)。
    /// - 续接字节齐全且合法:解码并写入字符;非法则输出替换字符 U+FFFD。
    /// - 续接字节非法(不是 10xxxxxx):序列残缺,输出 U+FFFD 且只消费引导字节本身,
    ///   让那个意外字节按自身规则重新处理。
    pub(super) fn consume_utf8_lead(&mut self, byte: u8, expected: u8, data: &[u8], i: &mut usize) {
        let need = expected as usize;
        if *i + need > data.len() {
            // 不完整:剩余字节不够,暂存等待下一批
            self.utf8_buf[0] = byte;
            self.utf8_len = 1;
            self.utf8_expected = expected;
            *i += 1;
            return;
        }

        let all_continuation = (1..need).all(|k| (data[*i + k] & 0xC0) == 0x80);
        if all_continuation {
            match std::str::from_utf8(&data[*i..*i + need]) {
                Ok(s) => {
                    if let Some(ch) = s.chars().next() {
                        self.put_char(ch);
                    }
                }
                Err(_) => self.put_char('\u{FFFD}'),
            }
            *i += need;
        } else {
            // 引导字节后紧跟非续接字节:残缺序列
            self.put_char('\u{FFFD}');
            *i += 1;
        }
    }
}

impl super::TerminalState {
    pub(super) fn hard_reset(&mut self) {
        self.record_reset_interrupted_agent_command();
        // Completion events are application-owned once produced. RIS clears
        // terminal history, but must not erase an already-finished event or
        // the Agent interruption emitted immediately above before the app can
        // drain it.
        let pending_completed_command_outputs =
            std::mem::take(&mut self.pending_completed_command_outputs);
        let consumed_command_ids = std::mem::take(&mut self.consumed_command_ids);
        let cols = self.grid.row_len();
        let rows = self.grid.rows();
        let max_scrollback = self.max_scrollback;
        let cell_size = self.kitty_graphics.cell_size_pixels();
        let next_raw_row_id = self.next_raw_row_id;
        let row_identity_revision = self.row_identity_revision;
        let next_command_sequence = self.next_command_sequence;
        *self = Self::new(cols, rows);
        self.pending_completed_command_outputs = pending_completed_command_outputs;
        self.consumed_command_ids = consumed_command_ids;
        // A reset replaces every physical row but must never restart the
        // allocator and let an old external origin retarget into the new grid.
        self.next_raw_row_id = next_raw_row_id;
        self.row_identity_revision = row_identity_revision;
        self.next_command_sequence = next_command_sequence;
        self.grid.row_ids.fill(RawRowId::UNTRACKED);
        self.alt_grid.row_ids.fill(RawRowId::UNTRACKED);
        self.fill_untracked_grid_row_ids();
        self.mark_row_identity_changed();
        self.set_max_scrollback(max_scrollback);
        self.kitty_graphics
            .set_cell_size_pixels(cell_size.0, cell_size.1);
    }

    pub fn max_scrollback(&self) -> usize {
        self.max_scrollback
    }

    /// 当前 scrollback 已有的行数(滚动上界)。
    pub fn scrollback_len(&self) -> usize {
        self.scrollback.len()
    }

    /// 当前是否处于备用屏幕缓冲(此时滚轮不滚动 scrollback)。
    pub fn is_alt_buffer(&self) -> bool {
        self.use_alt_buffer
    }

    pub fn set_max_scrollback(&mut self, max_scrollback: usize) {
        self.max_scrollback = max_scrollback.max(1);
        self.kitty_graphics
            .set_max_scrollback_rows(self.max_scrollback);

        let old_len = self.scrollback.len();
        while self.scrollback.len() > self.max_scrollback {
            if let Some(evicted) = self.scrollback.pop_front() {
                self.forget_output_provenance_for_raw_row(evicted.raw_row_id());
            }
        }
        if self.scrollback.len() != old_len {
            self.invalidate_scrollback_view_cache();
        }

        self.scroll_offset = self.scroll_offset.min(self.scrollback.len());
    }

    fn register_finished_output_provenance(&mut self, provenance: FinishedOutputProvenance) {
        let zone_id = provenance.range.zone_id;
        self.unregister_finished_output_zone(zone_id);
        for row in &provenance.rows {
            self.finished_output_owners
                .entry(row.row_id)
                .or_default()
                .push(FinishedOutputOwner {
                    zone_id,
                    start_col: row.start_col,
                    end_col: row.end_col,
                });
        }
        self.finished_output_provenance.insert(zone_id, provenance);
        self.mark_finished_output_provenance_changed();
    }

    fn unregister_finished_output_zone(&mut self, zone_id: u64) {
        let Some(provenance) = self.finished_output_provenance.remove(&zone_id) else {
            return;
        };
        for row in provenance.rows {
            let remove_bucket =
                if let Some(owners) = self.finished_output_owners.get_mut(&row.row_id) {
                    owners.retain(|owner| owner.zone_id != zone_id);
                    owners.is_empty()
                } else {
                    false
                };
            if remove_bucket {
                self.finished_output_owners.remove(&row.row_id);
            }
        }
        self.mark_finished_output_provenance_changed();
    }

    fn mark_finished_output_provenance_changed(&mut self) {
        if self.finished_output_revision != 0 {
            self.finished_output_revision =
                self.finished_output_revision.checked_add(1).unwrap_or(0);
        }
        // Even if the diagnostic counter is exhausted, clearing both exact
        // caches prevents a mutation from reusing the previous ownership map.
        self.projection_plan_cache = None;
        self.transformed_viewport_cache = None;
    }

    pub(super) fn invalidate_finished_output_spans(&mut self, spans: &[(RawRowId, usize, usize)]) {
        if self.use_alt_buffer || spans.is_empty() {
            return;
        }
        let mut zones = HashSet::new();
        for &(row_id, start_col, end_col) in spans {
            if !row_id.is_tracked() || start_col >= end_col {
                continue;
            }
            if let Some(owners) = self.finished_output_owners.get(&row_id) {
                zones.extend(
                    owners
                        .iter()
                        .filter(|owner| start_col < owner.end_col && owner.start_col < end_col)
                        .map(|owner| owner.zone_id),
                );
            }
        }
        for zone_id in zones {
            self.unregister_finished_output_zone(zone_id);
        }
    }

    fn invalidate_finished_output_span(&mut self, row: usize, start_col: usize, end_col: usize) {
        if self.use_alt_buffer || row >= self.grid.rows() {
            return;
        }
        self.invalidate_finished_output_spans(&[(self.grid.row_id(row), start_col, end_col)]);
    }

    pub(super) fn invalidate_grid_mutation_spans(&mut self, spans: &[(usize, usize, usize)]) {
        if self.use_alt_buffer {
            return;
        }
        let raw_spans: Vec<_> = spans
            .iter()
            .filter_map(|&(row, start, end)| {
                if row >= self.grid.rows() {
                    return None;
                }
                let (start, end) = self.expanded_cell_mutation_span(row, start, end);
                (start < end).then_some((self.grid.row_id(row), start, end))
            })
            .collect();
        self.invalidate_finished_output_spans(&raw_spans);
    }

    fn forget_output_provenance_for_raw_row(&mut self, row_id: RawRowId) {
        if !row_id.is_tracked() {
            return;
        }
        let zones: Vec<_> = self
            .finished_output_owners
            .get(&row_id)
            .into_iter()
            .flatten()
            .map(|owner| owner.zone_id)
            .collect();
        #[cfg(test)]
        FINISHED_OUTPUT_EVICTION_ROW_CHECKS.with(|checks| {
            checks.set(checks.get().saturating_add(zones.len()));
        });
        for zone_id in zones {
            self.unregister_finished_output_zone(zone_id);
        }
        if self.active_output_provenance.is_some_and(|active| {
            active.start_row_id == row_id || active.extent_row_id == Some(row_id)
        }) {
            self.active_output_provenance = None;
        }
    }

    pub(super) fn forget_output_provenance_for_grid_rows(
        &mut self,
        start_row: usize,
        end_row: usize,
    ) {
        if start_row > end_row || start_row >= self.grid.rows() {
            return;
        }
        let row_ids: Vec<_> = (start_row..=end_row.min(self.grid.rows() - 1))
            .map(|row| self.grid.row_id(row))
            .collect();
        for row_id in row_ids {
            self.forget_output_provenance_for_raw_row(row_id);
        }
    }

    pub(super) fn clear_scrollback_with_projection_provenance(&mut self) {
        let evicted: HashSet<_> = self
            .scrollback
            .iter()
            .map(ScrollbackLine::raw_row_id)
            .filter(|row| row.is_tracked())
            .collect();
        let zones: HashSet<_> = evicted
            .iter()
            .filter_map(|row| self.finished_output_owners.get(row))
            .flatten()
            .map(|owner| owner.zone_id)
            .collect();
        for zone_id in zones {
            self.unregister_finished_output_zone(zone_id);
        }
        self.scrollback.clear();
        if self.active_output_provenance.is_some_and(|active| {
            evicted.contains(&active.start_row_id)
                || active
                    .extent_row_id
                    .is_some_and(|row| evicted.contains(&row))
        }) {
            self.active_output_provenance = None;
        }
    }

    pub fn is_cursor_visible(&self) -> bool {
        // Cursor is visible when mode 25 is SET (via \x1b[?25h)
        // Hidden when mode 25 is RESET (via \x1b[?25l)
        // While viewing scrollback we intentionally hide the live cursor,
        // because the viewport no longer tracks the active prompt line.
        self.modes.contains(&25) && self.scroll_offset == 0
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    /// Current stable buffer boundary. When DEC's delayed-wrap flag is set,
    /// the cursor is visually parked on the last cell even though the logical
    /// boundary is after it; represent that as `column == cols` so output
    /// extraction does not drop the final character.
    fn current_buffer_anchor(&self) -> BufferAnchor {
        let column = if self.pending_wrap {
            self.grid.row_len()
        } else {
            self.cursor_col
        };
        BufferAnchor {
            line_id: self
                .total_lines_scrolled
                .saturating_add(self.cursor_row as u64),
            column,
        }
    }

    fn retained_raw_row(&self, absolute_row: usize) -> Option<(RawRowId, usize)> {
        if let Some(line) = self.scrollback.get(absolute_row) {
            return Some((line.raw_row_id(), line.columns()));
        }
        let grid_row = absolute_row.checked_sub(self.scrollback.len())?;
        (grid_row < self.grid.rows()).then(|| (self.grid.row_id(grid_row), self.grid.row_len()))
    }

    /// Stable physical-row identity at an absolute primary-buffer row.
    pub fn raw_row_id_at_absolute(&self, absolute_row: usize) -> Option<RawRowId> {
        self.retained_raw_row(absolute_row)
            .map(|(row_id, _)| row_id)
    }

    /// Convert an exact raw-cell origin back to the terminal's stable
    /// line-id coordinate space. Trimmed and untracked rows fail closed.
    pub fn raw_cell_anchor_to_buffer_anchor(&self, anchor: RawCellAnchor) -> Option<BufferAnchor> {
        let absolute = self.retained_absolute_for_raw(anchor.row_id)?;
        let (_, width) = self.retained_raw_row(absolute)?;
        (anchor.column <= width)
            .then(|| self.absolute_to_buffer_anchor((absolute, anchor.column)))?
    }

    /// Resolve a retained physical row by stable identity. The positional
    /// anchor is the O(1) common path (including full-screen scroll into
    /// history); the bounded live-grid fallback covers identity-preserving
    /// partial-region IL/DL/RI moves without scanning deep scrollback.
    fn retained_raw_row_absolute(
        &self,
        row_id: RawRowId,
        preferred: Option<BufferAnchor>,
    ) -> Option<(usize, usize)> {
        if !row_id.is_tracked() {
            return None;
        }
        if let Some(anchor) = preferred {
            if let Some((absolute, _)) = self.buffer_anchor_to_absolute(anchor) {
                if let Some((candidate, width)) = self.retained_raw_row(absolute) {
                    if candidate == row_id {
                        return Some((absolute, width));
                    }
                }
            }
        }
        self.grid
            .row_ids
            .iter()
            .position(|candidate| *candidate == row_id)
            .map(|row| (self.scrollback.len() + row, self.grid.row_len()))
    }

    fn retained_cell_is_wide_continuation(&self, absolute_row: usize, col: usize) -> Option<bool> {
        if let Some(line) = self.scrollback.get(absolute_row) {
            return line
                .decompress()
                .get(col)
                .map(|cell| cell.flags.wide_continuation());
        }
        let grid_row = absolute_row.checked_sub(self.scrollback.len())?;
        (grid_row < self.grid.rows() && col < self.grid.row_len())
            .then(|| self.grid.get(grid_row, col).flags.wide_continuation())
    }

    fn note_output_write(
        &mut self,
        grid_row: usize,
        row_id: RawRowId,
        start_col: usize,
        end_col: usize,
    ) {
        if self.use_alt_buffer {
            return;
        }
        let Some(snapshot) = self.active_output_provenance else {
            return;
        };
        if snapshot.invalid {
            return;
        }
        let write_start = BufferAnchor {
            line_id: self.total_lines_scrolled.saturating_add(grid_row as u64),
            column: start_col.min(self.grid.row_len()),
        };
        let extent = BufferAnchor {
            line_id: write_start.line_id,
            column: end_col.min(self.grid.row_len()),
        };
        let Some((write_absolute, _)) = self.retained_raw_row_absolute(row_id, Some(write_start))
        else {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        };
        let Some((start_absolute, _)) =
            self.retained_raw_row_absolute(snapshot.start_row_id, Some(snapshot.start))
        else {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        };
        if write_absolute < start_absolute {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        }

        let previous_extent_absolute = snapshot.extent_row_id.and_then(|extent_row_id| {
            self.retained_raw_row_absolute(extent_row_id, snapshot.extent)
                .map(|(absolute, _)| absolute)
        });
        let Some(active) = self.active_output_provenance.as_mut() else {
            return;
        };
        if write_absolute == start_absolute {
            if active.start_row_id != row_id {
                active.invalid = true;
                return;
            }
            if start_col < active.start.column {
                active.start.column = start_col;
            }
        }
        active.last_write_end = Some(extent);
        active.semantic_end = Some(extent);
        active.cursor_moved_since_output = false;
        match (active.extent, previous_extent_absolute) {
            (None, _) => {
                active.extent = Some(extent);
                active.extent_row_id = Some(row_id);
            }
            (Some(_), Some(previous_absolute)) if write_absolute > previous_absolute => {
                active.extent = Some(extent);
                active.extent_row_id = Some(row_id);
            }
            (Some(previous), Some(previous_absolute)) if write_absolute == previous_absolute => {
                if active.extent_row_id != Some(row_id) {
                    active.invalid = true;
                    return;
                }
                if extent.column > previous.column {
                    active.extent = Some(extent);
                }
            }
            (Some(_), Some(_)) => {}
            (Some(_), None) => active.invalid = true,
        }
    }

    /// Record a real hard line advance (LF/IND), never an explicit cursor move.
    /// `departed_*` describes the physical row before the move and
    /// `next_boundary` is the logical boundary immediately after it.
    pub(super) fn note_output_hard_line_advance(
        &mut self,
        departed_row_id: RawRowId,
        departed_boundary: BufferAnchor,
        next_boundary: BufferAnchor,
    ) {
        if self.use_alt_buffer {
            return;
        }
        let Some(snapshot) = self.active_output_provenance else {
            return;
        };
        if snapshot.invalid {
            return;
        }
        let Some((departed_absolute, departed_width)) =
            self.retained_raw_row_absolute(departed_row_id, Some(departed_boundary))
        else {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        };
        let Some((start_absolute, _)) =
            self.retained_raw_row_absolute(snapshot.start_row_id, Some(snapshot.start))
        else {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        };
        if departed_absolute < start_absolute {
            if let Some(active) = self.active_output_provenance.as_mut() {
                active.invalid = true;
            }
            return;
        }
        let extent = BufferAnchor {
            line_id: departed_boundary.line_id,
            column: departed_width,
        };
        let previous_extent_absolute = snapshot.extent_row_id.and_then(|extent_row_id| {
            self.retained_raw_row_absolute(extent_row_id, snapshot.extent)
                .map(|(absolute, _)| absolute)
        });
        let Some(active) = self.active_output_provenance.as_mut() else {
            return;
        };
        match (active.extent, previous_extent_absolute) {
            (None, _) => {
                active.extent = Some(extent);
                active.extent_row_id = Some(departed_row_id);
            }
            (Some(_), Some(previous_absolute)) if departed_absolute > previous_absolute => {
                active.extent = Some(extent);
                active.extent_row_id = Some(departed_row_id);
            }
            (Some(previous), Some(previous_absolute))
                if departed_absolute == previous_absolute && previous.column < departed_width =>
            {
                if active.extent_row_id != Some(departed_row_id) {
                    active.invalid = true;
                    return;
                }
                active.extent = Some(extent);
            }
            (Some(_), Some(_)) => {}
            (Some(_), None) => {
                active.invalid = true;
                return;
            }
        }
        active.semantic_end = Some(next_boundary);
        active.cursor_moved_since_output = false;
    }

    /// Mark a cursor-only control within an active output lifecycle. Horizontal
    /// moves may be resolved by a later write/LF; non-linear vertical moves are
    /// conservatively ineligible because a single contiguous raw range cannot
    /// prove that skipped or revisited rows belong to command output.
    pub(super) fn note_output_cursor_reposition(&mut self, non_linear: bool) {
        if self.use_alt_buffer {
            return;
        }
        if let Some(active) = self.active_output_provenance.as_mut() {
            active.cursor_moved_since_output = true;
            active.invalid |= non_linear;
        }
    }

    /// An automatic soft wrap before the first output cell must not make the
    /// already-full command/header row part of the output. Rebase C to the new
    /// physical row; later wraps retain the original output start.
    fn rebase_active_output_start_after_initial_wrap(&mut self, departed_row_id: RawRowId) {
        if self.use_alt_buffer {
            return;
        }
        let should_rebase = self.active_output_provenance.is_some_and(|active| {
            !active.invalid
                && active.start_row_id == departed_row_id
                && active.extent.is_none()
                && active.last_write_end.is_none()
        });
        if !should_rebase {
            return;
        }
        let start = BufferAnchor {
            line_id: self
                .total_lines_scrolled
                .saturating_add(self.cursor_row as u64),
            column: 0,
        };
        let row_id = self.grid.row_id(self.cursor_row);
        if let Some(active) = self.active_output_provenance.as_mut() {
            active.start = start;
            active.start_row_id = row_id;
        }
    }

    pub(super) fn bind_finished_output_provenance(
        &self,
        zone_id: u64,
        start_row: usize,
        start_col: usize,
        end_row: usize,
        end_col: usize,
    ) -> Option<FinishedOutputProvenance> {
        if start_row > end_row {
            return None;
        }
        let (start_id, start_width) = self.retained_raw_row(start_row)?;
        let (end_id, end_width) = self.retained_raw_row(end_row)?;
        if !start_id.is_tracked() || !end_id.is_tracked() {
            return None;
        }
        let mut start_col = start_col.min(start_width);
        if start_col > 0
            && start_col < start_width
            && self.retained_cell_is_wide_continuation(start_row, start_col)?
        {
            start_col -= 1;
        }
        let mut end_col = end_col.min(end_width);
        if end_col < end_width && self.retained_cell_is_wide_continuation(end_row, end_col)? {
            end_col = end_col.saturating_add(1).min(end_width);
        }
        if start_row == end_row && start_col >= end_col {
            return None;
        }
        let rows: Vec<_> = (start_row..=end_row)
            .map(|row| {
                let (row_id, width) = self.retained_raw_row(row)?;
                Some(FinishedOutputRow {
                    row_id,
                    start_col: if row == start_row { start_col } else { 0 },
                    end_col: if row == end_row { end_col } else { width },
                })
            })
            .collect::<Option<_>>()?;
        if rows
            .iter()
            .any(|row| !row.row_id.is_tracked() || row.start_col >= row.end_col)
        {
            return None;
        }
        Some(FinishedOutputProvenance {
            range: FinishedOutputRange {
                zone_id,
                start: RawCellBoundary {
                    row: start_id,
                    col: start_col,
                },
                end: RawCellBoundary {
                    row: end_id,
                    col: end_col,
                },
            },
            start_line_id: self.absolute_to_buffer_anchor((start_row, 0))?.line_id,
            rows,
        })
    }

    fn finish_active_output_provenance(
        &mut self,
        zone_id: u64,
    ) -> Option<(FinishedOutputProvenance, BufferAnchor, BufferAnchor)> {
        let active = self.active_output_provenance.take()?;
        if active.zone_id != zone_id || active.invalid || active.cursor_moved_since_output {
            return None;
        }
        let (start_row, _) =
            self.retained_raw_row_absolute(active.start_row_id, Some(active.start))?;
        let start_col = active.start.column;
        let (Some(extent), Some(extent_row_id)) = (active.extent, active.extent_row_id) else {
            return None;
        };
        let (extent_row, _) = self.retained_raw_row_absolute(extent_row_id, Some(extent))?;
        if start_row > extent_row {
            return None;
        };
        let provenance = self.bind_finished_output_provenance(
            zone_id,
            start_row,
            start_col,
            extent_row,
            extent.column,
        )?;
        let output_start =
            self.absolute_to_buffer_anchor((start_row, provenance.range.start.col))?;
        let cell_end = self.absolute_to_buffer_anchor((extent_row, provenance.range.end.col))?;
        let output_end = active
            .semantic_end
            .filter(|end| *end >= output_start && self.buffer_anchor_to_absolute(*end).is_some())
            .map_or(cell_end, |end| end.max(cell_end));
        Some((provenance, output_start, output_end))
    }

    /// Highest still-visible primary-grid row containing output at or after a
    /// semantic output anchor. A full-screen program can print below the
    /// cursor and then CUP back upward; mouse ownership must continue to cover
    /// those displayed rows even though the cursor no longer reaches them.
    pub(crate) fn primary_content_extent_from(&self, start: BufferAnchor) -> Option<u64> {
        if self.use_alt_buffer {
            return None;
        }
        let first_row = start
            .line_id
            .saturating_sub(self.total_lines_scrolled)
            .min(self.grid.rows() as u64) as usize;
        (first_row..self.grid.rows())
            .rfind(|row| !self.line_is_blank(*row))
            .map(|row| self.total_lines_scrolled.saturating_add(row as u64))
    }

    /// Translate a recorded `line_id` to its current `scrollback` index, or
    /// `None` if the line has been evicted (or now lives in the live grid,
    /// which means it's already on screen).
    fn line_id_to_scrollback_index(&self, line_id: u64) -> Option<usize> {
        if line_id >= self.total_lines_scrolled {
            // Line is either in the live grid (>= total_lines_scrolled) or
            // hasn't happened yet (impossible via this API). The grid is
            // already on screen, so caller can scroll to bottom.
            return None;
        }
        let first_scrollback_line_id = self
            .total_lines_scrolled
            .saturating_sub(self.scrollback.len() as u64);
        if line_id < first_scrollback_line_id {
            // Evicted from scrollback.
            return None;
        }
        Some((line_id - first_scrollback_line_id) as usize)
    }

    /// Drop marks that point to lines no longer in scrollback. Called
    /// lazily before navigation rather than on every scrollback push.
    fn prune_evicted_marks(&mut self) {
        let first_scrollback_line_id = self
            .total_lines_scrolled
            .saturating_sub(self.scrollback.len() as u64);
        while self
            .command_marks
            .front()
            .map(|m| m.line_id < first_scrollback_line_id)
            .unwrap_or(false)
        {
            self.command_marks.pop_front();
        }
    }

    /// Decode one OSC 133 metadata field without ever accepting a prefix as
    /// the complete value. Exact command actions depend on this distinction:
    /// an over-limit or invalid UTF-8 field must be rejected, not shortened
    /// and subsequently labelled exact.
    fn percent_decode_osc_133(value: &str, max_bytes: usize) -> Result<String, Osc133DecodeError> {
        let bytes = value.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len().min(max_bytes));
        let mut i = 0;
        while i < bytes.len() {
            let byte = if bytes[i] == b'%' {
                if i + 2 >= bytes.len() {
                    return Err(Osc133DecodeError::MalformedPercentEncoding);
                }
                let high = (bytes[i + 1] as char)
                    .to_digit(16)
                    .ok_or(Osc133DecodeError::MalformedPercentEncoding)?
                    as u8;
                let low = (bytes[i + 2] as char)
                    .to_digit(16)
                    .ok_or(Osc133DecodeError::MalformedPercentEncoding)?
                    as u8;
                i += 3;
                (high << 4) | low
            } else {
                let byte = bytes[i];
                i += 1;
                byte
            };
            if decoded.len() == max_bytes {
                return Err(Osc133DecodeError::TooLong);
            }
            decoded.push(byte);
        }

        String::from_utf8(decoded).map_err(|_| Osc133DecodeError::InvalidUtf8)
    }

    fn valid_osc_133_id(value: &str) -> Option<String> {
        let id = Self::percent_decode_osc_133(value, MAX_OSC_133_ID_BYTES).ok()?;
        // An execution id is rendered — the Commands sidebar and the block
        // export both print it — and it is a routing key the user is expected
        // to be able to tell apart from a neighbour. The shared parser refuses
        // the visual-spoofing class here for exactly that reason; ember checked
        // only for controls, so a right-to-left override or an invisible
        // joiner inside an id passed straight through to the UI.
        if id.is_empty()
            || id.chars().any(char::is_control)
            || id
                .chars()
                .any(jterm_core::review_input::is_terminal_visual_spoofing_character)
        {
            return None;
        }
        Some(id)
    }

    fn local_command_id(sequence: u64) -> String {
        format!("local:{sequence}")
    }

    fn next_command_identity(&mut self) -> Option<(u64, String)> {
        let sequence = self.next_command_sequence;
        if sequence == 0 {
            return None;
        }
        self.next_command_sequence = sequence.checked_add(1).unwrap_or(0);
        Some((sequence, Self::local_command_id(sequence)))
    }

    fn record_index_for_id(&self, id: &str) -> Option<usize> {
        self.command_records
            .iter()
            .rposition(|record| record.id == id)
    }

    pub(super) fn command_id_was_consumed(&self, id: &str) -> bool {
        self.consumed_command_ids
            .iter()
            .any(|consumed| consumed == id)
    }

    pub(super) fn remember_consumed_command_id(&mut self, id: Option<&str>) {
        let Some(id) = id.filter(|id| !id.is_empty()) else {
            return;
        };
        if self.command_id_was_consumed(id) {
            return;
        }
        if self.consumed_command_ids.len() >= MAX_CONSUMED_COMMAND_IDS {
            self.consumed_command_ids.pop_front();
        }
        self.consumed_command_ids.push_back(id.to_string());
    }

    fn active_record_index(&self) -> Option<usize> {
        self.command_records
            .iter()
            .rposition(|record| !record.complete)
    }

    fn adopt_record_id(&mut self, index: usize, requested_id: Option<&str>) {
        let Some(requested_id) = requested_id.and_then(Self::valid_osc_133_id) else {
            return;
        };
        if self.command_id_was_consumed(&requested_id) {
            return;
        }
        if self
            .command_records
            .iter()
            .enumerate()
            .any(|(other, record)| other != index && record.id == requested_id)
        {
            return;
        }
        if let Some(record) = self.command_records.get_mut(index) {
            record.id = requested_id;
        }
    }

    /// Decode one OSC 133 `cwd`/`cwd_url` field into a directory this terminal
    /// is willing to record and later spawn in.
    ///
    /// The shared journal validator bundles every rule that matters here — the
    /// 4 KiB budget, non-emptiness, control rejection, and the visual-spoofing
    /// class — and ember must apply the same one: a recorded cwd is shown in
    /// the pane header and the Commands sidebar, is what `is_valid_jsh_cwd`
    /// gates on the journal side, and becomes the working directory of a new
    /// session split from this block. A path carrying a bidi override or an
    /// invisible joiner would display as a directory the user did not choose.
    fn decode_osc_133_cwd(value: &str) -> Option<String> {
        Self::percent_decode_osc_133(value, MAX_OSC_133_CWD_BYTES)
            .ok()
            .filter(|cwd| jterm_core::execution_journal::is_valid_jsh_cwd(cwd))
    }

    fn apply_record_metadata(
        &mut self,
        index: usize,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        command_truncated: bool,
    ) {
        self.adopt_record_id(index, id);
        let decoded_command =
            command.map(|value| Self::percent_decode_osc_133(value, MAX_OSC_133_COMMAND_BYTES));
        let decoded_cwd = cwd.and_then(Self::decode_osc_133_cwd);
        if let Some(record) = self.command_records.get_mut(index) {
            // The disclosure is sticky and is honoured on whichever mark
            // carries it, not only on `C`. A shell that discloses the
            // shortening on `A` or `B` has said the recorded text cannot
            // authorize a re-run; dropping the flag on those marks left the
            // record looking like an ordinary complete command.
            if command_truncated {
                record.command = None;
                record.command_exact = false;
                record.command_truncated = true;
            }
            match decoded_command {
                Some(Ok(command)) => {
                    record.command = Some(command);
                    record.command_exact = true;
                }
                Some(Err(Osc133DecodeError::TooLong)) => {
                    // Preserve the execution row, but never expose a decoded
                    // prefix through exact copy/fill/rerun actions.
                    record.command = None;
                    record.command_exact = false;
                    record.command_truncated = true;
                }
                Some(Err(
                    Osc133DecodeError::MalformedPercentEncoding | Osc133DecodeError::InvalidUtf8,
                ))
                | None => {}
            }
            if let Some(cwd) = decoded_cwd {
                record.cwd = Some(cwd);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_command_record(
        &mut self,
        anchor: BufferAnchor,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        command_truncated: bool,
        start_mark_seen: bool,
    ) -> Option<usize> {
        let (sequence, local_id) = self.next_command_identity()?;
        if self.command_records.len() >= MAX_COMMAND_MARKS {
            if let Some(evicted) = self.command_records.pop_front() {
                self.unregister_finished_output_zone(evicted.sequence);
                if self
                    .active_output_provenance
                    .is_some_and(|active| active.zone_id == evicted.sequence)
                {
                    self.active_output_provenance = None;
                }
                self.captured_command_output_bytes =
                    self.captured_command_output_bytes.saturating_sub(
                        evicted
                            .captured_output
                            .as_ref()
                            .map(|output| output.text.len())
                            .unwrap_or(0),
                    );
            }
        }
        self.command_records.push_back(CommandRecord {
            id: local_id,
            sequence,
            command: None,
            command_exact: false,
            command_truncated: false,
            cwd: self.current_working_dir.clone(),
            cwd_after: None,
            session_id: None,
            seq: None,
            started_at_ms: None,
            prompt_start: anchor,
            command_start: None,
            output_start: None,
            output_end: None,
            end: None,
            exit_code: None,
            duration_ms: None,
            state: CommandState::Prompt,
            complete: false,
            start_mark_seen,
            completion_provenance: crate::block_mode::CompletionProvenance::Unknown,
            started_at: None,
            finished_at: None,
            agent_generation: None,
            captured_output: None,
            started_instant: None,
        });
        let index = self.command_records.len() - 1;
        self.apply_record_metadata(index, id, command, cwd, command_truncated);
        Some(index)
    }

    fn ensure_active_record(&mut self) -> Option<usize> {
        if let Some(index) = self.active_record_index() {
            return Some(index);
        }
        let anchor = self.current_buffer_anchor();
        let index = self.push_command_record(anchor, None, None, None, false, false)?;
        if self.command_marks.len() >= MAX_COMMAND_MARKS {
            self.command_marks.pop_front();
        }
        self.command_marks.push_back(CommandMark {
            line_id: anchor.line_id,
            exit_code: None,
        });
        Some(index)
    }

    fn record_prompt_start_with_metadata(
        &mut self,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        command_truncated: bool,
    ) {
        // Bypass the alt buffer entirely (less / vim emit no marks; if they
        // did, they'd contaminate the primary-screen command history).
        if self.use_alt_buffer {
            return;
        }
        // A fresh shell prompt is the only unambiguous boundary that clears
        // local input and any approval that never reached command start.
        self.agent_prompt_input_tainted = false;
        self.record_abandoned_armed_agent_command(true);
        let anchor = self.current_buffer_anchor();

        // Only coalesce truly duplicated A markers. A new A on the same row
        // after a completed zero-output command is still a distinct command.
        if let Some(index) = self.command_records.len().checked_sub(1) {
            let duplicate = self
                .command_records
                .get(index)
                .map(|record| {
                    !record.complete
                        && record.state == CommandState::Prompt
                        && record.prompt_start == anchor
                })
                .unwrap_or(false);
            if duplicate {
                self.apply_record_metadata(index, id, command, cwd, command_truncated);
                return;
            }
        }

        // If a shell omitted D, preserve a closed semantic range rather than
        // leaving an earlier record permanently "running".
        if let Some(index) = self.active_record_index() {
            self.finish_command_record(
                index,
                anchor,
                None,
                None,
                crate::block_mode::CompletionProvenance::BoundaryInferred,
            );
        }

        if self
            .push_command_record(anchor, id, command, cwd, command_truncated, false)
            .is_some()
        {
            if self.command_marks.len() >= MAX_COMMAND_MARKS {
                self.command_marks.pop_front();
            }
            self.command_marks.push_back(CommandMark {
                line_id: anchor.line_id,
                exit_code: None,
            });
        }
    }

    fn record_command_start(
        &mut self,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        command_truncated: bool,
    ) {
        if self.use_alt_buffer {
            return;
        }
        let Some(index) = self.ensure_active_record() else {
            return;
        };
        self.apply_record_metadata(index, id, command, cwd, command_truncated);
        let anchor = self.current_buffer_anchor();
        if let Some(record) = self.command_records.get_mut(index) {
            if matches!(record.state, CommandState::Prompt | CommandState::Editing) {
                record.command_start.get_or_insert(anchor);
                record.state = CommandState::Editing;
            }
        }
    }

    fn record_output_start(
        &mut self,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        command_truncated: bool,
        start_identity: Osc133StartIdentity,
    ) {
        if self.use_alt_buffer {
            return;
        }
        let Some(index) = self.ensure_active_record() else {
            return;
        };
        let anchor = self.current_buffer_anchor();
        let reconstructed = self
            .command_records
            .get(index)
            .filter(|record| record.command.is_none() && command.is_none() && !command_truncated)
            .and_then(|record| record.command_start)
            .and_then(|start| self.extract_text_range(start, anchor, MAX_OSC_133_COMMAND_BYTES))
            .map(|extracted| extracted.text.trim_end_matches(['\r', '\n']).to_string())
            .filter(|command| !command.is_empty());

        self.apply_record_metadata(index, id, command, cwd, command_truncated);
        let mut initialize_provenance = false;
        if let Some(record) = self.command_records.get_mut(index) {
            if record.command.is_none() && !record.command_truncated {
                record.command = reconstructed;
            }
            if record.output_start.is_none() {
                record.output_start = Some(anchor);
                initialize_provenance = true;
            }
            record.state = CommandState::Running;
            // Start identity is captured once, on the first `C` this record
            // sees. jsh emits exactly one `C` per execution; a repeated mark
            // carrying a different session, sequence or start timestamp would
            // otherwise rebind an already-observed Start generation to output
            // that was captured for the first one.
            if !record.start_mark_seen {
                record.session_id = start_identity.session_id;
                record.seq = start_identity.seq;
                record.started_at_ms = start_identity.started_at_ms;
            }
            record.start_mark_seen = true;
            record
                .started_at
                .get_or_insert_with(std::time::SystemTime::now);
            record
                .started_instant
                .get_or_insert_with(std::time::Instant::now);
        }
        if initialize_provenance {
            self.active_output_provenance = self.command_records.get(index).and_then(|record| {
                let (absolute_row, _) = self.buffer_anchor_to_absolute(anchor)?;
                let (start_row_id, _) = self.retained_raw_row(absolute_row)?;
                start_row_id.is_tracked().then_some(ActiveOutputProvenance {
                    zone_id: record.sequence,
                    start: anchor,
                    start_row_id,
                    extent: None,
                    extent_row_id: None,
                    last_write_end: None,
                    semantic_end: None,
                    cursor_moved_since_output: false,
                    invalid: false,
                })
            });
        }

        // Bind the local one-shot generation only after the shell begins the
        // exact command that was reviewed. OSC ids/commands are PTY input and
        // therefore never supply this authorization identity themselves.
        let armed_generation = self
            .armed_agent_execution
            .as_ref()
            .filter(|armed| {
                !self.agent_prompt_input_tainted
                    && self.command_records.get(index).is_some_and(|record| {
                        record.sequence == armed.command_sequence
                            && record.command.as_deref() == Some(armed.command.as_str())
                    })
            })
            .map(|armed| armed.generation);
        if let Some(record) = self.command_records.get_mut(index) {
            record.agent_generation = armed_generation;
        }
        if armed_generation.is_some() {
            self.armed_agent_execution = None;
        } else {
            // C arrived for a different/tainted command. Release the approval
            // generation now, but do not tombstone this record's id: it belongs
            // to the unrelated command that is legitimately still running.
            self.record_abandoned_armed_agent_command(false);
        }
    }

    pub(super) fn store_captured_command_output(&mut self, index: usize, output: ExtractedText) {
        let previous_bytes = self
            .command_records
            .get_mut(index)
            .and_then(|record| record.captured_output.take())
            .map(|previous| previous.text.len())
            .unwrap_or(0);
        self.captured_command_output_bytes = self
            .captured_command_output_bytes
            .saturating_sub(previous_bytes);

        let output_bytes = output.text.len();
        while self
            .captured_command_output_bytes
            .saturating_add(output_bytes)
            > MAX_CAPTURED_COMMAND_OUTPUT_BYTES
        {
            let Some(evict_index) =
                self.command_records
                    .iter()
                    .enumerate()
                    .find_map(|(candidate, record)| {
                        (candidate != index && record.captured_output.is_some())
                            .then_some(candidate)
                    })
            else {
                break;
            };
            if let Some(evicted) = self.command_records[evict_index].captured_output.take() {
                self.captured_command_output_bytes = self
                    .captured_command_output_bytes
                    .saturating_sub(evicted.text.len());
            }
        }

        if output_bytes <= MAX_CAPTURED_COMMAND_OUTPUT_BYTES {
            self.captured_command_output_bytes = self
                .captured_command_output_bytes
                .saturating_add(output_bytes);
            if let Some(record) = self.command_records.get_mut(index) {
                record.captured_output = Some(output);
            }
        }
    }

    fn capture_and_queue_completed_command_output(&mut self, index: usize) {
        let Some(record_before_capture) = self.command_records.get(index).cloned() else {
            return;
        };
        let extracted = record_before_capture
            .output_start
            .zip(record_before_capture.output_end)
            .and_then(|(start, end)| {
                self.extract_text_range(start, end, MAX_COMPLETED_COMMAND_OUTPUT_BYTES)
            });
        let output_available = extracted.is_some();
        if let Some(output) = extracted.as_ref() {
            self.store_captured_command_output(index, output.clone());
        }
        let Some(record) = self.command_records.get(index).cloned() else {
            return;
        };
        let extracted = extracted.unwrap_or_default();
        if self.pending_completed_command_outputs.len() >= MAX_PENDING_COMPLETED_COMMANDS {
            self.pending_completed_command_outputs.pop_front();
        }
        self.pending_completed_command_outputs
            .push_back(CompletedCommandEvent {
                start_mark_seen: record.start_mark_seen,
                completion_provenance: record.completion_provenance,
                completed: CompletedCommandOutput {
                    id: record.id,
                    session_id: record.session_id,
                    seq: record.seq,
                    started_at_ms: record.started_at_ms,
                    command: record.command,
                    cwd: record.cwd,
                    exit_code: record.exit_code,
                    duration_ms: record.duration_ms,
                    output: extracted.text,
                    output_available,
                    truncated: extracted.truncated,
                    total_bytes: extracted.total_bytes,
                    agent_generation: record.agent_generation,
                },
            });
    }

    fn queue_agent_termination(
        &mut self,
        id: String,
        command: Option<String>,
        cwd: Option<String>,
        generation: u64,
        start_mark_seen: bool,
        remember_id: bool,
    ) {
        let completed = CompletedCommandEvent {
            start_mark_seen,
            completion_provenance: crate::block_mode::CompletionProvenance::BoundaryInferred,
            completed: CompletedCommandOutput {
                id,
                // A boundary-inferred termination observed no `C` identity of
                // its own; leaving these empty keeps such an event out of the
                // journal by construction rather than by the trust check alone.
                session_id: None,
                seq: None,
                started_at_ms: None,
                command,
                cwd,
                exit_code: None,
                duration_ms: None,
                output: String::new(),
                output_available: false,
                truncated: false,
                total_bytes: 0,
                agent_generation: Some(generation),
            },
        };
        let consumed_id = completed.completed.id.clone();
        if self.pending_completed_command_outputs.len() >= MAX_PENDING_COMPLETED_COMMANDS {
            self.pending_completed_command_outputs.pop_front();
        }
        self.pending_completed_command_outputs.push_back(completed);
        if remember_id {
            self.remember_consumed_command_id(Some(&consumed_id));
        }
    }

    /// An approval can be locally armed before OSC 133 `C` arrives. A fresh
    /// prompt proves that execution never entered the correlated lifecycle;
    /// publish an explicit degraded event instead of silently dropping the
    /// generation and leaving the Agent panel waiting forever.
    fn record_abandoned_armed_agent_command(&mut self, remember_id: bool) {
        let Some(armed) = self.armed_agent_execution.take() else {
            return;
        };
        let record = self
            .command_records
            .iter()
            .find(|record| record.sequence == armed.command_sequence);
        let id = record
            .map(|record| record.id.clone())
            .unwrap_or_else(|| Self::local_command_id(armed.command_sequence));
        let cwd = record.and_then(|record| record.cwd.clone());
        self.queue_agent_termination(
            id,
            Some(armed.command),
            cwd,
            armed.generation,
            false,
            remember_id,
        );
    }

    /// Publish a boundary-inferred termination before RIS replaces the whole
    /// terminal. Prefer an execution already correlated at C; otherwise seal
    /// an approval still armed at the prompt. Ordinary semantic history stays
    /// subject to RIS clearing.
    fn record_reset_interrupted_agent_command(&mut self) {
        let active = self
            .active_record_index()
            .and_then(|index| self.command_records.get(index))
            .and_then(|record| {
                Some((
                    record.id.clone(),
                    record.command.clone(),
                    record.cwd.clone(),
                    record.agent_generation?,
                ))
            });
        if let Some((id, command, cwd, generation)) = active {
            self.queue_agent_termination(id, command, cwd, generation, true, true);
        } else {
            self.record_abandoned_armed_agent_command(true);
        }
    }

    fn finish_command_record(
        &mut self,
        index: usize,
        anchor: BufferAnchor,
        exit_code: Option<i32>,
        duration_ms: Option<u64>,
        completion_provenance: crate::block_mode::CompletionProvenance,
    ) {
        let publish_completion = completion_provenance
            == crate::block_mode::CompletionProvenance::ShellReported
            || self
                .command_records
                .get(index)
                .is_some_and(|record| record.start_mark_seen);
        let exact_output = self
            .command_records
            .get(index)
            .map(|record| record.sequence)
            .and_then(|zone_id| self.finish_active_output_provenance(zone_id));
        if let Some((provenance, _, _)) = exact_output.as_ref() {
            self.register_finished_output_provenance(provenance.clone());
        }
        if let Some(record) = self.command_records.get_mut(index) {
            if let Some((_, output_start, output_end)) = exact_output {
                record.output_start = Some(output_start);
                record.output_end = Some(output_end);
                record.end = Some(output_end);
            } else {
                record.output_end = Some(anchor);
                record.end = Some(anchor);
            }
            record.exit_code = exit_code;
            record.duration_ms = if completion_provenance
                == crate::block_mode::CompletionProvenance::ShellReported
            {
                duration_ms.or_else(|| {
                    record
                        .started_instant
                        .map(|started| started.elapsed().as_millis().min(u64::MAX as u128) as u64)
                })
            } else {
                None
            };
            record.state = CommandState::Complete;
            record.complete = true;
            record.completion_provenance = completion_provenance;
            record.finished_at = (completion_provenance
                == crate::block_mode::CompletionProvenance::ShellReported)
                .then(std::time::SystemTime::now);
            record.started_instant = None;
        }
        if publish_completion {
            self.capture_and_queue_completed_command_output(index);
        }
        let consumed_id = self
            .command_records
            .get(index)
            .map(|record| record.id.clone());
        self.remember_consumed_command_id(consumed_id.as_deref());
    }

    fn record_command_exit_with_metadata(
        &mut self,
        id: Option<&str>,
        command: Option<&str>,
        cwd: Option<&str>,
        exit_code: Option<i32>,
        duration_ms: Option<u64>,
        command_truncated: bool,
    ) {
        if self.use_alt_buffer {
            return;
        }
        let decoded_id = id.and_then(Self::valid_osc_133_id);
        // An explicitly malformed id is not the same as an omitted id. It
        // cannot authorize a fallback close of whichever block happens to be
        // current.
        if id.is_some() && decoded_id.is_none() {
            return;
        }
        if decoded_id
            .as_deref()
            .is_some_and(|id| self.command_id_was_consumed(id))
        {
            return;
        }
        let active = self.active_record_index();
        let index = match decoded_id.as_deref() {
            None => active,
            Some(decoded_id) => match self.record_index_for_id(decoded_id) {
                Some(index) => self
                    .command_records
                    .get(index)
                    .is_some_and(|record| !record.complete)
                    .then_some(index),
                None => {
                    // Some integrations first supply the execution id on D. A
                    // terminal-local placeholder may adopt it; an already
                    // shell-named active record may not be closed by a stale
                    // or out-of-order id belonging to another execution.
                    active.filter(|&index| {
                        self.command_records.get(index).is_some_and(|record| {
                            record.id == Self::local_command_id(record.sequence)
                        })
                    })
                }
            },
        };
        let Some(index) = index else {
            return;
        };
        if self
            .command_records
            .get(index)
            .map(|record| record.complete)
            .unwrap_or(true)
        {
            return;
        }
        // D's cwd is the post-command directory. It must not overwrite the
        // cwd captured at C, which is the authority for Retry/task provenance.
        self.apply_record_metadata(index, id, command, None, command_truncated);
        if let Some(cwd_after) = cwd.and_then(Self::decode_osc_133_cwd) {
            if let Some(record) = self.command_records.get_mut(index) {
                record.cwd_after = Some(cwd_after.clone());
            }
            self.current_working_dir = Some(cwd_after);
        }
        let anchor = self.current_buffer_anchor();
        self.finish_command_record(
            index,
            anchor,
            exit_code,
            duration_ms,
            crate::block_mode::CompletionProvenance::ShellReported,
        );
        if let Some(mark) = self.command_marks.back_mut() {
            mark.exit_code = exit_code;
        }
    }

    /// Decode the one outcome slot supported by an OSC 133 `D` packet.
    ///
    /// FinalTerm places the status positionally, while the family's own
    /// integrations also spell it `exit`/`exit_code`/`exit_status`. A named
    /// spelling is a slot even when its value is junk, so a trailing
    /// `exit=oops` cannot quietly erase a status already read positionally;
    /// an attempted *second* slot is ambiguous whichever way either value
    /// parses, so it yields no status at all rather than letting a repeat flip
    /// a failure badge to success. A bare non-numeric field is an unknown
    /// extension flag, not an outcome.
    ///
    /// This mirrors `jterm_core::parser::parse_osc133_exit_status`; the two
    /// decoders read the same packets and must agree about what a status is.
    fn parse_osc_133_exit_status<'a>(fields: impl Iterator<Item = &'a str>) -> Option<i32> {
        let mut seen = false;
        let mut parsed = None;
        for field in fields {
            let candidate = match field.split_once('=') {
                Some(("exit" | "exit_code" | "exit_status", value)) => {
                    Some(value.trim().parse::<i32>().ok())
                }
                Some(_) => None,
                None => field.trim().parse::<i32>().ok().map(Some),
            };
            let Some(candidate) = candidate else {
                continue;
            };
            if std::mem::replace(&mut seen, true) {
                return None;
            }
            parsed = candidate;
        }
        parsed
    }

    /// Parse and apply one OSC 133 payload (the part after `133;`). Supports
    /// FinalTerm A/B/C/D, Kitty `cmdline_url`, and jsh correlation metadata.
    ///
    /// Aliases name one semantic slot, and every slot is single-assignment.
    /// The payload is untrusted PTY output: last-wins would let a second
    /// spelling of the same key overwrite a journal correlation id, a command,
    /// a cwd or a truncation disclosure that the honest first spelling already
    /// established. A repeated slot therefore degrades to "absent" — except the
    /// truncation disclosure, where "absent" would mean "the command is
    /// complete", so a repeat fails closed to truncated instead. This is
    /// `jterm_core::parser::CommandMeta::from_fields`'s rule, and the two
    /// decoders read the same packets.
    pub(super) fn handle_osc_133(&mut self, value: &str) {
        let mut parts = value.split(';');
        let kind = parts.next().unwrap_or("");
        // `session_id`, `seq` and `started_at_ms` are Start identity: jsh sends
        // them on `C` and never on `D`, so accepting them anywhere else would
        // let a completion packet mint a lifecycle token for a Start generation
        // this terminal never observed. Same gate as the shared parser's
        // `accept_start_identity`.
        let accept_start_identity = kind == "C";
        let exit_code = (kind == "D")
            .then(|| Self::parse_osc_133_exit_status(parts.clone()))
            .flatten();
        let mut id = None;
        let mut session_id = None;
        let mut seq = None;
        let mut started_at_ms = None;
        let mut command = None;
        let mut cwd = None;
        let mut duration_ms = None;
        let mut command_truncated = false;
        let mut seen_id = false;
        let mut seen_session_id = false;
        let mut seen_seq = false;
        let mut seen_started_at_ms = false;
        let mut seen_command = false;
        let mut seen_cwd = false;
        let mut seen_duration = false;
        let mut seen_command_truncated = false;

        for part in parts {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            match key {
                "id" | "jsh_id" | "execution_id" | "command_id" => {
                    if std::mem::replace(&mut seen_id, true) {
                        id = None;
                        continue;
                    }
                    id = Some(value);
                }
                "session_id" => {
                    if !accept_start_identity {
                        continue;
                    }
                    if std::mem::replace(&mut seen_session_id, true) {
                        session_id = None;
                        continue;
                    }
                    // Percent-decoded, then held to jsh's exact session
                    // grammar: this value becomes a journal routing key.
                    session_id = Self::percent_decode_osc_133(value, MAX_OSC_133_SESSION_ID_BYTES)
                        .ok()
                        .filter(|id| jterm_core::execution_journal::is_valid_jsh_session_id(id));
                }
                "seq" => {
                    if !accept_start_identity {
                        continue;
                    }
                    if std::mem::replace(&mut seen_seq, true) {
                        seq = None;
                        continue;
                    }
                    // Integers, not text: jsh emits them unencoded and the
                    // shared parser reads them the same way.
                    seq = value.parse::<u64>().ok();
                }
                "started_at_ms" => {
                    if !accept_start_identity {
                        continue;
                    }
                    if std::mem::replace(&mut seen_started_at_ms, true) {
                        started_at_ms = None;
                        continue;
                    }
                    started_at_ms = value.parse::<u64>().ok();
                }
                "cmdline_url" | "command_url" | "command" | "cmdline" => {
                    if std::mem::replace(&mut seen_command, true) {
                        command = None;
                        continue;
                    }
                    command = Some(value);
                }
                "cwd" | "cwd_url" => {
                    if std::mem::replace(&mut seen_cwd, true) {
                        cwd = None;
                        continue;
                    }
                    cwd = Some(value);
                }
                "duration" | "duration_ms" => {
                    if std::mem::replace(&mut seen_duration, true) {
                        duration_ms = None;
                        continue;
                    }
                    duration_ms = value.trim().parse::<u64>().ok();
                }
                "cmd_truncated" | "command_truncated" => {
                    if std::mem::replace(&mut seen_command_truncated, true) {
                        // A second disclosure is ambiguous. "Not truncated" is
                        // the answer that re-enables replay of a partial
                        // command, so it is the one a repeat may not produce.
                        command_truncated = true;
                        continue;
                    }
                    let value = value.trim();
                    command_truncated = match value {
                        "0" => false,
                        "1" => true,
                        value if value.eq_ignore_ascii_case("false") => false,
                        value if value.eq_ignore_ascii_case("true") => true,
                        // The producer chose to send the disclosure but did not
                        // encode its state. That is inexact, not the default
                        // `false` that claims the command is complete.
                        _ => true,
                    };
                }
                _ => {}
            }
        }

        // jsh emits either an exact command or `cmd_truncated=1`, never both.
        // A contradictory producer must not smuggle a partial prefix into the
        // exact-command path merely by attaching the honest disclosure too.
        if command_truncated {
            command = None;
        }

        match kind {
            "A" => {
                self.record_prompt_start_with_metadata(id, command, cwd, command_truncated);
            }
            "B" => self.record_command_start(id, command, cwd, command_truncated),
            "C" => self.record_output_start(
                id,
                command,
                cwd,
                command_truncated,
                Osc133StartIdentity {
                    session_id,
                    seq,
                    started_at_ms,
                },
            ),
            "D" => self.record_command_exit_with_metadata(
                id,
                command,
                cwd,
                exit_code,
                duration_ms,
                command_truncated,
            ),
            _ => {}
        }
    }

    /// Canonical semantic command history in execution order. Records remain
    /// listed after their terminal rows are evicted; range extraction/jumping
    /// then returns `None`/`false` for the unavailable anchors.
    pub fn command_records(&self) -> &VecDeque<CommandRecord> {
        &self.command_records
    }

    pub fn command_record(&self, id: &str) -> Option<&CommandRecord> {
        self.record_index_for_id(id)
            .and_then(|index| self.command_records.get(index))
    }

    /// Remove every finalized OSC 133 record while leaving the live prompt (or
    /// the command currently running) intact — the terminal-state half of
    /// Warp's "Clear Blocks", adapted from frost's `clear_completed_blocks`.
    ///
    /// frost also discards the cleared blocks' buffer rows. ember deliberately
    /// keeps them: the record deque is the canonical block representation here
    /// (records already survive row eviction), and row surgery could not be
    /// undone cell-exactly through the projection/raw-row-id layers. Cleared
    /// output therefore stays as ordinary scrollback text while every badge,
    /// gutter stripe, navigation target, and export record derived from the
    /// removed blocks disappears; the records themselves are stashed so
    /// [`Self::undo_clear_blocks`] can restore them. Legacy `command_marks`
    /// navigation keeps working against the retained rows. Record sequences
    /// stay monotonic, so a stale UI id can never target a record created
    /// after the clear.
    ///
    /// Returns how many records were cleared. An empty result leaves any
    /// existing undo stash untouched, so a reflexive second clear cannot
    /// destroy a real undo snapshot (anvil/forge semantics).
    pub fn clear_completed_blocks(&mut self) -> usize {
        let mut retained = VecDeque::with_capacity(self.command_records.len());
        let mut cleared = Vec::new();
        for record in std::mem::take(&mut self.command_records) {
            if record.complete {
                cleared.push(record);
            } else {
                retained.push_back(record);
            }
        }
        self.command_records = retained;
        let cleared_count = cleared.len();
        if cleared_count == 0 {
            return 0;
        }

        let mut captured_output_bytes = 0usize;
        let mut provenance = Vec::new();
        for record in &cleared {
            captured_output_bytes = captured_output_bytes.saturating_add(
                record
                    .captured_output
                    .as_ref()
                    .map(|output| output.text.len())
                    .unwrap_or(0),
            );
            if let Some(entry) = self
                .finished_output_provenance
                .get(&record.sequence)
                .cloned()
            {
                provenance.push(entry);
            }
            self.unregister_finished_output_zone(record.sequence);
        }
        self.captured_command_output_bytes = self
            .captured_command_output_bytes
            .saturating_sub(captured_output_bytes);
        self.cleared_blocks_stash = Some(ClearedBlocksStash {
            records: cleared,
            provenance,
            captured_output_bytes,
        });
        cleared_count
    }

    /// Restore the records removed by the most recent
    /// [`Self::clear_completed_blocks`]. They are older than anything created
    /// since, so they re-enter ahead of the retained deque (anvil/forge
    /// prepend semantics) together with their finished-output sidecars.
    ///
    /// The `MAX_COMMAND_MARKS` bound is then enforced exactly like a natural
    /// eviction: the oldest restored records are dropped first, and a still
    /// live prompt/running record is never evicted. The buffer was never
    /// touched by the clear, so restored anchors and provenance revalidate
    /// against the same rows unless ordinary scrollback eviction removed them
    /// in the meantime — in which case the existing fail-closed paths apply.
    /// Returns how many records were actually restored; consumes the stash
    /// either way (single-level undo).
    pub fn undo_clear_blocks(&mut self) -> usize {
        let Some(stash) = self.cleared_blocks_stash.take() else {
            return 0;
        };
        let stashed = stash.records.len();
        if stashed == 0 {
            return 0;
        }
        for entry in stash.provenance {
            self.register_finished_output_provenance(entry);
        }
        for record in stash.records.into_iter().rev() {
            self.command_records.push_front(record);
        }
        self.captured_command_output_bytes = self
            .captured_command_output_bytes
            .saturating_add(stash.captured_output_bytes);

        let mut evicted = 0usize;
        while self.command_records.len() > MAX_COMMAND_MARKS
            && self
                .command_records
                .front()
                .is_some_and(|record| record.complete)
        {
            if let Some(record) = self.command_records.pop_front() {
                evicted = evicted.saturating_add(1);
                self.unregister_finished_output_zone(record.sequence);
                self.captured_command_output_bytes =
                    self.captured_command_output_bytes.saturating_sub(
                        record
                            .captured_output
                            .as_ref()
                            .map(|output| output.text.len())
                            .unwrap_or(0),
                    );
            }
        }
        stashed.saturating_sub(evicted.min(stashed))
    }

    /// Exact half-open raw output range for one completed command sequence.
    /// Every retained row identity is revalidated so eviction, replacement,
    /// resize, or stale ids fail closed instead of retargeting terminal cells.
    #[allow(dead_code)] // Public terminal contract; projection policy wiring lands next.
    pub fn finished_output_range(&self, zone_id: u64) -> Option<FinishedOutputRange> {
        self.command_records
            .iter()
            .any(|record| record.sequence == zone_id && record.complete)
            .then_some(())?;
        let provenance = self.finished_output_provenance.get(&zone_id)?;
        if provenance.range.zone_id != zone_id || provenance.rows.is_empty() {
            return None;
        }
        let preferred_start = BufferAnchor {
            line_id: provenance.start_line_id,
            column: provenance.range.start.col,
        };
        let (start_absolute, _) =
            self.retained_raw_row_absolute(provenance.range.start.row, Some(preferred_start))?;
        for (offset, expected) in provenance.rows.iter().enumerate() {
            if self
                .retained_raw_row(start_absolute.saturating_add(offset))?
                .0
                != expected.row_id
            {
                return None;
            }
        }
        let end_absolute = start_absolute.checked_add(provenance.rows.len().checked_sub(1)?)?;
        let (start_id, start_width) = self.retained_raw_row(start_absolute)?;
        let (end_id, end_width) = self.retained_raw_row(end_absolute)?;
        (start_id == provenance.range.start.row
            && end_id == provenance.range.end.row
            && provenance.range.start.col <= start_width
            && provenance.range.end.col <= end_width
            && (start_absolute < end_absolute
                || provenance.range.start.col < provenance.range.end.col))
            .then_some(provenance.range)
    }

    /// Revision of the retained completed-output ownership index. Callers may
    /// use this to avoid revalidating unchanged collapse ids every frame.
    pub fn finished_output_revision(&self) -> u64 {
        self.finished_output_revision
    }

    /// Whether the current editable prompt is provably empty. Block recall is
    /// intentionally stricter than an ordinary paste: it must never erase or
    /// append to text the user already entered while reviewing a selection.
    pub fn prompt_input_is_empty(&self) -> bool {
        if self.agent_prompt_input_tainted || !self.shell_is_prompt_ready() {
            return false;
        }
        let Some(record) = self.command_records.back() else {
            return false;
        };
        let Some(start) = record.command_start else {
            return false;
        };
        self.extract_text_range(
            start,
            self.current_buffer_anchor(),
            MAX_OSC_133_COMMAND_BYTES,
        )
        .is_some_and(|text| text.text.is_empty())
    }

    /// The command the shell reported as running via OSC 133, if any.
    ///
    /// Only the newest record can be running; an earlier one still marked
    /// `Running` means its `D` was lost, and reporting it in a pane header
    /// would pin a stale command there forever.
    pub fn running_command(&self) -> Option<&str> {
        self.command_records
            .back()
            .filter(|record| !record.complete && record.state == CommandState::Running)
            .and_then(|record| record.command.as_deref())
            .map(str::trim)
            .filter(|command| !command.is_empty())
    }

    /// Elapsed wall time for the newest live OSC 133 command. This is
    /// renderer-only state: completed records still prefer the shell-reported
    /// duration at `D`, while the live block badge uses this monotonic clock.
    pub fn running_duration_ms(&self) -> Option<u64> {
        self.command_records
            .back()
            .filter(|record| !record.complete && record.state == CommandState::Running)
            .and_then(|record| record.started_instant)
            .map(|started| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
    }

    /// True after B and before C. This is the safe state for placing a
    /// command into the shell editor without racing a running foreground job.
    pub fn shell_is_prompt_ready(&self) -> bool {
        self.command_records
            .back()
            .map(|record| !record.complete && record.state == CommandState::Editing)
            .unwrap_or(false)
    }

    /// Whether this terminal has observed at least one OSC 133 prompt mark.
    ///
    /// Block actions use this to distinguish an ordinary empty result from a
    /// pane whose shell is not reporting semantic command boundaries at all.
    pub fn has_prompt_marks(&self) -> bool {
        !self.command_records.is_empty() || !self.command_marks.is_empty()
    }

    /// Record accepted non-Agent input before PTY echo arrives. Clearing the
    /// visible line does not silently re-authorize an approval; only a fresh
    /// OSC 133 prompt resets this bit.
    pub fn note_user_input(&mut self, input: &[u8]) {
        if !input.is_empty() && self.shell_is_prompt_ready() {
            self.agent_prompt_input_tainted = true;
        }
    }

    /// Arm one approved command on the current fresh, empty prompt. The
    /// generation is application-local and is consumed at OSC 133 C.
    pub fn arm_agent_execution(
        &mut self,
        generation: u64,
        command: &str,
    ) -> Result<(), &'static str> {
        if generation == 0 || command.is_empty() {
            return Err("invalid Agent execution identity");
        }
        if crate::review_text::validate_single_line(
            command,
            crate::review_text::MAX_AGENT_COMMAND_BYTES,
        )
        .is_err()
        {
            return Err("the Agent command is unsafe to review or execute");
        }
        if self.agent_prompt_input_tainted {
            return Err("the shell prompt already contains local input");
        }
        if self.armed_agent_execution.is_some() {
            return Err("another Agent command is already armed");
        }
        let Some(record) = self.command_records.back() else {
            return Err("shell integration has not reported a prompt");
        };
        if record.complete || record.state != CommandState::Editing {
            return Err("the shell is not waiting at an editable prompt");
        }
        let sequence = record.sequence;
        let visible_input = record
            .command_start
            .and_then(|start| {
                self.extract_text_range(
                    start,
                    self.current_buffer_anchor(),
                    MAX_OSC_133_COMMAND_BYTES,
                )
            })
            .map(|text| text.text.trim().is_empty())
            // A missing/evicted anchor means we cannot prove the prompt is
            // empty, so approval must fail closed.
            .unwrap_or(false);
        if !visible_input {
            return Err("the shell prompt is not empty");
        }
        self.armed_agent_execution = Some(ArmedAgentExecution {
            generation,
            command_sequence: sequence,
            command: command.to_string(),
        });
        Ok(())
    }

    pub fn disarm_agent_execution(&mut self, generation: u64) {
        if self
            .armed_agent_execution
            .as_ref()
            .is_some_and(|armed| armed.generation == generation)
        {
            self.armed_agent_execution = None;
        }
    }

    #[allow(dead_code)] // Public library compatibility; the app consumes provenance-aware events.
    pub fn take_completed_command_outputs(&mut self) -> Vec<CompletedCommandOutput> {
        self.pending_completed_command_outputs
            .drain(..)
            .filter(|event| {
                event.completion_provenance
                    == crate::block_mode::CompletionProvenance::ShellReported
            })
            .map(|event| event.completed)
            .collect()
    }

    /// Provenance-aware completion drain for application consumers. The
    /// source-compatible output-only API above remains available to library
    /// users that do not yet distinguish inferred lifecycle termination.
    pub fn take_completed_command_events(&mut self) -> Vec<CompletedCommandEvent> {
        self.pending_completed_command_outputs.drain(..).collect()
    }

    /// Resolve a stable line-id anchor into the current raw terminal buffer
    /// (`scrollback` followed by the live grid).
    pub fn buffer_anchor_to_absolute(&self, anchor: BufferAnchor) -> Option<(usize, usize)> {
        let first_scrollback_line_id = self
            .total_lines_scrolled
            .saturating_sub(self.scrollback.len() as u64);
        let absolute_row = if anchor.line_id < self.total_lines_scrolled {
            if anchor.line_id < first_scrollback_line_id {
                return None;
            }
            (anchor.line_id - first_scrollback_line_id) as usize
        } else {
            let grid_row = (anchor.line_id - self.total_lines_scrolled) as usize;
            if grid_row >= self.grid.rows() {
                return None;
            }
            self.scrollback.len().saturating_add(grid_row)
        };
        Some((absolute_row, anchor.column))
    }

    /// Monotonic line id of the top visible viewport row. Only a meaningful
    /// per-row mapping when [`Self::viewport_buffer_mapping_is_exact`]; block
    /// chrome shares that gate with search overlays.
    pub fn viewport_top_line_id(&self) -> u64 {
        self.total_lines_scrolled
            .saturating_sub(self.scroll_offset as u64)
    }

    /// Whether raw scrollback rows and the currently rendered visual rows have
    /// a one-to-one coordinate mapping. Historical lines are reflowed lazily
    /// after a width change; until the terminal model exposes per-cell origins,
    /// drawing a raw-column search span there would confidently highlight the
    /// wrong cell. Callers should omit that overlay instead.
    pub fn viewport_buffer_mapping_is_exact(&self) -> bool {
        if self.scroll_offset == 0 {
            return true;
        }
        let cols = self.grid.row_len();
        let rows = self.grid.rows();
        let cache_key = ViewportMappingExactCache {
            cols,
            rows,
            scroll_offset: self.scroll_offset,
            scrollback_len: self.scrollback.len(),
            total_lines_scrolled: self.total_lines_scrolled,
            exact: false,
        };
        let cached_exact = self.viewport_mapping_exact_cache.get().and_then(|cached| {
            (cached.cols == cache_key.cols
                && cached.rows == cache_key.rows
                && cached.scroll_offset == cache_key.scroll_offset
                && cached.scrollback_len == cache_key.scrollback_len
                && cached.total_lines_scrolled == cache_key.total_lines_scrolled)
                .then_some(cached.exact)
        });
        if let Some(exact) = cached_exact {
            return exact;
        }
        let mut start = self
            .scrollback
            .len()
            .saturating_sub(self.scroll_offset.saturating_add(rows));
        while start > 0 && self.scrollback[start - 1].is_wrapped {
            start -= 1;
        }
        let exact = self
            .scrollback
            .iter()
            .skip(start)
            .all(|line| line.columns() == cols && !line.is_wrapped);
        self.viewport_mapping_exact_cache
            .set(Some(ViewportMappingExactCache { exact, ..cache_key }));
        exact
    }

    /// Resolve a stable buffer anchor into the current viewport using the
    /// same absolute-row semantics as text selection. This intentionally
    /// keeps search and selection aligned across resize/reflow until both can
    /// share a richer logical-line mapping.
    pub fn buffer_anchor_to_viewport(&self, anchor: BufferAnchor) -> Option<(usize, usize)> {
        if !self.viewport_buffer_mapping_is_exact() {
            return None;
        }
        let (absolute_row, column) = self.buffer_anchor_to_absolute(anchor)?;
        self.absolute_row_to_viewport(absolute_row)
            .map(|viewport_row| (viewport_row, column))
    }

    /// Resolve a stable buffer anchor through the exact provenance of a
    /// projected viewport. Hidden cells and structural rows fail closed.
    pub fn buffer_anchor_to_projected(
        &self,
        viewport: &ProjectedViewport,
        anchor: BufferAnchor,
    ) -> Option<DisplayPoint> {
        let (absolute_row, column) = self.buffer_anchor_to_absolute(anchor)?;
        let row_id = self.raw_row_id_at_absolute(absolute_row)?;
        viewport.display_point_for(RawCellAnchor { row_id, column })
    }

    /// Classify and, when visible, reveal a stable buffer anchor in the exact
    /// current projected plan. Hidden output is reported to the caller but is
    /// never expanded here; stale or ambiguous provenance fails closed.
    #[allow(dead_code)] // Public library/search hook; binary wiring may be feature-gated.
    pub fn reveal_buffer_anchor_in_projection(
        &mut self,
        policy: &ProjectionPolicy,
        view_state: &mut ProjectionViewState,
        anchor: BufferAnchor,
    ) -> ProjectedBufferAnchorLocation {
        if policy.is_identity() || self.use_alt_buffer {
            return ProjectedBufferAnchorLocation::Identity;
        }
        let Some((absolute_row, column)) = self.buffer_anchor_to_absolute(anchor) else {
            return ProjectedBufferAnchorLocation::Unmapped;
        };
        let Some(row_id) = self.raw_row_id_at_absolute(absolute_row) else {
            return ProjectedBufferAnchorLocation::Unmapped;
        };
        let raw = RawCellAnchor { row_id, column };
        let cols = self.grid.row_len().max(1);
        let plan_key = self.projection_plan_cache_key(cols, policy);
        let Some(plan) = self.cached_collapsed_projection_plan(cols, policy) else {
            return ProjectedBufferAnchorLocation::Unmapped;
        };
        if plan.plan_revision == 0 || !self.projection_plan_key_matches_current_source(&plan_key) {
            return ProjectedBufferAnchorLocation::Unmapped;
        }
        if let Some(summary_row) = plan.summary_owning_raw_cell(raw) {
            return match plan.row(summary_row).map(|row| row.kind) {
                Some(ProjectedRowKind::CollapsedSummary { key, .. })
                    if key.policy_revision == policy.revision() =>
                {
                    ProjectedBufferAnchorLocation::Hidden {
                        zone_id: key.zone_id,
                    }
                }
                _ => ProjectedBufferAnchorLocation::Unmapped,
            };
        }
        let Some(document_row) = plan
            .raw_cell_document_row(raw)
            .or_else(|| plan.raw_row_document_row(raw.row_id))
        else {
            return ProjectedBufferAnchorLocation::Unmapped;
        };
        let max_start = plan.document_rows().saturating_sub(self.grid.rows());
        view_state.offset_from_bottom = max_start.saturating_sub(document_row.min(max_start));
        view_state.follow_bottom = false;
        view_state.top_anchor = Some(ProjectedTopAnchor::RawCell(raw));
        view_state.last_plan_key = Some(plan_key);
        ProjectedBufferAnchorLocation::Visible { document_row }
    }

    /// Reveal the exact synthetic row owned by one effective collapse without
    /// changing the requested policy. This is used by block-bottom navigation:
    /// a hidden raw edge must land on its summary, never a neighbouring row.
    pub fn reveal_collapsed_summary(
        &mut self,
        policy: &ProjectionPolicy,
        view_state: &mut ProjectionViewState,
        zone_id: u64,
    ) -> bool {
        if self.use_alt_buffer || !policy.is_collapsed(zone_id) {
            return false;
        }
        let cols = self.grid.row_len().max(1);
        let plan_key = self.projection_plan_cache_key(cols, policy);
        let Some(plan) = self.cached_collapsed_projection_plan(cols, policy) else {
            return false;
        };
        let Some((document_row, hidden_range)) =
            plan.rows
                .iter()
                .enumerate()
                .find_map(|(row, planned)| match planned.kind {
                    ProjectedRowKind::CollapsedSummary {
                        key, hidden_range, ..
                    } if key.zone_id == zone_id && key.policy_revision == policy.revision() => {
                        Some((row, hidden_range))
                    }
                    _ => None,
                })
        else {
            return false;
        };
        let max_start = plan.document_rows().saturating_sub(self.grid.rows());
        view_state.offset_from_bottom = max_start.saturating_sub(document_row.min(max_start));
        view_state.follow_bottom = false;
        view_state.top_anchor = Some(ProjectedTopAnchor::Summary {
            zone_id,
            hidden_range,
        });
        view_state.last_plan_key = Some(plan_key);
        true
    }

    /// Cheap render invalidation token for terminal text selection. The raw
    /// selection value remains public for compatibility; transformed changes
    /// advance this token without exposing projected document coordinates.
    pub fn selection_revision(&self) -> u64 {
        self.selection_revision
    }

    /// Scroll enough to reveal a stable buffer anchor. Historical matches are
    /// placed at the top of the viewport; live-grid matches return to the live
    /// tail. If the row is already visible, the current viewport is preserved.
    pub fn scroll_to_buffer_anchor(&mut self, anchor: BufferAnchor) -> bool {
        let Some((absolute_row, _)) = self.buffer_anchor_to_absolute(anchor) else {
            return false;
        };
        if self.absolute_row_to_viewport(absolute_row).is_some() {
            return true;
        }

        if absolute_row < self.scrollback.len() {
            self.scroll_offset = self
                .scrollback
                .len()
                .saturating_sub(absolute_row)
                .min(self.scrollback.len());
        } else {
            self.scroll_offset = 0;
        }
        true
    }

    /// Convert a current raw-buffer coordinate to a stable line-id anchor.
    #[allow(dead_code)] // Public library surface for other jterm frontends.
    pub fn absolute_to_buffer_anchor(&self, absolute: (usize, usize)) -> Option<BufferAnchor> {
        let (row, column) = absolute;
        let line_id = if row < self.scrollback.len() {
            self.total_lines_scrolled
                .saturating_sub(self.scrollback.len() as u64)
                .saturating_add(row as u64)
        } else {
            let grid_row = row - self.scrollback.len();
            if grid_row >= self.grid.rows() {
                return None;
            }
            self.total_lines_scrolled.saturating_add(grid_row as u64)
        };
        Some(BufferAnchor { line_id, column })
    }

    fn absolute_row_cells(&self, absolute_row: usize) -> Option<Vec<TerminalCell>> {
        if absolute_row < self.scrollback.len() {
            return self
                .scrollback
                .get(absolute_row)
                .map(ScrollbackLine::decompress);
        }
        let grid_row = absolute_row - self.scrollback.len();
        (grid_row < self.grid.rows()).then(|| self.grid[grid_row].to_vec())
    }

    fn absolute_row_is_wrapped(&self, absolute_row: usize) -> Option<bool> {
        if absolute_row < self.scrollback.len() {
            return self
                .scrollback
                .get(absolute_row)
                .map(|line| line.is_wrapped);
        }
        let grid_row = absolute_row - self.scrollback.len();
        self.grid.row_wrapped.get(grid_row).copied()
    }

    /// Extract normalized display text from `[start, end)`. Soft-wrapped rows
    /// are joined without a newline, hard row boundaries retain one newline,
    /// right-padding and wide-character continuation cells are omitted, and
    /// the returned allocation never exceeds `max_bytes`.
    pub fn extract_text_range(
        &self,
        start: BufferAnchor,
        end: BufferAnchor,
        max_bytes: usize,
    ) -> Option<ExtractedText> {
        if end < start {
            return None;
        }
        let (start_row, start_col) = self.buffer_anchor_to_absolute(start)?;
        let (end_row, end_col) = self.buffer_anchor_to_absolute(end)?;
        if end_row < start_row || (end_row == start_row && end_col < start_col) {
            return None;
        }

        let mut extracted = BoundedTextBuilder::new(max_bytes);
        for absolute_row in start_row..=end_row {
            let cells = self.absolute_row_cells(absolute_row)?;
            let row_start = if absolute_row == start_row {
                start_col.min(cells.len())
            } else {
                0
            };
            let mut row_end = if absolute_row == end_row {
                end_col.min(cells.len())
            } else {
                cells.len()
            };
            while row_end > row_start
                && matches!(cells[row_end - 1].character, ' ' | '\0')
                && !cells[row_end - 1].flags.wide_continuation()
            {
                row_end -= 1;
            }
            for cell in &cells[row_start..row_end] {
                if !cell.flags.wide_continuation() {
                    extracted.push(cell.character);
                }
            }

            if absolute_row < end_row
                && !self.absolute_row_is_wrapped(absolute_row).unwrap_or(false)
            {
                extracted.push('\n');
            }
        }
        Some(extracted.finish())
    }

    /// Same extraction API for callers that already hold raw absolute buffer
    /// coordinates (for example selection/search results).
    #[allow(dead_code)] // Public library surface for other jterm frontends.
    pub fn extract_absolute_text_range(
        &self,
        start: (usize, usize),
        end: (usize, usize),
        max_bytes: usize,
    ) -> Option<ExtractedText> {
        let start = self.absolute_to_buffer_anchor(start)?;
        let end = self.absolute_to_buffer_anchor(end)?;
        self.extract_text_range(start, end, max_bytes)
    }

    /// Extract full rows for an inclusive stable line-id range.
    #[allow(dead_code)] // Public library surface for other jterm frontends.
    pub fn extract_text_by_line_ids(
        &self,
        start_line_id: u64,
        end_line_id: u64,
        max_bytes: usize,
    ) -> Option<ExtractedText> {
        if end_line_id < start_line_id {
            return None;
        }
        let end_absolute = self.buffer_anchor_to_absolute(BufferAnchor {
            line_id: end_line_id,
            column: 0,
        })?;
        let end_cells = self.absolute_row_cells(end_absolute.0)?;
        self.extract_text_range(
            BufferAnchor {
                line_id: start_line_id,
                column: 0,
            },
            BufferAnchor {
                line_id: end_line_id,
                column: end_cells.len(),
            },
            max_bytes,
        )
    }

    pub fn command_output_text(&self, id: &str, max_bytes: usize) -> Option<ExtractedText> {
        let record = self.command_record(id)?;
        if let Some(captured) = record.captured_output.as_ref() {
            if captured.text.len() <= max_bytes {
                return Some(captured.clone());
            }
            let mut builder = BoundedTextBuilder::new(max_bytes);
            for ch in captured.text.chars() {
                builder.push(ch);
            }
            let mut recapped = builder.finish();
            recapped.total_bytes = captured.total_bytes;
            recapped.truncated = true;
            return Some(recapped);
        }
        let start = record.output_start?;
        let end = record
            .output_end
            .unwrap_or_else(|| self.current_buffer_anchor());
        self.extract_text_range(start, end, max_bytes)
    }

    /// Scroll directly to a semantic command by id.
    pub fn scroll_to_command(&mut self, id: &str) -> bool {
        if self.use_alt_buffer {
            return false;
        }
        let Some(anchor) = self.command_record(id).map(|record| record.prompt_start) else {
            return false;
        };
        if self.buffer_anchor_to_absolute(anchor).is_none() {
            return false;
        }
        self.scroll_to_line_id(anchor.line_id)
    }

    /// Resolve the retained physical row containing a 1-based logical output
    /// line. Soft-wrapped rows stay in one logical line. Captured snapshots
    /// may outlive their raw anchors, so failure is expected and callers fall
    /// back to the block header without guessing a coordinate.
    pub fn command_output_line_anchor(&self, id: &str, line_no: usize) -> Option<BufferAnchor> {
        if self.use_alt_buffer || line_no == 0 {
            return None;
        }
        let record = self.command_record(id)?;
        let start = record.output_start?;
        let end = record.output_end?;
        let (start_row, _) = self.buffer_anchor_to_absolute(start)?;
        let (end_row, end_col) = self.buffer_anchor_to_absolute(end)?;
        if end_row < start_row {
            return None;
        }
        let last_row = if end_col == 0 && end_row > start_row {
            end_row - 1
        } else {
            end_row
        };
        let mut logical_line = 1usize;
        for row in start_row..=last_row {
            if logical_line == line_no {
                return self.absolute_to_buffer_anchor((row, 0));
            }
            if row < last_row && !self.absolute_row_is_wrapped(row)? {
                logical_line = logical_line.checked_add(1)?;
            }
        }
        None
    }

    /// Resolve the retained physical row containing the start of a cached
    /// search match. `match_start..match_end` is a Unicode-scalar range in the
    /// complete original logical line. The walk mirrors `extract_text_range`:
    /// the first output column is honored, trailing padding and wide
    /// continuations do not become characters, and soft wraps concatenate.
    /// The whole span is validated, so a captured snapshot that outlived or
    /// diverged from live rows fails closed instead of targeting unrelated
    /// terminal content.
    pub fn command_output_match_anchor(
        &self,
        id: &str,
        line_no: usize,
        match_start: usize,
        match_end: usize,
    ) -> Option<BufferAnchor> {
        if self.use_alt_buffer || line_no == 0 || match_start >= match_end {
            return None;
        }
        let record = self.command_record(id)?;
        let output_start = record.output_start?;
        let output_end = record.output_end?;
        let (output_start_row, output_start_col) = self.buffer_anchor_to_absolute(output_start)?;
        let (output_end_row, output_end_col) = self.buffer_anchor_to_absolute(output_end)?;
        if output_end_row < output_start_row
            || (output_end_row == output_start_row && output_end_col < output_start_col)
        {
            return None;
        }
        let line_start = self.command_output_line_anchor(id, line_no)?;
        let (line_start_row, _) = self.buffer_anchor_to_absolute(line_start)?;
        let last_row = if output_end_col == 0 && output_end_row > output_start_row {
            output_end_row - 1
        } else {
            output_end_row
        };
        if line_start_row > last_row {
            return None;
        }

        let mut remaining_start = match_start;
        let mut remaining_end = match_end;
        let mut target_row = None;
        for row in line_start_row..=last_row {
            let cells = self.absolute_row_cells(row)?;
            let row_start = if row == output_start_row {
                output_start_col.min(cells.len())
            } else {
                0
            };
            let mut row_end = if row == output_end_row {
                output_end_col.min(cells.len())
            } else {
                cells.len()
            };
            while row_end > row_start
                && matches!(cells[row_end - 1].character, ' ' | '\0')
                && !cells[row_end - 1].flags.wide_continuation()
            {
                row_end -= 1;
            }
            let chars = cells[row_start..row_end]
                .iter()
                .filter(|cell| !cell.flags.wide_continuation())
                .count();
            if target_row.is_none() && remaining_start < chars {
                target_row = Some(row);
            }
            if remaining_end <= chars {
                return target_row.and_then(|row| self.absolute_to_buffer_anchor((row, 0)));
            }
            if !self.absolute_row_is_wrapped(row)? {
                return None;
            }
            remaining_start = remaining_start.saturating_sub(chars);
            remaining_end = remaining_end.saturating_sub(chars);
        }
        None
    }

    /// Scroll the selected block's semantic top or bottom edge into view.
    /// Bottom navigation aligns the edge with the viewport bottom when it is
    /// retained, matching the private-scroll card action in anvil/forge while
    /// keeping Ember's one continuous grid.
    pub fn command_edge_anchor(&self, id: &str, bottom: bool) -> Option<BufferAnchor> {
        if self.use_alt_buffer {
            return None;
        }
        let index = self.record_index_for_id(id)?;
        let cols = self.grid.row_len();
        let normalized = |anchor: BufferAnchor| {
            if cols > 0 && anchor.column >= cols {
                anchor.line_id.saturating_add(1)
            } else {
                anchor.line_id
            }
        };
        let start = normalized(self.command_records[index].prompt_start);
        let target = if bottom {
            self.command_records
                .get(index + 1)
                .map(|record| normalized(record.prompt_start).saturating_sub(1))
                .or_else(|| self.command_records[index].end.map(normalized))
                .unwrap_or(start)
        } else {
            start
        };
        let anchor = BufferAnchor {
            line_id: target,
            column: 0,
        };
        self.buffer_anchor_to_absolute(anchor)?;
        Some(anchor)
    }

    pub fn scroll_to_command_edge(&mut self, id: &str, bottom: bool) -> bool {
        let Some(target) = self.command_edge_anchor(id, bottom) else {
            return false;
        };
        if !bottom {
            return self.scroll_to_line_id(target.line_id);
        }
        let desired_top = target
            .line_id
            .saturating_sub(self.grid.rows().saturating_sub(1) as u64);
        self.scroll_to_line_id(desired_top) || self.scroll_to_line_id(target.line_id)
    }

    /// Scroll the viewport so the row at `line_id` lands at the top of
    /// the visible area (or as close as possible). Returns true if the
    /// jump did anything.
    fn scroll_to_line_id(&mut self, line_id: u64) -> bool {
        if let Some(scrollback_idx) = self.line_id_to_scrollback_index(line_id) {
            // Target a scrollback row; scroll_offset = scrollback.len() - idx
            // puts that row at the top of the viewport.
            let new_offset = self.scrollback.len().saturating_sub(scrollback_idx);
            self.scroll_offset = new_offset.min(self.scrollback.len());
            true
        } else if line_id >= self.total_lines_scrolled {
            // Already in the live grid; just snap to the bottom.
            self.scroll_offset = 0;
            true
        } else {
            false
        }
    }

    /// Move the viewport to the prompt mark immediately before the
    /// currently-visible top row. Returns true on a successful jump.
    pub fn jump_to_prev_command(&mut self) -> bool {
        if self.use_alt_buffer {
            return false;
        }
        self.prune_evicted_marks();

        // The "current top" line id of the viewport.
        let top_line_id = self
            .total_lines_scrolled
            .saturating_sub(self.scroll_offset as u64);

        // Find the latest mark strictly before the current top.
        let target = self
            .command_marks
            .iter()
            .rev()
            .find(|m| m.line_id < top_line_id)
            .copied();

        match target {
            Some(mark) => self.scroll_to_line_id(mark.line_id),
            None => false,
        }
    }

    /// Move the viewport to the next prompt mark after the currently-visible
    /// top row. Returns true on a successful jump.
    pub fn jump_to_next_command(&mut self) -> bool {
        if self.use_alt_buffer {
            return false;
        }
        self.prune_evicted_marks();

        let top_line_id = self
            .total_lines_scrolled
            .saturating_sub(self.scroll_offset as u64);

        let target = self
            .command_marks
            .iter()
            .find(|m| m.line_id > top_line_id)
            .copied();

        match target {
            Some(mark) => self.scroll_to_line_id(mark.line_id),
            None => {
                // No further mark; if we were scrolled up, snap to live view.
                if self.scroll_offset != 0 {
                    self.scroll_offset = 0;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn get_mouse_report(&self, button: u8, col: usize, row: usize) -> Option<Vec<u8>> {
        // Check if any mouse reporting mode is enabled
        if !self.modes.contains(&1000) && !self.modes.contains(&1002) && !self.modes.contains(&1003)
        {
            return None;
        }

        // SGR format (mode 1006) is preferred: CSI < button ; col ; row M/m
        // Standard format (mode 1000/1002): CSI M button col row (3 bytes)

        if self.modes.contains(&1006) {
            // SGR format: CSI < button ; x ; y M (button press) or m (button release)
            // For now, we'll generate press events (M) - release tracking would need more state
            // Decimal SGR coordinates are not subject to the one-byte legacy
            // protocol limit. Keep them 1-indexed without truncating at 255.
            let x = col.saturating_add(1);
            let y = row.saturating_add(1);
            Some(format!("\x1b[<{};{};{}M", button, x, y).into_bytes())
        } else {
            // Standard xterm format: CSI M button col row (raw bytes)
            // Coordinates are 1-indexed, offset by 32, and capped at 223 so
            // the encoded value fits in one byte. Clamp before narrowing to
            // u8; casting first makes coordinates >= 256 wrap around.
            let button_byte = button.saturating_add(32);
            let col_byte = 32 + col.saturating_add(1).min(223) as u8;
            let row_byte = 32 + row.saturating_add(1).min(223) as u8;
            Some(vec![b'\x1b', b'[', b'M', button_byte, col_byte, row_byte])
        }
    }

    pub fn get_mouse_release_report(&self, button: u8, col: usize, row: usize) -> Option<Vec<u8>> {
        if !self.modes.contains(&1000) && !self.modes.contains(&1002) && !self.modes.contains(&1003)
        {
            return None;
        }

        if self.modes.contains(&1006) {
            // SGR format: lowercase 'm' for release
            let x = col.saturating_add(1);
            let y = row.saturating_add(1);
            Some(format!("\x1b[<{};{};{}m", button, x, y).into_bytes())
        } else {
            // Standard xterm: release is button 3
            let button_byte = 32 + 3u8;
            let col_byte = 32 + col.saturating_add(1).min(223) as u8;
            let row_byte = 32 + row.saturating_add(1).min(223) as u8;
            Some(vec![b'\x1b', b'[', b'M', button_byte, col_byte, row_byte])
        }
    }

    pub fn is_mouse_enabled(&self) -> bool {
        self.modes.contains(&1000) || self.modes.contains(&1002) || self.modes.contains(&1003)
    }

    /// 1002 reports motion only while a button is held; 1003 reports all
    /// pointer motion. Mode 1000 is press/release only.
    pub fn should_report_mouse_motion(&self, button_down: bool) -> bool {
        self.modes.contains(&1003) || (button_down && self.modes.contains(&1002))
    }

    pub fn is_alt_buffer_active(&self) -> bool {
        self.use_alt_buffer
    }

    pub fn is_bracketed_paste_enabled(&self) -> bool {
        self.modes.contains(&2004)
    }

    pub fn is_application_cursor_keys(&self) -> bool {
        self.modes.contains(&1)
    }

    pub fn is_paste_events_enabled(&self) -> bool {
        self.modes.contains(&5522)
    }

    pub fn keyboard_enhancement_flags(&self) -> u16 {
        self.keyboard_enhancement_flags
    }

    pub fn xterm_modify_other_keys(&self) -> u16 {
        self.xterm_modify_other_keys
    }

    pub fn xterm_format_other_keys(&self) -> u16 {
        self.xterm_format_other_keys
    }

    pub fn is_report_all_keys_enabled(&self) -> bool {
        self.modes.contains(&2031) || (self.keyboard_enhancement_flags & 0b1000) != 0
    }

    fn sanitized_osc_5522_mimes(mime_types: &[String]) -> Vec<String> {
        let mut seen = HashSet::new();
        mime_types
            .iter()
            .filter(|mime| Self::is_valid_osc_5522_mime(mime))
            .filter(|mime| seen.insert((*mime).clone()))
            .take(MAX_OSC_5522_MIME_TYPES)
            .cloned()
            .collect()
    }

    fn build_osc_5522_mime_list(mime_types: &[String], password: Option<&str>) -> Vec<u8> {
        let mut output = Vec::new();

        output.extend_from_slice(b"\x1b]5522;type=read:status=OK");
        if let Some(password) = password {
            let encoded_password =
                base64::engine::general_purpose::STANDARD.encode(password.as_bytes());
            output.extend_from_slice(b":pw=");
            output.extend_from_slice(encoded_password.as_bytes());
        }
        output.extend_from_slice(Self::osc_terminator());

        for mime_type in mime_types {
            let encoded_mime =
                base64::engine::general_purpose::STANDARD.encode(mime_type.as_bytes());
            output.extend_from_slice(b"\x1b]5522;type=read:status=DATA:mime=");
            output.extend_from_slice(encoded_mime.as_bytes());
            output.extend_from_slice(Self::osc_terminator());
        }

        output.extend_from_slice(b"\x1b]5522;type=read:status=DONE\x1b\\");
        output
    }

    /// Build the unsolicited MIME list sent only after a real user paste
    /// action. The returned password grants one short-lived read of one of the
    /// MIME types in this exact list.
    pub fn build_paste_event(&mut self, mime_types: &[String]) -> Vec<u8> {
        if !self.is_paste_events_enabled() {
            self.pending_paste_grant = None;
            return Vec::new();
        }

        let mime_types = Self::sanitized_osc_5522_mimes(mime_types);
        let token = uuid::Uuid::new_v4().to_string();
        self.pending_paste_grant = Some(PendingPasteGrant {
            token: token.clone(),
            offered_mimes: mime_types.iter().cloned().collect(),
            expires_at: std::time::Instant::now() + OSC_5522_PASTE_GRANT_TTL,
        });
        Self::build_osc_5522_mime_list(&mime_types, Some(&token))
    }

    pub fn take_clipboard_read_requests(&mut self) -> Vec<ClipboardReadRequest> {
        std::mem::take(&mut self.pending_clipboard_requests)
    }

    /// Plan the complete identity document before any viewport slicing.
    ///
    /// Scrollback contributes cached geometry only: this path neither decodes
    /// compressed cell records nor scans their bytes. The live grid is already
    /// at display width, so its rows are appended as hard, full-width
    /// boundaries without inspecting their cells.
    #[allow(dead_code)] // Dormant P1 planner; viewport/collapse wiring lands later.
    pub(super) fn identity_projection_plan(&self, cols: usize) -> ProjectionPlan {
        let cols = cols.max(1);
        let source_base = usize::try_from(
            self.total_lines_scrolled
                .saturating_sub(self.scrollback.len() as u64),
        )
        .unwrap_or(usize::MAX.saturating_sub(self.scrollback.len()));
        let history_layouts = self
            .scrollback
            .iter()
            .enumerate()
            .map(|(absolute_row, line)| {
                #[cfg(test)]
                PROJECTION_PLAN_HISTORY_LAYOUT_VISITS.with(|visits| {
                    visits.set(visits.get().saturating_add(1));
                });
                RawRowLayout::new(
                    source_base.saturating_add(absolute_row),
                    line.raw_row_id(),
                    line.reflow_content_len(),
                    line.reflow_wide_continuations()
                        .iter()
                        .copied()
                        .map(usize::from),
                    line.is_wrapped,
                )
            });
        let history_len = self.scrollback.len();
        let grid_layouts = (0..self.grid.rows()).map(|grid_row| {
            RawRowLayout::new(
                source_base
                    .saturating_add(history_len)
                    .saturating_add(grid_row),
                self.grid.row_id(grid_row),
                cols,
                std::iter::empty(),
                self.grid.row_wrapped[grid_row],
            )
        });

        ProjectionPlan::identity(history_layouts, grid_layouts, cols)
    }

    /// Resolve requested collapse ids to exact retained raw coordinates before
    /// building a full-document plan. Invalid, stale and overlapping ranges
    /// fail closed; two disjoint ranges on the same physical row remain valid.
    fn resolved_collapses(&self, policy: &ProjectionPolicy) -> Vec<ResolvedCollapse> {
        let source_base = usize::try_from(
            self.total_lines_scrolled
                .saturating_sub(self.scrollback.len() as u64),
        )
        .unwrap_or(usize::MAX.saturating_sub(self.scrollback.len()));
        let mut candidates = Vec::new();
        for zone_id in policy.collapsed_zone_ids() {
            let Some(range) = self.finished_output_range(zone_id) else {
                continue;
            };
            let Some(provenance) = self.finished_output_provenance.get(&zone_id) else {
                continue;
            };
            let preferred = BufferAnchor {
                line_id: provenance.start_line_id,
                column: range.start.col,
            };
            let Some((start_absolute, _)) =
                self.retained_raw_row_absolute(range.start.row, Some(preferred))
            else {
                continue;
            };
            let Some(end_absolute) = start_absolute.checked_add(provenance.rows.len() - 1) else {
                continue;
            };
            if self
                .retained_raw_row(end_absolute)
                .is_none_or(|(row, _)| row != range.end.row)
            {
                continue;
            }
            candidates.push(ResolvedCollapse {
                range,
                start_absolute: source_base.saturating_add(start_absolute),
                end_absolute: source_base.saturating_add(end_absolute),
            });
        }
        candidates.sort_unstable_by_key(|collapse| {
            (
                collapse.start_absolute,
                collapse.range.start.col,
                collapse.end_absolute,
                collapse.range.end.col,
                collapse.range.zone_id,
            )
        });

        // Reject an entire connected overlap component. Silently choosing one
        // owner would make a stale policy retarget neighbouring block output.
        let mut resolved = Vec::with_capacity(candidates.len());
        let mut index = 0usize;
        while index < candidates.len() {
            let component_start = index;
            let mut component_end = (
                candidates[index].end_absolute,
                candidates[index].range.end.col,
            );
            index += 1;
            while let Some(candidate) = candidates.get(index) {
                let start = (candidate.start_absolute, candidate.range.start.col);
                if start >= component_end {
                    break;
                }
                component_end =
                    component_end.max((candidate.end_absolute, candidate.range.end.col));
                index += 1;
            }
            if index == component_start + 1 {
                resolved.push(candidates[component_start]);
            }
        }
        resolved
    }

    fn projection_plan_cache_key(
        &self,
        cols: usize,
        policy: &ProjectionPolicy,
    ) -> ProjectionPlanCacheKey {
        ProjectionPlanCacheKey {
            total_lines_scrolled: self.total_lines_scrolled,
            row_identity_revision: self.row_identity_revision,
            finished_output_revision: self.finished_output_revision,
            scrollback_len: self.scrollback.len(),
            rows: self.grid.rows(),
            cols,
            row_wrapped: self.grid.row_wrapped.iter().copied().collect(),
            policy_revision: policy.revision(),
            policy_ids: policy.ids(),
            full_screen_scroll_revision: self.full_screen_scroll_revision,
        }
    }

    fn projection_source_base(&self) -> Option<usize> {
        usize::try_from(
            self.total_lines_scrolled
                .checked_sub(self.scrollback.len() as u64)?,
        )
        .ok()
    }

    fn projection_source_to_absolute(&self, source: usize) -> Option<usize> {
        let absolute = source.checked_sub(self.projection_source_base()?)?;
        (absolute < self.scrollback.len().saturating_add(self.grid.rows())).then_some(absolute)
    }

    /// Return an exact cached full-document plan. A stale policy is resolved
    /// first and exits here without visiting historical layout metadata.
    fn cached_collapsed_projection_plan(
        &mut self,
        cols: usize,
        policy: &ProjectionPolicy,
    ) -> Option<std::sync::Arc<ProjectionPlan>> {
        let key = self.projection_plan_cache_key(cols, policy);
        if let Some((cached_key, plan)) = &self.projection_plan_cache {
            if self.finished_output_revision != 0 && *cached_key == key {
                return (!plan.effective_collapsed.is_empty()).then(|| std::sync::Arc::clone(plan));
            }
        }

        if let Some((cached_key, mut cached_plan)) = self.projection_plan_cache.take() {
            if let Some(plan) = std::sync::Arc::get_mut(&mut cached_plan) {
                if self.try_advance_collapsed_projection_plan(&cached_key, &key, plan) {
                    if plan.effective_collapsed.is_empty() {
                        self.projection_plan_cache = Some((key, cached_plan));
                        return None;
                    }
                    let plan_revision = self.next_projection_plan_revision;
                    if plan_revision != 0 {
                        let incremental_appended_rows = usize::try_from(
                            key.total_lines_scrolled
                                .saturating_sub(cached_key.total_lines_scrolled),
                        )
                        .unwrap_or(usize::MAX);
                        plan.incremental_from = Some(cached_key);
                        plan.incremental_appended_rows = incremental_appended_rows;
                        plan.plan_revision = plan_revision;
                        self.next_projection_plan_revision =
                            plan_revision.checked_add(1).unwrap_or(0);
                        self.projection_plan_cache =
                            Some((key, std::sync::Arc::clone(&cached_plan)));
                        return Some(cached_plan);
                    }
                }
            }
        }

        let resolved = self.resolved_collapses(policy);
        if resolved.is_empty() {
            return None;
        }
        #[cfg(test)]
        PROJECTION_PLAN_BUILD_COUNT.with(|count| count.set(count.get().saturating_add(1)));
        let mut plan = self
            .identity_projection_plan(cols)
            .splice_collapses(&resolved, policy.revision());
        if plan.effective_collapsed.is_empty() {
            self.projection_plan_cache = Some((key, std::sync::Arc::new(plan)));
            return None;
        }
        let plan_revision = self.next_projection_plan_revision;
        if plan_revision == 0 {
            return None;
        }
        plan.plan_revision = plan_revision;
        self.next_projection_plan_revision = plan_revision.checked_add(1).unwrap_or(0);
        let plan = std::sync::Arc::new(plan);
        self.projection_plan_cache = Some((key, std::sync::Arc::clone(&plan)));
        Some(plan)
    }

    fn try_advance_collapsed_projection_plan(
        &self,
        old_key: &ProjectionPlanCacheKey,
        new_key: &ProjectionPlanCacheKey,
        plan: &mut ProjectionPlan,
    ) -> bool {
        let Some(appended) = new_key
            .total_lines_scrolled
            .checked_sub(old_key.total_lines_scrolled)
            .and_then(|rows| usize::try_from(rows).ok())
            .filter(|rows| *rows > 0)
        else {
            return false;
        };
        let Some(old_source_base) = old_key
            .total_lines_scrolled
            .checked_sub(old_key.scrollback_len as u64)
            .and_then(|base| usize::try_from(base).ok())
        else {
            return false;
        };
        if old_key.finished_output_revision == 0
            || old_key.full_screen_scroll_revision == 0
            || new_key.full_screen_scroll_revision == 0
            || new_key
                .full_screen_scroll_revision
                .checked_sub(old_key.full_screen_scroll_revision)
                != Some(appended as u64)
            || new_key
                .row_identity_revision
                .checked_sub(old_key.row_identity_revision)
                != Some((appended as u64).saturating_mul(2))
            || old_key.finished_output_revision != new_key.finished_output_revision
            || old_key.policy_revision != new_key.policy_revision
            || old_key.policy_ids != new_key.policy_ids
            || old_key.cols != new_key.cols
            || old_key.rows != new_key.rows
            || plan.history_rows != old_key.scrollback_len
            || plan.raw_absolute_base != old_source_base
            || old_key.row_wrapped.iter().any(|wrapped| *wrapped)
            || new_key.row_wrapped.iter().any(|wrapped| *wrapped)
            || old_key.scrollback_len > 0
                && self
                    .scrollback
                    .get(new_key.scrollback_len.saturating_sub(appended + 1))
                    .is_some_and(|line| line.is_wrapped)
        {
            return false;
        }
        let Some(expected_history) = old_key.scrollback_len.checked_add(appended) else {
            return false;
        };
        let evicted = expected_history.saturating_sub(new_key.scrollback_len);
        if expected_history < new_key.scrollback_len || evicted > old_key.scrollback_len {
            return false;
        }
        let evicted_old_history = evicted.min(old_key.scrollback_len);
        if !plan.front_rows_are_independently_evictable(evicted_old_history) {
            return false;
        }
        let Some(old_grid_start) = plan.raw_rows.get(old_key.scrollback_len) else {
            return false;
        };
        if plan
            .resolved_collapses
            .iter()
            .any(|collapse| collapse.end_absolute >= old_grid_start.absolute_row)
        {
            return false;
        }
        if new_key.scrollback_len > 0 {
            let retained_new_history = appended.min(new_key.scrollback_len);
            if new_key.scrollback_len > retained_new_history {
                let Some(expected_front) = plan
                    .raw_rows
                    .get(evicted_old_history)
                    .map(|row| row.raw_row)
                else {
                    return false;
                };
                if self.scrollback.front().map(ScrollbackLine::raw_row_id) != Some(expected_front) {
                    return false;
                }
            }
        }

        let source_base = match self.projection_source_base() {
            Some(base) => base,
            None => return false,
        };
        let retained_new_history = appended.min(new_key.scrollback_len);
        let appended_start = new_key.scrollback_len.saturating_sub(retained_new_history);
        let appended_layouts = (appended_start..new_key.scrollback_len).map(|absolute| {
            let line = &self.scrollback[absolute];
            #[cfg(test)]
            PROJECTION_PLAN_HISTORY_LAYOUT_VISITS.with(|visits| {
                visits.set(visits.get().saturating_add(1));
            });
            RawRowLayout::new(
                source_base.saturating_add(absolute),
                line.raw_row_id(),
                line.reflow_content_len(),
                line.reflow_wide_continuations()
                    .iter()
                    .copied()
                    .map(usize::from),
                line.is_wrapped,
            )
        });
        let grid_layouts = (0..self.grid.rows()).map(|row| {
            RawRowLayout::new(
                source_base
                    .saturating_add(self.scrollback.len())
                    .saturating_add(row),
                self.grid.row_id(row),
                self.grid.row_len(),
                std::iter::empty(),
                self.grid.row_wrapped[row],
            )
        });
        plan.advance_full_screen_scroll(evicted_old_history, appended_layouts, grid_layouts)
    }

    fn bump_selection_revision(&mut self) {
        self.selection_revision = self.selection_revision.checked_add(1).unwrap_or(1);
    }

    /// Raw and transformed document coordinates are intentionally exclusive.
    /// A transformed selection is re-anchored through retained raw identities
    /// when a compatible plan replaces the one it was created against.
    fn enter_transformed_selection_space(&mut self, plan: &ProjectionPlan) {
        let raw_changed = self.selection.take().is_some();
        let previous = self.projected_selection.clone();
        if plan.plan_revision == 0 {
            self.projected_selection = None;
        } else if self
            .projected_selection
            .as_ref()
            .is_some_and(|selection| selection.plan_revision != plan.plan_revision)
        {
            self.reanchor_projected_selection(plan);
        }
        let projected_changed = self.projected_selection != previous;
        if raw_changed || projected_changed {
            self.bump_selection_revision();
        }
    }

    /// Carry a normal transformed selection across a safe plan rebuild.
    /// Width changes, column selections, hidden-set changes and lost/ambiguous
    /// endpoint identities all fail closed to the previous clear behavior.
    fn reanchor_projected_selection(&mut self, plan: &ProjectionPlan) {
        let Some(selection) = self.projected_selection.clone() else {
            return;
        };
        if selection.mode == SelectionMode::Block
            || selection.plan_cols != plan.cols
            || selection.hidden != plan.effective_collapsed
        {
            self.projected_selection = None;
            return;
        }
        let (Some(anchor), Some(active)) = (
            plan.selection_point_for_anchor(selection.anchor.anchor),
            plan.selection_point_for_anchor(selection.active.anchor),
        ) else {
            self.projected_selection = None;
            return;
        };
        self.projected_selection = Some(ProjectedSelection {
            plan_revision: plan.plan_revision,
            anchor: ProjectedSelectionEndpoint {
                point: anchor,
                ..selection.anchor
            },
            active: ProjectedSelectionEndpoint {
                point: active,
                ..selection.active
            },
            ..selection
        });
    }

    fn new_projected_selection(
        viewport: &ProjectedViewport,
        plan_revision: u64,
        anchor: ProjectedSelectionEndpoint,
        active: ProjectedSelectionEndpoint,
        mode: SelectionMode,
    ) -> ProjectedSelection {
        ProjectedSelection {
            plan_revision,
            plan_cols: viewport.columns(),
            hidden: viewport.effective_collapsed().clone(),
            anchor,
            active,
            mode,
        }
    }

    fn leave_transformed_selection_space(&mut self) {
        if self.projected_selection.take().is_some() {
            self.bump_selection_revision();
        }
    }

    fn materialize_projection_plan(
        &self,
        plan: &ProjectionPlan,
        document_start: usize,
        viewport_rows: usize,
    ) -> MaterializedProjection {
        let document_start = document_start.min(plan.document_rows());
        let visible_rows = plan
            .document_rows()
            .saturating_sub(document_start)
            .min(viewport_rows);
        let top_padding = viewport_rows.saturating_sub(visible_rows);
        let mut cells = Vec::with_capacity(viewport_rows);
        let mut row_wrapped = Vec::with_capacity(viewport_rows);
        let mut row_kinds = Vec::with_capacity(viewport_rows);
        let mut row_sources = Vec::with_capacity(viewport_rows);
        let mut origins = Vec::new();
        for _ in 0..top_padding {
            cells.push(vec![TerminalCell::default(); plan.cols]);
            row_wrapped.push(false);
            row_kinds.push(ProjectedRowKind::Padding);
            row_sources.push(None);
        }

        let mut history_cache: HashMap<usize, Vec<TerminalCell>> = HashMap::new();
        for planned_row in plan.rows.iter().skip(document_start).take(visible_rows) {
            let display_row = cells.len();
            let mut line = vec![TerminalCell::default(); plan.cols];
            for slice in &planned_row.raw_slices {
                let Some(source_absolute) =
                    self.projection_source_to_absolute(slice.source.absolute_row)
                else {
                    continue;
                };
                let source = if source_absolute < self.scrollback.len() {
                    history_cache
                        .entry(source_absolute)
                        .or_insert_with(|| {
                            #[cfg(test)]
                            PROJECTION_VIEW_HISTORY_DECOMPRESSES.with(|count| {
                                count.set(count.get().saturating_add(1));
                            });
                            self.scrollback[source_absolute].decompress()
                        })
                        .as_slice()
                } else {
                    let Some(grid_row) = source_absolute
                        .checked_sub(self.scrollback.len())
                        .filter(|row| *row < self.grid.rows())
                    else {
                        continue;
                    };
                    &self.grid[grid_row]
                };
                let Some(source_end) = slice.source.col_start.checked_add(slice.len) else {
                    continue;
                };
                let Some(view_end) = slice.view_col_start.checked_add(slice.len) else {
                    continue;
                };
                if source_end > source.len() || view_end > line.len() {
                    continue;
                }
                line[slice.view_col_start..view_end]
                    .copy_from_slice(&source[slice.source.col_start..source_end]);
                if slice.narrow_wide_body {
                    line[slice.view_col_start].flags.set_wide(false);
                }
                if let Some(origin) = slice.origin {
                    origins.push(OriginSpan {
                        display_start: DisplayPoint::new(display_row, slice.view_col_start),
                        raw_start: RawCellAnchor {
                            row_id: origin.row,
                            column: origin.col_start,
                        },
                        len: slice.len,
                    });
                }
            }
            cells.push(line);
            row_wrapped.push(planned_row.wrapped);
            row_kinds.push(planned_row.kind);
            row_sources.push(planned_row.row_source.and_then(|source| {
                Some(super::projection::RawRowSource {
                    raw_row: source.raw_row,
                    raw_absolute_row: self
                        .projection_source_to_absolute(source.raw_absolute_row)?,
                })
            }));
        }

        MaterializedProjection {
            cells,
            row_wrapped,
            row_kinds,
            row_sources,
            origins,
            document_start,
            top_padding,
        }
    }

    fn projected_top_anchor(&self, viewport: &ProjectedViewport) -> Option<ProjectedTopAnchor> {
        viewport.stable_top_anchor().or_else(|| {
            let view_row = viewport.top_padding();
            let absolute = viewport
                .view_row_absolute(view_row)
                .unwrap_or_else(|| viewport.legacy_absolute_row(view_row));
            let (row, _) = self.retained_raw_row(absolute)?;
            row.is_tracked().then_some(ProjectedTopAnchor::RawRow(row))
        })
    }

    fn retained_absolute_for_raw(&self, row: RawRowId) -> Option<usize> {
        if !row.is_tracked() {
            return None;
        }
        self.scrollback
            .iter()
            .position(|line| line.raw_row_id() == row)
            .or_else(|| {
                self.grid
                    .row_ids
                    .iter()
                    .position(|candidate| *candidate == row)
                    .map(|grid_row| self.scrollback.len().saturating_add(grid_row))
            })
    }

    fn restore_identity_scroll_from_projection(&mut self, state: &ProjectionViewState) {
        if state.follow_bottom {
            self.scroll_offset = 0;
            return;
        }
        let row = match state.top_anchor {
            Some(ProjectedTopAnchor::RawCell(anchor)) => Some(anchor.row_id),
            Some(ProjectedTopAnchor::RawRow(row)) => Some(row),
            Some(ProjectedTopAnchor::Summary { hidden_range, .. }) => Some(hidden_range.start.row),
            None => None,
        };
        let Some(absolute) = row.and_then(|row| self.retained_absolute_for_raw(row)) else {
            self.scroll_offset = self.scroll_offset.min(self.scrollback.len());
            return;
        };
        let desired_start = absolute.min(self.scrollback.len());
        self.scroll_offset = self.scrollback.len().saturating_sub(desired_start);
    }

    /// Materialize a primary-screen block document after applying the
    /// session-owned policy. Identity, bypass and ineffective policies return
    /// the exact P0 viewport allocation. Rebuilds preserve a stable top raw or
    /// synthetic anchor, while offset zero continues following the bottom.
    pub fn projected_viewport_with_state(
        &mut self,
        projection: HistoryProjection,
        block_mode: bool,
        policy: &ProjectionPolicy,
        view_state: &mut ProjectionViewState,
    ) -> ProjectedViewport {
        if !block_mode || self.use_alt_buffer {
            self.leave_transformed_selection_space();
            return self.projected_viewport(projection, block_mode);
        }
        if policy.is_identity() {
            self.leave_transformed_selection_space();
            if view_state.last_plan_key.is_some() {
                self.restore_identity_scroll_from_projection(view_state);
                view_state.last_plan_key = None;
            }
            return self.projected_viewport(projection, true);
        }

        if view_state.last_plan_key.is_none() && self.scroll_offset > 0 {
            let identity = self.projected_viewport(projection, true);
            view_state.top_anchor = self.projected_top_anchor(&identity);
            view_state.follow_bottom = false;
        }

        let cols = self.grid.row_len().max(1);
        let plan_key = self.projection_plan_cache_key(cols, policy);
        let Some(plan) = self.cached_collapsed_projection_plan(cols, policy) else {
            self.leave_transformed_selection_space();
            if view_state.last_plan_key.is_some() {
                self.restore_identity_scroll_from_projection(view_state);
                view_state.last_plan_key = None;
            }
            return self.projected_viewport(projection, true);
        };
        self.enter_transformed_selection_space(&plan);
        let viewport_rows = self.grid.rows();
        let max_offset = plan.document_rows().saturating_sub(viewport_rows);
        if view_state.last_plan_key.as_ref() != Some(&plan_key) {
            if view_state.follow_bottom {
                view_state.offset_from_bottom = 0;
            } else if plan.incremental_from.as_ref() == view_state.last_plan_key.as_ref() {
                view_state.offset_from_bottom = view_state
                    .offset_from_bottom
                    .saturating_add(plan.incremental_appended_rows)
                    .min(max_offset);
            } else if let Some(target) = view_state
                .top_anchor
                .and_then(|anchor| plan.document_row_for_anchor(anchor))
            {
                view_state.offset_from_bottom = max_offset.saturating_sub(target.min(max_offset));
            } else {
                view_state.offset_from_bottom = view_state.offset_from_bottom.min(max_offset);
            }
        }
        let document_start = max_offset.saturating_sub(view_state.offset_from_bottom);
        let cursor_row_id = self.grid.row_id(self.cursor_row.min(self.grid.rows() - 1));
        let viewport_key = TransformedViewportCacheKey {
            plan: plan_key.clone(),
            projection_revision: plan.plan_revision,
            grid_version: self.grid_version,
            document_start,
            viewport_rows,
            cursor_row: cursor_row_id,
            cursor_col: self.cursor_col,
        };
        let viewport = if let Some((cached_key, viewport)) = &self.transformed_viewport_cache {
            if *cached_key == viewport_key {
                viewport.clone()
            } else {
                let materialized =
                    self.materialize_projection_plan(&plan, document_start, viewport_rows);
                let cursor = materialized
                    .origins
                    .iter()
                    .find_map(|span| {
                        let end = span.raw_start.column.saturating_add(span.len);
                        (span.raw_start.row_id == cursor_row_id
                            && (span.raw_start.column..end).contains(&self.cursor_col))
                        .then(|| {
                            DisplayPoint::new(
                                span.display_start.row,
                                span.display_start.column + self.cursor_col - span.raw_start.column,
                            )
                        })
                    })
                    .unwrap_or_else(|| DisplayPoint::new(usize::MAX, usize::MAX));
                let key = ProjectionCacheKey::new(
                    self.grid_version,
                    plan.plan_revision,
                    self.total_lines_scrolled,
                    self.row_identity_revision,
                    self.scrollback.len(),
                    view_state.offset_from_bottom,
                    viewport_rows,
                    cols,
                    false,
                    ProjectionMode::Transformed,
                );
                ProjectedViewport::new_transformed(
                    key,
                    materialized.cells,
                    materialized.row_wrapped,
                    materialized.row_kinds,
                    materialized.row_sources,
                    materialized.origins,
                    cursor,
                    plan.document_rows(),
                    materialized.document_start,
                    materialized.top_padding,
                    plan.effective_collapsed.clone(),
                )
            }
        } else {
            let materialized =
                self.materialize_projection_plan(&plan, document_start, viewport_rows);
            let cursor = materialized
                .origins
                .iter()
                .find_map(|span| {
                    let end = span.raw_start.column.saturating_add(span.len);
                    (span.raw_start.row_id == cursor_row_id
                        && (span.raw_start.column..end).contains(&self.cursor_col))
                    .then(|| {
                        DisplayPoint::new(
                            span.display_start.row,
                            span.display_start.column + self.cursor_col - span.raw_start.column,
                        )
                    })
                })
                .unwrap_or_else(|| DisplayPoint::new(usize::MAX, usize::MAX));
            let key = ProjectionCacheKey::new(
                self.grid_version,
                plan.plan_revision,
                self.total_lines_scrolled,
                self.row_identity_revision,
                self.scrollback.len(),
                view_state.offset_from_bottom,
                viewport_rows,
                cols,
                false,
                ProjectionMode::Transformed,
            );
            ProjectedViewport::new_transformed(
                key,
                materialized.cells,
                materialized.row_wrapped,
                materialized.row_kinds,
                materialized.row_sources,
                materialized.origins,
                cursor,
                plan.document_rows(),
                materialized.document_start,
                materialized.top_padding,
                plan.effective_collapsed.clone(),
            )
        };
        if self
            .transformed_viewport_cache
            .as_ref()
            .is_none_or(|(cached_key, _)| *cached_key != viewport_key)
        {
            self.transformed_viewport_cache = Some((viewport_key, viewport.clone()));
        }
        view_state.offset_from_bottom = viewport.scroll_offset();
        view_state.follow_bottom = view_state.offset_from_bottom == 0;
        view_state.top_anchor = (!view_state.follow_bottom)
            .then(|| self.projected_top_anchor(&viewport))
            .flatten();
        view_state.last_plan_key = Some(plan_key);
        viewport
    }

    /// Test oracle for the future viewport materializer. It intentionally
    /// reads only `RawSlice::source`: an untracked row has no stable origin,
    /// but its terminal bytes must still survive identity projection.
    #[cfg(test)]
    pub(super) fn materialize_identity_projection_plan(
        &self,
        plan: &ProjectionPlan,
    ) -> (Vec<Vec<TerminalCell>>, Vec<bool>) {
        let mut history_cache: HashMap<usize, Vec<TerminalCell>> = HashMap::new();
        let rows = plan
            .rows
            .iter()
            .map(|planned_row| {
                let mut cells = vec![TerminalCell::default(); plan.cols];
                for slice in &planned_row.raw_slices {
                    let source_absolute = self
                        .projection_source_to_absolute(slice.source.absolute_row)
                        .expect("planned source should remain retained");
                    let source = if source_absolute < self.scrollback.len() {
                        history_cache
                            .entry(source_absolute)
                            .or_insert_with(|| {
                                PROJECTION_PLAN_ORACLE_HISTORY_DECOMPRESSES.with(|count| {
                                    count.set(count.get().saturating_add(1));
                                });
                                self.scrollback[source_absolute].decompress()
                            })
                            .as_slice()
                    } else {
                        let grid_row = source_absolute - self.scrollback.len();
                        &self.grid[grid_row]
                    };
                    let source_end = slice
                        .source
                        .col_start
                        .checked_add(slice.len)
                        .expect("planned source slice overflow");
                    let view_end = slice
                        .view_col_start
                        .checked_add(slice.len)
                        .expect("planned view slice overflow");
                    cells[slice.view_col_start..view_end]
                        .copy_from_slice(&source[slice.source.col_start..source_end]);
                    if slice.narrow_wide_body {
                        cells[slice.view_col_start].flags.set_wide(false);
                    }
                }
                cells
            })
            .collect();
        let wrapped = plan.rows.iter().map(|row| row.wrapped).collect();
        (rows, wrapped)
    }

    /// Return the next joined logical-line boundary and the number of visual
    /// rows that logical line occupies at `new_cols`.
    ///
    /// `ScrollbackLine` caches the number of cells retained by the historical
    /// trailing-blank rule, so this counting pass performs no decompression.
    fn reflow_span(
        lines: &VecDeque<ScrollbackLine>,
        start: usize,
        end: usize,
        new_cols: usize,
    ) -> (usize, usize) {
        debug_assert!(start < end);
        debug_assert!(new_cols > 0);

        let mut next = start;
        let mut logical_cells = 0usize;
        loop {
            let line = &lines[next];
            logical_cells = logical_cells.saturating_add(line.reflow_content_len());
            next += 1;
            if !line.is_wrapped || next >= end {
                break;
            }
        }

        let visual_rows = if logical_cells == 0 {
            1
        } else {
            logical_cells.div_ceil(new_cols)
        };
        (next, visual_rows)
    }

    /// Lazily materialize only the historical rows that can enter the current
    /// viewport.  The old path cloned every compressed line in the scrollback
    /// tail, decoded all of them, recompressed all reflowed rows, and finally
    /// decoded the visible handful again.
    ///
    /// This implementation first counts visual rows from cached per-line
    /// lengths, then decodes only logical lines intersecting the requested
    /// range.  Recompressing the selected rows keeps byte-for-byte historical
    /// cell semantics (including the existing style normalization) while
    /// bounding that work to `viewport_rows`.
    fn reflowed_viewport_rows(
        lines: &VecDeque<ScrollbackLine>,
        start: usize,
        end: usize,
        new_cols: usize,
        scroll_offset: usize,
        viewport_rows: usize,
        blank_cell: &TerminalCell,
    ) -> Vec<Vec<TerminalCell>> {
        if start >= end || viewport_rows == 0 {
            return Vec::new();
        }

        let mut total_visual_rows = 0usize;
        let mut source = start;
        while source < end {
            let (next, visual_rows) = Self::reflow_span(lines, source, end, new_cols);
            total_visual_rows = total_visual_rows.saturating_add(visual_rows);
            source = next;
        }

        // This is the same range selected by the former `skip` /
        // `visible_start` calculation: begin `scroll_offset` visual rows from
        // the tail, then retain at most one terminal viewport.
        let target_start = total_visual_rows.saturating_sub(scroll_offset);
        let target_end = target_start
            .saturating_add(viewport_rows)
            .min(total_visual_rows);
        let mut result = Vec::with_capacity(target_end.saturating_sub(target_start));

        source = start;
        let mut visual_start = 0usize;
        while source < end && visual_start < target_end {
            let (next, visual_rows) = Self::reflow_span(lines, source, end, new_cols);
            let visual_end = visual_start.saturating_add(visual_rows);

            if visual_end > target_start {
                let mut logical_line = Vec::new();
                for line in lines.range(source..next) {
                    let decompressed = line.decompress();
                    logical_line.extend_from_slice(Self::strip_trailing_blanks(&decompressed));
                }

                let first_chunk = target_start.saturating_sub(visual_start);
                let last_chunk = target_end.saturating_sub(visual_start).min(visual_rows);
                for chunk_index in first_chunk..last_chunk {
                    let mut row = if logical_line.is_empty() {
                        vec![*blank_cell; new_cols]
                    } else {
                        let cell_start = chunk_index.saturating_mul(new_cols);
                        let cell_end = cell_start.saturating_add(new_cols).min(logical_line.len());
                        logical_line[cell_start..cell_end].to_vec()
                    };
                    row.resize(new_cols, *blank_cell);

                    // Preserve the exact cell normalization of reflow_lines()
                    // without recompressing the entire historical tail.
                    let normalized =
                        ScrollbackLine::compress(&row, chunk_index + 1 < visual_rows).decompress();
                    result.push(normalized);
                }
            }

            visual_start = visual_end;
            source = next;
        }

        result
    }

    pub fn get_visible_cells(&mut self) -> std::sync::Arc<Vec<Vec<TerminalCell>>> {
        if let Some((cached_version, cached_offset, ref cells)) = self.visible_cells_cache {
            if cached_version == self.grid_version && cached_offset == self.scroll_offset {
                return std::sync::Arc::clone(cells);
            }
        }

        // Cache miss - rebuild
        let rows = self.grid.rows();
        let cols = if rows > 0 { self.grid.row_len() } else { 80 };

        // Try to recycle the previous allocation. The renderer drops its returned
        // Arc each frame, so by the next miss we are usually the sole owner and can
        // refill the existing nested Vecs in place instead of reallocating per row.
        let prev = self.visible_cells_cache.take();
        let prev_version = prev.as_ref().map(|(v, _, _)| *v);
        let prev_offset = prev.as_ref().map(|(_, o, _)| *o);
        let mut recycled = prev.map(|(_, _, a)| a);

        if self.scroll_offset == 0 {
            // Fast path: copy current grid, reusing inner Vec capacity when possible.
            if let Some(mut arc) = recycled.take() {
                if let Some(buf) = std::sync::Arc::get_mut(&mut arc) {
                    #[cfg(test)]
                    VISIBLE_CELLS_RECYCLE_COUNT.with(|count| {
                        count.set(count.get().saturating_add(1));
                    });
                    // Incremental path: if the recycled buffer already holds a same-sized
                    // snapshot taken at scroll_offset==0, only re-copy rows whose
                    // row_versions changed since that snapshot. Untouched rows already
                    // hold valid data, turning an O(rows*cols) copy into O(dirty cells).
                    let can_incremental = prev_offset == Some(0)
                        && buf.len() == rows
                        && buf.iter().all(|r| r.len() == cols);
                    if can_incremental {
                        let base = prev_version.unwrap_or(0);
                        for (r, (dst, chunk)) in buf.iter_mut().zip(self.grid.iter()).enumerate() {
                            if self.row_versions[r] > base {
                                dst.clear();
                                dst.extend_from_slice(chunk);
                            }
                        }
                    } else {
                        buf.resize_with(rows, Vec::new);
                        for (dst, chunk) in buf.iter_mut().zip(self.grid.iter()) {
                            dst.clear();
                            dst.extend_from_slice(chunk);
                        }
                    }
                    self.visible_cells_cache = Some((
                        self.grid_version,
                        self.scroll_offset,
                        std::sync::Arc::clone(&arc),
                    ));
                    return arc;
                }
                // 仍被他处共享,无法原地复用;放回供下方 fallback 分支重建。
                recycled = Some(arc);
            }
        }

        let cells = if self.scroll_offset == 0 {
            // Fast path (shared allocation): fresh copy of current grid.
            self.grid.to_vec()
        } else {
            // Historical path: count from cached compressed-line metadata and
            // materialize only rows that can enter this viewport.
            let blank_cell = self.create_blank_cell();

            let mut start_idx = self
                .scrollback
                .len()
                .saturating_sub(self.scroll_offset + rows);
            while start_idx > 0 && self.scrollback[start_idx - 1].is_wrapped {
                start_idx -= 1;
            }
            let end_idx = self.scrollback.len();
            let mut result = Self::reflowed_viewport_rows(
                &self.scrollback,
                start_idx,
                end_idx,
                cols,
                self.scroll_offset,
                rows,
                &blank_cell,
            );

            for row in self.grid.iter() {
                if result.len() < rows {
                    result.push(self.normalize_line_width(row.to_vec(), cols));
                } else {
                    break;
                }
            }

            while result.len() < rows {
                result.push(self.blank_line(cols));
            }

            result
        };

        // Reuse the recycled Arc's outer allocation if we still solely own it.
        let arc = match recycled.take() {
            Some(mut arc) => match std::sync::Arc::get_mut(&mut arc) {
                Some(buf) => {
                    *buf = cells;
                    arc
                }
                None => std::sync::Arc::new(cells),
            },
            None => std::sync::Arc::new(cells),
        };
        self.visible_cells_cache = Some((
            self.grid_version,
            self.scroll_offset,
            std::sync::Arc::clone(&arc),
        ));
        arc
    }

    fn append_projected_origin_span(
        origins: &mut Vec<super::projection::OriginSpan>,
        display_start: DisplayPoint,
        raw_start: RawCellAnchor,
        len: usize,
    ) {
        if len == 0 || !raw_start.row_id.is_tracked() {
            return;
        }

        // Preserve one affine run whenever both coordinate spaces continue
        // contiguously. This keeps lookup metadata proportional to raw rows,
        // not viewport cells, in the common identity case.
        if let Some(previous) = origins.last_mut() {
            let display_contiguous = previous.display_start.row == display_start.row
                && previous.display_start.column.saturating_add(previous.len)
                    == display_start.column;
            let raw_contiguous = previous.raw_start.row_id == raw_start.row_id
                && previous.raw_start.column.saturating_add(previous.len) == raw_start.column;
            if display_contiguous && raw_contiguous {
                previous.len = previous.len.saturating_add(len);
                return;
            }
        }

        origins.push(super::projection::OriginSpan {
            display_start,
            raw_start,
            len,
        });
    }

    /// Build stable raw-cell provenance for the exact rows materialized by
    /// `get_visible_cells`. This mirrors the existing lazy reflow window but
    /// never changes it: retained content gets an affine origin span and the
    /// blank cells introduced only to pad a projected row remain unmapped.
    ///
    /// Returns cell origins AND per-display-row provenance. `row_sources` is
    /// reflow-aware: it names the raw row that OWNS each display row, which is
    /// what `ProjectedViewport::view_row_absolute` — and therefore block
    /// chrome — needs. The legacy
    /// `scrollback.len() - scroll_offset + display_row` arithmetic is only
    /// correct while every retained line is full width and unwrapped, so
    /// deriving it here is what lets command cards survive scrolling back over
    /// ordinary soft-wrapped output.
    fn identity_projection_geometry(
        &self,
        viewport_rows: usize,
        cols: usize,
    ) -> (
        Vec<super::projection::OriginSpan>,
        Vec<Option<super::projection::RawRowSource>>,
    ) {
        if viewport_rows == 0 || cols == 0 {
            return (Vec::new(), vec![None; viewport_rows]);
        }

        let mut origins = Vec::with_capacity(viewport_rows);
        let mut row_sources: Vec<Option<super::projection::RawRowSource>> =
            vec![None; viewport_rows];
        let grid_source = |grid_row: usize| super::projection::RawRowSource {
            raw_row: self.grid.row_id(grid_row),
            raw_absolute_row: self.scrollback.len().saturating_add(grid_row),
        };
        if self.scroll_offset == 0 {
            for (row, slot) in row_sources
                .iter_mut()
                .enumerate()
                .take(viewport_rows.min(self.grid.rows()))
            {
                Self::append_projected_origin_span(
                    &mut origins,
                    DisplayPoint::new(row, 0),
                    RawCellAnchor {
                        row_id: self.grid.row_id(row),
                        column: 0,
                    },
                    cols.min(self.grid[row].len()),
                );
                *slot = Some(grid_source(row));
            }
            return (origins, row_sources);
        }

        let start_idx = self
            .scrollback
            .len()
            .saturating_sub(self.scroll_offset.saturating_add(viewport_rows));
        let mut start_idx = start_idx;
        while start_idx > 0 && self.scrollback[start_idx - 1].is_wrapped {
            start_idx -= 1;
        }
        let end_idx = self.scrollback.len();

        let mut total_visual_rows = 0usize;
        let mut source = start_idx;
        while source < end_idx {
            let (next, visual_rows) = Self::reflow_span(&self.scrollback, source, end_idx, cols);
            total_visual_rows = total_visual_rows.saturating_add(visual_rows);
            source = next;
        }
        let target_start = total_visual_rows.saturating_sub(self.scroll_offset);
        let target_end = target_start
            .saturating_add(viewport_rows)
            .min(total_visual_rows);
        source = start_idx;
        let mut visual_start = 0usize;
        let mut display_row = 0usize;
        while source < end_idx && visual_start < target_end {
            let (next, visual_rows) = Self::reflow_span(&self.scrollback, source, end_idx, cols);
            let visual_end = visual_start.saturating_add(visual_rows);
            if visual_end > target_start {
                let first_chunk = target_start.saturating_sub(visual_start);
                let last_chunk = target_end.saturating_sub(visual_start).min(visual_rows);

                for chunk_index in first_chunk..last_chunk {
                    let chunk_start = chunk_index.saturating_mul(cols);
                    let chunk_end = chunk_start.saturating_add(cols);
                    let mut logical_start = 0usize;
                    // The raw row that owns this display row. An entirely
                    // empty logical line produces no origin span at all
                    // (`reflow_span` still gives it one visual row), so
                    // provenance cannot be derived from `origins` alone — that
                    // row still belongs to its command card. Default to the
                    // span's FIRST row and upgrade to the first row whose
                    // content actually reaches this chunk.
                    let mut owner = source;
                    let mut owner_found = false;
                    for raw_index in source..next {
                        let raw_len = self.scrollback[raw_index].reflow_content_len();
                        let raw_end = logical_start.saturating_add(raw_len);
                        if !owner_found && raw_end > chunk_start {
                            owner = raw_index;
                            owner_found = true;
                        }
                        let intersection_start = chunk_start.max(logical_start);
                        let intersection_end = chunk_end.min(raw_end);
                        if intersection_start < intersection_end {
                            Self::append_projected_origin_span(
                                &mut origins,
                                DisplayPoint::new(display_row, intersection_start - chunk_start),
                                RawCellAnchor {
                                    row_id: self.scrollback[raw_index].raw_row_id(),
                                    column: intersection_start - logical_start,
                                },
                                intersection_end - intersection_start,
                            );
                        }
                        logical_start = raw_end;
                    }
                    if let Some(slot) = row_sources.get_mut(display_row) {
                        *slot = Some(super::projection::RawRowSource {
                            raw_row: self.scrollback[owner].raw_row_id(),
                            raw_absolute_row: owner,
                        });
                    }
                    display_row = display_row.saturating_add(1);
                }
            }
            visual_start = visual_end;
            source = next;
        }

        // The legacy materializer fills any remaining viewport rows from the
        // live grid and finally with structural blank rows. Only the former
        // have a stable primary-buffer origin.
        for grid_row in 0..self.grid.rows() {
            if display_row >= viewport_rows {
                break;
            }
            Self::append_projected_origin_span(
                &mut origins,
                DisplayPoint::new(display_row, 0),
                RawCellAnchor {
                    row_id: self.grid.row_id(grid_row),
                    column: 0,
                },
                cols.min(self.grid[grid_row].len()),
            );
            if let Some(slot) = row_sources.get_mut(display_row) {
                *slot = Some(grid_source(grid_row));
            }
            display_row = display_row.saturating_add(1);
        }

        (origins, row_sources)
    }

    /// Materialize the current viewport through the versioned history
    /// projection boundary. P0 is deliberately identity-only: it shares the
    /// exact visible-cell allocation and wraps it with stable provenance and
    /// pass-through geometry used by all viewport consumers.
    pub fn projected_viewport(
        &mut self,
        projection: HistoryProjection,
        block_mode: bool,
    ) -> ProjectedViewport {
        self.leave_transformed_selection_space();
        let rows = self.grid.rows();
        let columns = self.grid.row_len();
        let mode = if !block_mode || self.use_alt_buffer {
            super::projection::ProjectionMode::Bypass
        } else {
            super::projection::ProjectionMode::Identity
        };
        let key = ProjectionCacheKey::new(
            self.grid_version,
            projection.revision(),
            self.total_lines_scrolled,
            self.row_identity_revision,
            self.scrollback.len(),
            self.scroll_offset,
            rows,
            columns,
            self.use_alt_buffer,
            mode,
        );
        if let Some(cached) = self
            .projected_viewport_cache
            .as_ref()
            .filter(|cached| cached.key() == key)
        {
            return cached.clone();
        }

        // A stale identity/bypass viewport owns one extra clone of the same
        // cells Arc as `visible_cells_cache`. Release that internal clone
        // before asking the legacy cache to rebuild, otherwise Arc::get_mut
        // can never recycle the allocation even after the renderer dropped
        // its previous-frame viewport. A caller-held viewport remains an
        // external owner and correctly keeps the rebuild copy-on-write.
        self.projected_viewport_cache = None;

        // Identity fast path: do not clone cells. `get_visible_cells` already
        // owns the legacy incremental/live and lazy historical caches.
        let cells = self.get_visible_cells();
        let row_wrapped = self.get_visible_row_wrapped();
        let (origins, row_sources) = self.identity_projection_geometry(cells.len(), columns);
        let viewport = ProjectedViewport::new(
            key,
            cells,
            row_wrapped,
            origins,
            row_sources,
            DisplayPoint::new(self.cursor_row, self.cursor_col),
        );
        self.projected_viewport_cache = Some(viewport.clone());
        viewport
    }

    pub fn get_cursor_pos(&self) -> (usize, usize) {
        (self.cursor_row, self.cursor_col)
    }

    /// How much of the shell's work this terminal can actually see.
    fn shell_phase(&self) -> click_cursor::ShellPhase {
        match self.command_records.back() {
            Some(record) if record.complete => click_cursor::ShellPhase::Unknown,
            Some(record) => match record.state {
                CommandState::Running => click_cursor::ShellPhase::Running,
                CommandState::Editing => click_cursor::ShellPhase::Editing,
                _ => click_cursor::ShellPhase::Unknown,
            },
            // A shell without OSC 133 integration. Staying `Unknown` keeps the
            // feature working under plain bash.
            None => click_cursor::ShellPhase::Unknown,
        }
    }

    /// The cells a click is allowed to travel over: the whole soft-wrapped
    /// logical line the cursor sits on, ending one past its last character.
    ///
    /// The prompt is inside this span. That is deliberate — clicking it means
    /// "go to the start of the line", and a line editor ignores the extra
    /// `Left`s once the buffer start is reached. The *end* is what has to be
    /// exact: a `Right` past the buffer end is what accepts jsh's inline
    /// suggestion.
    fn editable_span(&self) -> Option<click_cursor::InputSpan> {
        let rows = self.grid.rows();
        let cols = self.grid.cols();
        if rows == 0 || cols == 0 {
            return None;
        }

        let cursor_row = self.cursor_row.min(rows - 1);
        let mut first = cursor_row;
        while first > 0 && self.grid.row_wrapped[first - 1] {
            first -= 1;
        }
        let mut last = cursor_row;
        while last + 1 < rows && self.grid.row_wrapped[last] {
            last += 1;
        }

        let occupied = |row: usize, col: usize| {
            let cell = self.grid.get(row, col);
            // A wide character's continuation cell holds a blank but is
            // still occupied, so trailing CJK must not be trimmed away.
            cell.flags.wide_continuation() || !matches!(cell.character, ' ' | '\0' | '\u{a0}')
        };
        // One past the last occupied cell, looking only at columns before
        // `col_bound` on `row_bound` itself.
        let scan_back = |row_bound: usize, col_bound: usize| {
            let mut end = click_cursor::Cell::new(first as i64, 0);
            'scan: for row in (first..=row_bound).rev() {
                let cols_here = if row == row_bound { col_bound } else { cols };
                for col in (0..cols_here).rev() {
                    if occupied(row, col) {
                        end = click_cursor::Cell::new(row as i64, col as i64 + 1);
                        break 'scan;
                    }
                }
            }
            end
        };
        let mut end = scan_back(last, cols);

        // A right-aligned decoration — jsh and fish paint the previous
        // command's duration flush with the right edge of the input row — is
        // on the row but not in the buffer. Its shape gives it away: a
        // trailing run that reaches the right edge, parted from everything
        // before it by a wide blank gap, entirely right of the cursor. Clip
        // it, or a click in the gap overshoots the buffer end — and past-end
        // `Right`s are how jsh accepts an inline suggestion, even one that is
        // not on screen at the moment.
        if end.col + 1 >= cols as i64 && end.col > 0 {
            let end_row = end.row as usize;
            let mut run_start = end.col as usize;
            while run_start > 0 && occupied(end_row, run_start - 1) {
                run_start -= 1;
            }
            let mut gap_start = run_start;
            while gap_start > 0 && !occupied(end_row, gap_start - 1) {
                gap_start -= 1;
            }
            if run_start - gap_start >= 3
                && (end_row as i64, run_start as i64) > (cursor_row as i64, self.cursor_col as i64)
            {
                end = scan_back(end_row, gap_start);
            }
        }

        // Trailing spaces the user typed are part of the buffer even though the
        // scan above cannot tell them from padding, so never place the end
        // before where the shell has its cursor.
        let cursor = click_cursor::Cell::new(cursor_row as i64, self.cursor_col as i64);
        if (end.row, end.col) < (cursor.row, cursor.col) {
            end = cursor;
        }

        // A fish-style shell paints its inline suggestion past the cursor and
        // then parks the cursor back at the end of what was typed. Those cells
        // are a preview, not buffer — the backwards scan above cannot tell them
        // from typed text, and every `Right` spent on them is the shell
        // *accepting* the suggestion. Cut the span at the first one.
        if let Some(ghost) = self.inline_suggestion_start(cursor, end) {
            end = ghost;
        }

        Some(click_cursor::InputSpan {
            start: click_cursor::Cell::new(first as i64, 0),
            end,
        })
    }

    /// Where inline-suggestion ("ghost") text begins between `from` and `end`,
    /// if it begins at all.
    ///
    /// The scan runs forward from the cursor because that is where a suggestion
    /// starts: shells only offer one when the caret is at the end of the
    /// buffer, so the first suggestion-styled cell at or after the cursor is
    /// where the real input stops.
    fn inline_suggestion_start(
        &self,
        from: click_cursor::Cell,
        end: click_cursor::Cell,
    ) -> Option<click_cursor::Cell> {
        let rows = self.grid.rows();
        let cols = self.grid.cols();
        let mut row = from.row.max(0) as usize;
        let mut col = from.col.max(0) as usize;
        while row < rows && (row as i64, col as i64) < (end.row, end.col) {
            if col >= cols {
                row += 1;
                col = 0;
                continue;
            }
            if is_inline_suggestion_cell(self.grid.get(row, col)) {
                return Some(click_cursor::Cell::new(row as i64, col as i64));
            }
            col += 1;
        }
        None
    }

    /// Arrow-key bytes that walk the shell's line editor to a clicked cell, or
    /// nothing when this click must not move it.
    ///
    /// `click_row`/`click_col` are viewport coordinates, which only line up
    /// with the grid while the scrollback is at the bottom — the
    /// `scrolled_back` guard is what makes that assumption safe.
    pub fn click_cursor_move(&self, click_row: usize, click_col: usize, enabled: bool) -> Vec<u8> {
        let guards = click_cursor::Guards {
            enabled,
            mouse_reporting: self.is_mouse_enabled(),
            alt_screen: self.is_alt_buffer_active(),
            scrolled_back: self.scroll_offset != 0,
            phase: self.shell_phase(),
        };
        if !click_cursor::click_may_move_cursor(&guards) {
            return Vec::new();
        }

        let columns = self.grid.cols() as i64;
        let cursor = click_cursor::Cell::new(self.cursor_row as i64, self.cursor_col as i64);
        let click = click_cursor::Cell::new(click_row as i64, click_col as i64);
        let Some(span) = self.editable_span() else {
            return Vec::new();
        };
        // The pinned core still clamps every out-of-span click to Home/End.
        // Refuse rows belonging to completed blocks here so selecting history
        // cannot silently move the live shell cursor.
        let first_row = span.start.row.min(span.end.row);
        let last_row = span.start.row.max(span.end.row);
        if click.row < first_row || click.row > last_row {
            return Vec::new();
        }
        let Some(target) = click_cursor::target_cell(cursor, click, columns, Some(span)) else {
            return Vec::new();
        };

        let steps = click_cursor::char_steps(cursor, target, columns, |row, col| {
            row >= 0
                && col >= 0
                && (row as usize) < self.grid.rows()
                && (col as usize) < self.grid.cols()
                && self
                    .grid
                    .get(row as usize, col as usize)
                    .flags
                    .wide_continuation()
        });
        click_cursor::arrow_bytes(steps, self.is_application_cursor_keys())
    }

    /// 获取当前可见行的wrapped状态，用于跨行链接检测
    pub fn get_visible_row_wrapped(&self) -> Vec<bool> {
        let rows = self.grid.rows();

        if self.scroll_offset == 0 {
            // Fast path: just get current grid wrapped flags
            self.grid.row_wrapped.clone()
        } else {
            // Slow path: need to reconstruct from scrollback
            // For simplicity, when scrolling we disable wrapped link detection
            // by returning all false (can be improved later with full reflow)
            vec![false; rows]
        }
    }

    pub fn get_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output_buffer)
    }

    #[inline]
    pub(super) fn viewport_row_to_absolute(&self, viewport_row: usize) -> usize {
        self.scrollback.len().saturating_sub(self.scroll_offset) + viewport_row
    }

    #[inline]
    pub fn absolute_row_to_viewport(&self, absolute_row: usize) -> Option<usize> {
        let top = self.viewport_row_to_absolute(0);
        let viewport_row = absolute_row.checked_sub(top)?;
        (viewport_row < self.grid.rows()).then_some(viewport_row)
    }

    pub fn clear_text_selection(&mut self) {
        if self.selection.take().is_some() || self.projected_selection.take().is_some() {
            self.bump_selection_revision();
        }
    }

    #[allow(dead_code)] // Public library compatibility; UI may query raw storage directly.
    pub fn has_text_selection(&self) -> bool {
        self.selection.is_some() || self.projected_selection.is_some()
    }

    fn set_raw_selection(&mut self, selection: Selection) {
        self.projected_selection = None;
        self.selection = Some(selection);
        self.bump_selection_revision();
    }

    fn set_projected_selection(&mut self, selection: ProjectedSelection) {
        self.selection = None;
        self.projected_selection = Some(selection);
        self.bump_selection_revision();
    }

    fn projection_plan_key_matches_current_source(&self, key: &ProjectionPlanCacheKey) -> bool {
        self.finished_output_revision != 0
            && key.finished_output_revision != 0
            && key.total_lines_scrolled == self.total_lines_scrolled
            && key.row_identity_revision == self.row_identity_revision
            && key.finished_output_revision == self.finished_output_revision
            && key.scrollback_len == self.scrollback.len()
            && key.rows == self.grid.rows()
            && key.cols == self.grid.row_len().max(1)
            && key.row_wrapped.as_slice() == self.grid.row_wrapped.as_slice()
    }

    fn projected_selection_plan_revision(&self, viewport: &ProjectedViewport) -> Option<u64> {
        let revision = viewport.plan_revision()?;
        let (key, plan) = self.projection_plan_cache.as_ref()?;
        (plan.plan_revision == revision
            && self.projection_plan_key_matches_current_source(key)
            && plan.document_rows() == viewport.document_rows())
        .then_some(revision)
    }

    /// Normalize a wide continuation to its glyph body, then return a stable
    /// projected-document coordinate. A structural hole in a partial row is
    /// not selectable; a genuine empty tracked raw row is.
    fn projected_selection_point(
        viewport: &ProjectedViewport,
        display_pos: (usize, usize),
    ) -> Option<ProjectedSelectionEndpoint> {
        let mut point = DisplayPoint::new(display_pos.0, display_pos.1);
        if viewport
            .cells()
            .get(point.row)?
            .get(point.column)?
            .flags
            .wide_continuation()
            && point.column > 0
        {
            point.column -= 1;
        }
        let anchor = match viewport.raw_anchor_at(point) {
            Some(anchor) => ProjectedSelectionAnchor::Cell(anchor),
            None => {
                if viewport.row_has_origin(point.row) {
                    return None;
                }
                let source = viewport.row_source_at(point.row)?;
                if !source.raw_row.is_tracked()
                    || !matches!(viewport.row_kind(point.row), Some(ProjectedRowKind::Raw))
                {
                    return None;
                }
                ProjectedSelectionAnchor::Row {
                    row: source.raw_row,
                    column: point.column,
                }
            }
        };
        Some(ProjectedSelectionEndpoint {
            point: (viewport.view_document_row(point.row)?, point.column),
            anchor,
        })
    }

    fn wide_atomic_selection_columns(
        mut left: usize,
        mut right: usize,
        mut flags_at: impl FnMut(usize) -> Option<StyleFlags>,
    ) -> Option<(usize, usize)> {
        if left > right {
            return None;
        }
        if flags_at(left).is_some_and(|flags| flags.wide_continuation()) {
            left = left.checked_sub(1)?;
        }
        if flags_at(right).is_some_and(|flags| flags.wide()) {
            let continuation = right.checked_add(1)?;
            if flags_at(continuation).is_some_and(|flags| flags.wide_continuation()) {
                right = continuation;
            }
        }
        Some((left, right))
    }

    fn projected_plan_cell_flags(
        &self,
        planned: &ProjectionPlanRow,
        view_column: usize,
        history_cache: &mut Option<(usize, Vec<TerminalCell>)>,
    ) -> Option<StyleFlags> {
        let slice = planned.raw_slices.iter().find(|slice| {
            view_column >= slice.view_col_start
                && view_column < slice.view_col_start.saturating_add(slice.len)
        })?;
        let source_column = slice
            .source
            .col_start
            .checked_add(view_column.checked_sub(slice.view_col_start)?)?;
        let source_absolute = self.projection_source_to_absolute(slice.source.absolute_row)?;
        if source_absolute < self.scrollback.len() {
            if history_cache.as_ref().map(|(row, _)| *row) != Some(source_absolute) {
                *history_cache = Some((
                    source_absolute,
                    self.scrollback.get(source_absolute)?.decompress(),
                ));
            }
            return history_cache
                .as_ref()?
                .1
                .get(source_column)
                .map(|cell| cell.flags);
        }
        let grid_row = slice.source.absolute_row;
        let grid_row = self
            .projection_source_to_absolute(grid_row)?
            .checked_sub(self.scrollback.len())?;
        (grid_row < self.grid.rows())
            .then(|| self.grid[grid_row].get(source_column))
            .flatten()
            .map(|cell| cell.flags)
    }

    /// Start a new selection at a viewport-relative position.
    /// Converts to absolute buffer coordinates internally.
    #[allow(dead_code)] // Public library compatibility; UI uses the projected variant.
    pub fn start_selection(&mut self, viewport_pos: (usize, usize)) {
        self.start_selection_with_mode(viewport_pos, SelectionMode::Normal);
    }

    pub fn start_selection_projected(
        &mut self,
        viewport: &ProjectedViewport,
        display_pos: (usize, usize),
    ) {
        self.start_selection_with_projected_mode(viewport, display_pos, SelectionMode::Normal);
    }

    #[allow(dead_code)] // Public library compatibility; UI uses the projected variant.
    pub fn start_block_selection(&mut self, viewport_pos: (usize, usize)) {
        self.start_selection_with_mode(viewport_pos, SelectionMode::Block);
    }

    pub fn start_block_selection_projected(
        &mut self,
        viewport: &ProjectedViewport,
        display_pos: (usize, usize),
    ) {
        self.start_selection_with_projected_mode(viewport, display_pos, SelectionMode::Block);
    }

    #[allow(dead_code)]
    pub(super) fn start_selection_with_mode(
        &mut self,
        viewport_pos: (usize, usize),
        mode: SelectionMode,
    ) {
        let abs = (
            self.viewport_row_to_absolute(viewport_pos.0),
            viewport_pos.1,
        );
        self.set_raw_selection(Selection {
            anchor: abs,
            active: abs,
            mode,
        });
    }

    fn start_selection_with_projected_mode(
        &mut self,
        viewport: &ProjectedViewport,
        display_pos: (usize, usize),
        mode: SelectionMode,
    ) {
        if viewport.is_transformed() {
            let Some(plan_revision) = self.projected_selection_plan_revision(viewport) else {
                self.clear_text_selection();
                return;
            };
            let Some(point) = Self::projected_selection_point(viewport, display_pos) else {
                self.clear_text_selection();
                return;
            };
            self.set_projected_selection(Self::new_projected_selection(
                viewport,
                plan_revision,
                point,
                point,
                mode,
            ));
            return;
        }
        let abs = (viewport.legacy_absolute_row(display_pos.0), display_pos.1);
        self.set_raw_selection(Selection {
            anchor: abs,
            active: abs,
            mode,
        });
    }

    /// Update the active end of the current selection with a viewport-relative position.
    #[allow(dead_code)] // Public library compatibility; UI uses the projected variant.
    pub fn update_selection(&mut self, viewport_pos: (usize, usize)) {
        let abs_row = self.viewport_row_to_absolute(viewport_pos.0);
        if let Some(ref mut sel) = self.selection {
            sel.active = (abs_row, viewport_pos.1);
            self.bump_selection_revision();
        }
    }

    pub fn update_selection_projected(
        &mut self,
        viewport: &ProjectedViewport,
        display_pos: (usize, usize),
    ) {
        if viewport.is_transformed() {
            let Some(plan_revision) = self.projected_selection_plan_revision(viewport) else {
                self.clear_text_selection();
                return;
            };
            let Some(point) = Self::projected_selection_point(viewport, display_pos) else {
                // Dragging across a summary/padding row keeps the last valid
                // endpoint; synthetic rows never become anchors themselves.
                return;
            };
            let Some(selection) = self.projected_selection.as_mut() else {
                return;
            };
            if selection.plan_revision != plan_revision {
                self.clear_text_selection();
                return;
            }
            if selection.active.point != point.point {
                selection.active = point;
                self.bump_selection_revision();
            }
            return;
        }
        self.projected_selection = None;
        let abs_row = viewport.legacy_absolute_row(display_pos.0);
        if let Some(ref mut sel) = self.selection {
            sel.active = (abs_row, display_pos.1);
            self.bump_selection_revision();
        }
    }

    /// Select the word at the given (row, col) position in the visible grid.
    /// Word boundaries are determined by character class: alphanumeric/underscore,
    /// whitespace, or punctuation/symbols.
    #[allow(dead_code)] // Public library compatibility; UI uses the projected variant.
    pub fn select_word_at(&mut self, row: usize, col: usize) {
        let visible = self.get_visible_cells();
        let wrapped = self.get_visible_row_wrapped();
        let viewport_base = self.viewport_row_to_absolute(0);
        self.select_word_in_view(visible.as_ref(), &wrapped, row, col, |viewport_row| {
            viewport_base.saturating_add(viewport_row)
        });
    }

    pub fn select_word_at_projected(
        &mut self,
        viewport: &ProjectedViewport,
        row: usize,
        col: usize,
    ) {
        if viewport.is_transformed() {
            let Some(plan_revision) = self.projected_selection_plan_revision(viewport) else {
                self.clear_text_selection();
                return;
            };
            let Some((start, end)) =
                Self::word_span_in_view(viewport.cells(), viewport.row_wrapped(), row, col)
            else {
                return;
            };
            let (Some(anchor), Some(active)) = (
                Self::projected_selection_point(viewport, start),
                Self::projected_selection_point(viewport, end),
            ) else {
                self.clear_text_selection();
                return;
            };
            self.set_projected_selection(Self::new_projected_selection(
                viewport,
                plan_revision,
                anchor,
                active,
                SelectionMode::Normal,
            ));
            return;
        }
        self.select_word_in_view(
            viewport.cells(),
            viewport.row_wrapped(),
            row,
            col,
            |display_row| viewport.legacy_absolute_row(display_row),
        );
    }

    fn select_word_in_view(
        &mut self,
        visible: &[Vec<TerminalCell>],
        wrapped: &[bool],
        row: usize,
        col: usize,
        to_absolute_row: impl Fn(usize) -> usize,
    ) {
        let Some((anchor, active)) = Self::word_span_in_view(visible, wrapped, row, col) else {
            return;
        };
        self.set_raw_selection(Selection {
            anchor: (to_absolute_row(anchor.0), anchor.1),
            active: (to_absolute_row(active.0), active.1),
            mode: SelectionMode::Normal,
        });
    }

    fn word_span_in_view(
        visible: &[Vec<TerminalCell>],
        wrapped: &[bool],
        row: usize,
        col: usize,
    ) -> Option<((usize, usize), (usize, usize))> {
        let line = visible.get(row)?;
        let cols = line.len();
        if col >= cols || cols == 0 {
            return None;
        }
        let mut start_col = col;
        if line[start_col].flags.wide_continuation() && start_col > 0 {
            start_col -= 1;
        }
        let mut logical_start_row = row;
        while logical_start_row > 0 && wrapped.get(logical_start_row - 1).copied().unwrap_or(false)
        {
            logical_start_row -= 1;
        }
        let mut logical_end_row = row;
        while logical_end_row + 1 < visible.len()
            && wrapped.get(logical_end_row).copied().unwrap_or(false)
        {
            logical_end_row += 1;
        }
        let mut logical_cells =
            Vec::with_capacity((logical_end_row - logical_start_row + 1).saturating_mul(cols));
        for logical_row in visible
            .iter()
            .take(logical_end_row.saturating_add(1))
            .skip(logical_start_row)
        {
            logical_cells.extend_from_slice(logical_row);
        }
        let logical_col = (row - logical_start_row)
            .saturating_mul(cols)
            .saturating_add(start_col);

        if let Some((left, right)) = Self::select_extended_token_span(&logical_cells, logical_col) {
            let first_row = logical_start_row + left / cols;
            let last_row = logical_start_row + right / cols;
            return Some(((first_row, left % cols), (last_row, right % cols)));
        }

        let ch = line[start_col].character;
        let class = char_class(ch);

        // Expand left
        let mut left = start_col;
        while left > 0 {
            let prev = left - 1;
            let c = line[prev].character;
            if line[prev].flags.wide_continuation() {
                left = prev;
                continue;
            }
            if char_class(c) != class {
                break;
            }
            left = prev;
        }

        // Expand right
        let mut right = start_col;
        loop {
            let next = if line[right].flags.wide() {
                right + 2
            } else {
                right + 1
            };
            if next >= cols {
                break;
            }
            if line[next].flags.wide_continuation() {
                // shouldn't happen after a non-wide char, but skip
                if next + 1 < cols {
                    if char_class(line[next + 1].character) != class {
                        break;
                    }
                    right = next + 1;
                    continue;
                }
                break;
            }
            if char_class(line[next].character) != class {
                break;
            }
            right = next;
        }
        // If the selected end is a wide char, include its continuation cell
        if line[right].flags.wide() && right + 1 < cols {
            right += 1;
        }

        Some(((row, left), (row, right)))
    }

    #[allow(dead_code)] // Public library compatibility; UI uses the projected variant.
    pub fn select_line_at(&mut self, row: usize) {
        let visible = self.get_visible_cells();
        let viewport_base = self.viewport_row_to_absolute(0);
        self.select_line_in_view(visible.as_ref(), row, |viewport_row| {
            viewport_base.saturating_add(viewport_row)
        });
    }

    pub fn select_line_at_projected(&mut self, viewport: &ProjectedViewport, row: usize) {
        if viewport.is_transformed() {
            let Some(plan_revision) = self.projected_selection_plan_revision(viewport) else {
                self.clear_text_selection();
                return;
            };
            let Some((left, mut right)) = viewport.real_column_bounds(row) else {
                self.clear_text_selection();
                return;
            };
            let Some(line) = viewport.cells().get(row) else {
                self.clear_text_selection();
                return;
            };
            while right > left {
                let cell = &line[right];
                if !cell.flags.wide_continuation() && cell.character != ' ' {
                    break;
                }
                right -= 1;
            }
            if line
                .get(right)
                .is_some_and(|cell| cell.flags.wide() && right + 1 < line.len())
            {
                right += 1;
            }
            let (Some(anchor), Some(active)) = (
                Self::projected_selection_point(viewport, (row, left)),
                Self::projected_selection_point(viewport, (row, right)),
            ) else {
                self.clear_text_selection();
                return;
            };
            self.set_projected_selection(Self::new_projected_selection(
                viewport,
                plan_revision,
                anchor,
                active,
                SelectionMode::Normal,
            ));
            return;
        }
        self.select_line_in_view(viewport.cells(), row, |display_row| {
            viewport.legacy_absolute_row(display_row)
        });
    }

    fn select_line_in_view(
        &mut self,
        visible: &[Vec<TerminalCell>],
        row: usize,
        to_absolute_row: impl Fn(usize) -> usize,
    ) {
        if row >= visible.len() {
            return;
        }

        let line = &visible[row];
        let mut right = line.len().saturating_sub(1);
        while right > 0 {
            let cell = &line[right];
            if !cell.flags.wide_continuation() && cell.character != ' ' {
                break;
            }
            right -= 1;
        }

        if line
            .get(right)
            .is_some_and(|cell| cell.flags.wide() && right + 1 < line.len())
        {
            right += 1;
        }

        let abs_row = to_absolute_row(row);
        self.set_raw_selection(Selection {
            anchor: (abs_row, 0),
            active: (abs_row, right),
            mode: SelectionMode::Normal,
        });
    }

    pub(super) fn select_extended_token_span(
        line: &[TerminalCell],
        start_col: usize,
    ) -> Option<(usize, usize)> {
        let cols = line.len();
        if start_col >= cols {
            return None;
        }

        let start_char = line[start_col].character;
        if !is_extended_token_char(start_char) {
            return None;
        }

        let mut left = start_col;
        while left > 0 {
            let prev = left - 1;
            if line[prev].flags.wide_continuation() {
                left = prev;
                continue;
            }
            if !is_extended_token_char(line[prev].character) {
                break;
            }
            left = prev;
        }

        let mut right = start_col;
        loop {
            let next = if line[right].flags.wide() {
                right + 2
            } else {
                right + 1
            };
            if next >= cols {
                break;
            }
            if line[next].flags.wide_continuation() {
                if next + 1 < cols && is_extended_token_char(line[next + 1].character) {
                    right = next + 1;
                    continue;
                }
                break;
            }
            if !is_extended_token_char(line[next].character) {
                break;
            }
            right = next;
        }

        while left < start_col && is_token_prefix_wrapper(line[left].character) {
            left += 1;
        }

        while right > start_col && is_token_suffix_wrapper(line[right].character) {
            right -= if line[right].flags.wide_continuation() && right > 0 {
                2
            } else {
                1
            };
        }

        if left > right || start_col < left || start_col > right {
            return None;
        }

        let mut has_alnum = false;
        let mut has_separator = false;
        for cell in &line[left..=right] {
            if cell.flags.wide_continuation() {
                continue;
            }
            let ch = cell.character;
            has_alnum |= ch.is_alphanumeric();
            has_separator |= is_extended_token_separator(ch);
        }

        if !has_alnum || !has_separator {
            return None;
        }

        if line[right].flags.wide() && right + 1 < cols {
            right += 1;
        }

        Some((left, right))
    }

    fn copy_projected_selection(&self) -> Option<String> {
        let selection = self.projected_selection.as_ref()?;
        let (plan_key, plan) = self.projection_plan_cache.as_ref()?;
        if selection.plan_revision == 0
            || plan.plan_revision != selection.plan_revision
            || !self.projection_plan_key_matches_current_source(plan_key)
        {
            return None;
        }
        let (start, end) = if selection.anchor.point <= selection.active.point {
            (selection.anchor.point, selection.active.point)
        } else {
            (selection.active.point, selection.anchor.point)
        };
        if start.0 >= plan.document_rows() || end.0 >= plan.document_rows() {
            return None;
        }

        let mut result = String::new();
        // Projected slices are monotonic in raw document order, so retaining
        // only the most recently decoded history row bounds memory while
        // sharing a source used by multiple visible fragments.
        let mut history_cache: Option<(usize, Vec<TerminalCell>)> = None;
        let mut hard_break_pending = false;
        for document_row in start.0..=end.0 {
            let planned = plan.row(document_row)?;
            let (selected_left, selected_right) = if selection.mode == SelectionMode::Block {
                (
                    selection.anchor.point.1.min(selection.active.point.1),
                    selection.anchor.point.1.max(selection.active.point.1),
                )
            } else {
                (
                    if document_row == start.0 { start.1 } else { 0 },
                    if document_row == end.0 {
                        end.1
                    } else {
                        plan.cols.saturating_sub(1)
                    },
                )
            };
            let (selected_left, selected_right) =
                Self::wide_atomic_selection_columns(selected_left, selected_right, |column| {
                    self.projected_plan_cell_flags(planned, column, &mut history_cache)
                })?;

            if matches!(planned.kind, ProjectedRowKind::CollapsedSummary { .. }) {
                hard_break_pending = !result.ends_with('\n');
                continue;
            }
            if !matches!(planned.kind, ProjectedRowKind::Raw) {
                continue;
            }
            if hard_break_pending {
                result.push('\n');
                hard_break_pending = false;
            }
            if planned.raw_slices.is_empty()
                && planned.row_source.is_some()
                && selected_left <= selected_right
            {
                result.extend(std::iter::repeat_n(
                    ' ',
                    selected_right
                        .min(plan.cols.saturating_sub(1))
                        .saturating_sub(selected_left)
                        .saturating_add(1),
                ));
            }
            for slice in &planned.raw_slices {
                let slice_start = slice.view_col_start;
                let slice_end = slice_start.checked_add(slice.len)?;
                let overlap_start = slice_start.max(selected_left);
                let overlap_end = slice_end.min(selected_right.saturating_add(1));
                if overlap_start >= overlap_end {
                    continue;
                }
                let source_absolute =
                    self.projection_source_to_absolute(slice.source.absolute_row)?;
                let source = if source_absolute < self.scrollback.len() {
                    if history_cache.as_ref().map(|(row, _)| *row) != Some(source_absolute) {
                        history_cache = Some((
                            source_absolute,
                            self.scrollback.get(source_absolute)?.decompress(),
                        ));
                    }
                    history_cache.as_ref()?.1.as_slice()
                } else {
                    let grid_row = slice.source.absolute_row;
                    let grid_row = self
                        .projection_source_to_absolute(grid_row)?
                        .checked_sub(self.scrollback.len())?;
                    if grid_row >= self.grid.rows() {
                        return None;
                    }
                    &self.grid[grid_row]
                };
                let source_start = slice
                    .source
                    .col_start
                    .checked_add(overlap_start - slice_start)?;
                let source_end = source_start.checked_add(overlap_end - overlap_start)?;
                for cell in source.get(source_start..source_end)? {
                    if !cell.flags.wide_continuation() {
                        result.push(cell.character);
                    }
                }
            }
            if document_row < end.0 && (selection.mode == SelectionMode::Block || !planned.wrapped)
            {
                result.push('\n');
                hard_break_pending = false;
            }
        }
        Some(result)
    }

    pub fn copy_selection(&self) -> Option<String> {
        if self.projected_selection.is_some() {
            return self.copy_projected_selection();
        }
        self.selection.map(|sel| {
            let (start, end) = if sel.anchor <= sel.active {
                (sel.anchor, sel.active)
            } else {
                (sel.active, sel.anchor)
            };
            let mut result = String::new();
            let scrollback_len = self.scrollback.len();
            let grid_rows = self.grid.rows();
            let cols = self.grid.row_len();
            let total_rows = scrollback_len + grid_rows;

            for abs_row in start.0..=end.0.min(total_rows.saturating_sub(1)) {
                let start_col = if abs_row == start.0 { start.1 } else { 0 };
                let end_col = if abs_row == end.0 {
                    end.1.min(cols.saturating_sub(1))
                } else {
                    cols.saturating_sub(1)
                };

                // 行是否因到达行末被自动换行(软换行)。复制时软换行不应插入 \n,
                // 否则像 URL 这种被终端宽度截断的字符串会被切断成多段。
                let row_wrapped = if abs_row < scrollback_len {
                    self.scrollback[abs_row].is_wrapped
                } else {
                    let grid_row = abs_row - scrollback_len;
                    self.grid
                        .row_wrapped
                        .get(grid_row)
                        .copied()
                        .unwrap_or(false)
                };

                let mut line_buf = String::new();
                if abs_row < scrollback_len {
                    // Read from scrollback
                    let line = self.scrollback[abs_row].decompress();
                    for cell in line.iter().take(end_col.saturating_add(1)).skip(start_col) {
                        if !cell.flags.wide_continuation() {
                            line_buf.push(cell.character);
                        }
                    }
                } else {
                    // Read from current grid
                    let grid_row = abs_row - scrollback_len;
                    if grid_row < grid_rows {
                        for col in start_col..=end_col {
                            let cell = self.grid.get(grid_row, col);
                            if !cell.flags.wide_continuation() {
                                line_buf.push(cell.character);
                            }
                        }
                    }
                }

                // 软换行(URL 等被终端宽度截断)拼接时去掉尾部填充空白,
                // 避免还原后的字符串里夹杂大段空格。
                if row_wrapped && abs_row < end.0 {
                    let trimmed_len = line_buf.trim_end_matches(' ').len();
                    line_buf.truncate(trimmed_len);
                }

                result.push_str(&line_buf);

                if abs_row < end.0 && !row_wrapped {
                    result.push('\n');
                }
            }

            result
        })
    }

    pub fn scroll(&mut self, lines: isize) {
        // Don't scroll ordinary alternate-screen apps (less, vim, git log, etc.).
        // Synchronized TUIs such as Codex may archive snapshots into local
        // scrollback, in which case wheel/scrollbar navigation should work.
        if self.use_alt_buffer && self.scrollback.is_empty() {
            return;
        }

        if lines > 0 {
            // Scroll up (show earlier lines)
            self.scroll_offset = self.scroll_offset.saturating_add(lines as usize);
        } else {
            // Scroll down (show later lines)
            self.scroll_offset = self.scroll_offset.saturating_sub((-lines) as usize);
        }

        // Clamp scroll_offset to valid range
        let max_scroll = self.scrollback.len();
        self.scroll_offset = self.scroll_offset.min(max_scroll);

        // Selection endpoints are absolute buffer coordinates, so moving the
        // viewport must not discard them. `row_selection_cols` remaps the same
        // selection onto whichever part is currently visible; a later plain
        // primary click is responsible for clearing it.

        // When scrolling to bottom (offset 0), reset to live view
        if self.scroll_offset == 0 {
            self.scroll_offset = 0;
        }
    }

    pub(super) fn strip_trailing_blanks(cells: &[TerminalCell]) -> &[TerminalCell] {
        let mut end = cells.len();
        while end > 0 && cells[end - 1].is_reflow_trimmable_blank() {
            end -= 1;
        }
        &cells[..end]
    }

    #[cfg(test)]
    pub(super) fn reflow_lines(
        lines: &[ScrollbackLine],
        new_cols: usize,
        blank_cell: &TerminalCell,
    ) -> Vec<ScrollbackLine> {
        let mut result = Vec::new();
        let len = lines.len();
        let mut i = 0;

        while i < len {
            let mut logical_line: Vec<TerminalCell> = Vec::new();
            let decompressed = lines[i].decompress();
            logical_line.extend_from_slice(Self::strip_trailing_blanks(&decompressed));
            while i < len && lines[i].is_wrapped {
                i += 1;
                if i < len {
                    let dc = lines[i].decompress();
                    logical_line.extend_from_slice(Self::strip_trailing_blanks(&dc));
                }
            }
            i += 1;

            if logical_line.is_empty() {
                result.push(ScrollbackLine::compress(
                    &vec![*blank_cell; new_cols],
                    false,
                ));
                continue;
            }

            let chunks: Vec<&[TerminalCell]> = logical_line.chunks(new_cols).collect();
            let num_chunks = chunks.len();
            for (ci, chunk) in chunks.into_iter().enumerate() {
                if chunk.len() == new_cols {
                    result.push(ScrollbackLine::compress(chunk, ci + 1 < num_chunks));
                } else {
                    let mut cells = chunk.to_vec();
                    cells.resize(new_cols, *blank_cell);
                    result.push(ScrollbackLine::compress(&cells, ci + 1 < num_chunks));
                }
            }
        }

        result
    }

    pub fn on_resize(&mut self, cols: usize, rows: usize) {
        if cols == 0 || rows == 0 {
            return;
        }

        let (cols, rows) = clamp_terminal_dimensions(cols, rows);

        // A forced render pass may repeat the current PTY dimensions (for
        // example after focusing the already-active tab). Treat that as the
        // no-op it is: clearing scroll_offset here would undo a semantic
        // command jump performed earlier in the same frame.
        if cols == self.grid.row_len() && rows == self.grid.rows() {
            return;
        }

        // Reflow and grid resize can keep a RawRowId while changing its cell
        // geometry or contents. Exact output ownership is therefore sealed
        // to the dimensions at D; a real resize invalidates both completed
        // and in-flight provenance. Same-size repaint requests returned above.
        if !self.finished_output_provenance.is_empty() || !self.finished_output_owners.is_empty() {
            self.finished_output_provenance.clear();
            self.finished_output_owners.clear();
            self.mark_finished_output_provenance_changed();
        }
        self.active_output_provenance = None;

        // Dimensions and row contents are part of every renderer/search cache
        // key. A resize can happen while the PTY is otherwise idle, so it must
        // invalidate them independently of new parser input.
        self.grid_version = self.grid_version.saturating_add(1);
        self.invalidate_scrollback_view_cache();

        let old_rows = self.grid.rows();
        let had_full_screen_region = old_rows == 0
            || (self.scroll_region_top == 0 && self.scroll_region_bottom + 1 >= old_rows);

        let blank_cell = self.create_blank_cell();
        // `grid` is always the active screen (the buffers are swapped on
        // DECSET/DECRST 47/1047/1049), while `alt_grid` is the hidden screen.
        // Only the active screen may inherit the application's current SGR
        // background when it grows. Reusing that cell for the hidden screen
        // lets a full-screen app such as Vim paint its background into the
        // saved primary screen during a resize; the leaked block then becomes
        // visible after Vim exits.
        let inactive_blank_cell = TerminalCell::default();

        // 缩小高度时,grid.resize 默认保留顶部行、丢弃底部(含光标行与近期输出),
        // 导致缩小窗口丢失最新输出。改为把顶部溢出行压入 scrollback 并将内容上移,
        // 尽量保留底部内容并保持光标可见(与 xterm/kitty 一致)。仅处理主屏 ——
        // 备用屏应用会在 SIGWINCH 后自行重绘,无需保留。
        if rows < old_rows && !self.use_alt_buffer && !self.grid.is_empty() {
            let need = old_rows - rows;
            // 最多从顶部移除到光标所在行,避免把光标行本身推入 scrollback;
            // 剩余需移除的行位于光标下方,由随后的 grid.resize 截断(通常为空白)。
            let from_top = need.min(self.cursor_row);
            if from_top > 0 {
                for r in 0..from_top {
                    let mut line =
                        ScrollbackLine::compress(&self.grid[r], self.grid.row_wrapped[r]);
                    line.set_raw_row_id(self.grid.row_id(r));
                    self.push_scrollback_transferred_with_options(line, false);
                }
                self.grid.scroll_up_by(from_top, blank_cell);
                self.kitty_graphics
                    .scroll_region_up(0, old_rows.saturating_sub(1), from_top, true);
                self.cursor_row -= from_top;
            }
        }

        self.grid.resize(rows, cols, blank_cell);
        self.alt_grid.resize(rows, cols, inactive_blank_cell);
        self.fill_untracked_grid_row_ids();
        self.mark_row_identity_changed();
        // Narrowing can cut a double-width character in half at the new right
        // edge; its lead half cannot stay behind on its own.
        for row in 0..rows {
            self.grid.clear_dangling_wide_at_row_end(row, blank_cell);
            self.alt_grid
                .clear_dangling_wide_at_row_end(row, inactive_blank_cell);
        }
        self.kitty_graphics.resize(cols, rows);

        // CRITICAL: Sync row_versions size with grid size to prevent dirty mark loss
        // When grid grows, we need to extend row_versions; when it shrinks, truncate it
        if rows != self.row_versions.len() {
            self.row_versions.resize(rows, self.grid_version);
        }
        self.row_versions.fill(self.grid_version);
        self.dirty_region.mark_all(rows);

        // Clamp, do not zero. A resize is a window drag, a pane split, a font
        // change or a sidebar toggle — none of them is a request to leave the
        // history the user is reading. The shrink branch above deliberately
        // routes its evicted rows through the pinning push, which increments
        // scroll_offset so the top anchor row does not move; zeroing here threw
        // that away. frost pins the offset across a resize for the same reason.
        self.scroll_offset = self.scroll_offset.min(self.scrollback.len());
        self.pending_wrap = false;
        self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        self.cursor_col = self.cursor_col.min(cols.saturating_sub(1));
        self.saved_cursor_row = self.saved_cursor_row.min(rows.saturating_sub(1));
        self.saved_cursor_col = self.saved_cursor_col.min(cols.saturating_sub(1));

        // Resize tab stops: keep existing stops, default new columns to every 8th.
        if cols != self.tab_stops.len() {
            let old_len = self.tab_stops.len();
            self.tab_stops.resize(cols, false);
            for c in old_len..cols {
                self.tab_stops[c] = c % 8 == 0;
            }
        }

        // Clamp saved cursor state (DECSC/CSI s) to new bounds.
        if let Some(s) = self.saved_state.as_mut() {
            s.row = s.row.min(rows.saturating_sub(1));
            s.col = s.col.min(cols.saturating_sub(1));
        }
        self.alt_cursor_row = self.alt_cursor_row.min(rows.saturating_sub(1));
        self.alt_cursor_col = self.alt_cursor_col.min(cols.saturating_sub(1));
        if had_full_screen_region {
            self.scroll_region_top = 0;
            self.scroll_region_bottom = rows.saturating_sub(1);
        } else {
            self.scroll_region_top = self.scroll_region_top.min(rows.saturating_sub(1));
            self.scroll_region_bottom = self.scroll_region_bottom.min(rows.saturating_sub(1));

            if self.scroll_region_top > self.scroll_region_bottom {
                self.scroll_region_top = 0;
                self.scroll_region_bottom = rows.saturating_sub(1);
            }
        }
    }

    pub fn get_dimensions(&self) -> (usize, usize) {
        if self.grid.is_empty() {
            (0, 0)
        } else {
            (self.grid.row_len(), self.grid.rows())
        }
    }

    #[inline]
    #[allow(dead_code)] // Public library compatibility; renderer uses projected rows.
    pub fn row_selection_cols(&self, viewport_row: usize) -> Option<(usize, usize)> {
        self.row_selection_cols_at_absolute(self.viewport_row_to_absolute(viewport_row))
    }

    pub fn row_selection_cols_projected(
        &self,
        viewport: &ProjectedViewport,
        display_row: usize,
    ) -> Option<(usize, usize)> {
        if viewport.is_transformed() {
            let selection = self.projected_selection.as_ref()?;
            if self.projected_selection_plan_revision(viewport)? != selection.plan_revision
                || !matches!(viewport.row_kind(display_row), Some(ProjectedRowKind::Raw))
            {
                return None;
            }
            let document_row = viewport.view_document_row(display_row)?;
            let (start, end) = if selection.anchor.point <= selection.active.point {
                (selection.anchor.point, selection.active.point)
            } else {
                (selection.active.point, selection.anchor.point)
            };
            if document_row < start.0 || document_row > end.0 {
                return None;
            }
            let (mut left, mut right) = if selection.mode == SelectionMode::Block {
                (
                    selection.anchor.point.1.min(selection.active.point.1),
                    selection.anchor.point.1.max(selection.active.point.1),
                )
            } else {
                (
                    if document_row == start.0 { start.1 } else { 0 },
                    if document_row == end.0 {
                        end.1
                    } else {
                        usize::MAX
                    },
                )
            };
            let (real_left, real_right) = viewport.real_column_bounds(display_row)?;
            left = left.max(real_left);
            right = right.min(real_right);
            let cells = viewport.cells().get(display_row)?;
            (left, right) = Self::wide_atomic_selection_columns(left, right, |column| {
                cells.get(column).map(|cell| cell.flags)
            })?;
            return (left <= right).then_some((left, right));
        }
        self.row_selection_cols_at_absolute(viewport.legacy_absolute_row(display_row))
    }

    fn row_selection_cols_at_absolute(&self, abs_row: usize) -> Option<(usize, usize)> {
        let sel = self.selection?;
        let (start, end) = if sel.anchor <= sel.active {
            (sel.anchor, sel.active)
        } else {
            (sel.active, sel.anchor)
        };

        if abs_row < start.0 || abs_row > end.0 {
            return None;
        }

        match sel.mode {
            SelectionMode::Block => {
                let col_min = sel.anchor.1.min(sel.active.1);
                let col_max = sel.anchor.1.max(sel.active.1);
                Some((col_min, col_max))
            }
            SelectionMode::Normal => {
                let col_start = if abs_row == start.0 { start.1 } else { 0 };
                let col_end = if abs_row == end.0 { end.1 } else { usize::MAX };
                Some((col_start, col_end))
            }
        }
    }

    // IME support methods
    pub fn set_preedit(&mut self, text: String, cursor: usize) {
        self.preedit_text = text;
        self.preedit_cursor = cursor;
    }

    pub fn clear_preedit(&mut self) {
        self.preedit_text.clear();
        self.preedit_cursor = 0;
    }
}
