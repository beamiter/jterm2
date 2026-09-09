pub mod agent;
mod agent_panel;
mod ai_chat_panel;
mod ai_chat_store;
mod ai_command_suggestion;
mod app;
mod block_export;
mod block_mode;
mod block_search;
mod bottom_bar;
mod clipboard;
mod color;
mod command_correction;
mod command_palette;
mod config;
mod config_panel;
mod debug;
mod debug_panel;
mod execution_journal;
mod font_file;
mod gpu;
mod help;
mod history_persistence;
mod history_picker;
mod image_drop;
mod jsh_ui;
mod keybindings;
mod kitty_graphics;
mod layout;
mod link;
mod pane_header;
mod persistence_file;
mod pty;
mod remote_fs;
mod remote_picker;
mod review_text;
mod search;
mod search_replace;
mod search_replace_panel;
mod session;
mod session_manager;
mod session_persistence;
mod shell;
mod sidebar;
mod ssh_files_follow;
mod tab_manager;
mod terminal;
mod theme;
mod ui;
mod windows_compat;
mod workflow_picker;
mod workflows;

use crate::theme::ThemeExt as _;
use app::events::{
    normalize_terminal_shortcut_events, restore_missing_image_paste_key_event,
    semantic_paste_modifiers, should_restore_terminal_shortcut_event,
};
use base64::Engine;
use clipboard::{ClipboardContent, ClipboardManager};
use eframe::egui;
pub(crate) use jterm_core::char_width;
use parking_lot::Mutex as ParkingMutex;
use session::Session;
use session_manager::{ProtocolResponseQueueError, ProtocolResponseSender, SessionManager};
use shell::{ShellEvent, ShellSession};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use terminal::{clamp_terminal_dimensions, TerminalState};
use ui::{grid_position_from_content, TerminalRenderer};

// 全局标志，用于信号处理
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Must stay equal to the installed entry's basename
/// (`data/io.github.beamiter.ember.desktop`): the desktop shell pairs a window
/// with its launcher entry through this id.
const WINDOW_APP_ID: &str = "io.github.beamiter.ember";

/// The same artwork the installer puts in the icon theme, embedded so a window
/// carries its icon even when the .desktop entry is not installed.
const WINDOW_ICON_PNG: &[u8] = include_bytes!("../data/io.github.beamiter.ember-128.png");

/// A nonzero shell exit inside this window after spawn means the shell never
/// became interactive: no human could have typed `exit` that fast.
const SHELL_STARTUP_GRACE: Duration = Duration::from_millis(1500);
#[cfg(test)]
static NEXT_KITTY_PASTE_IMAGE_ID: AtomicU32 = AtomicU32::new(1);
#[cfg(test)]
const KITTY_BASE64_CHUNK_BYTES: usize = 4096;

fn workspace_drag_pointer_cancelled(
    primary_down: bool,
    any_released: bool,
    has_pointer: bool,
) -> bool {
    (!primary_down || !has_pointer) && !any_released
}

/// 设置信号处理器，确保收到SIGINT/SIGTERM时能正常退出
/// 这允许Drop逻辑执行，从而清理所有jsh子进程
#[cfg(unix)]
fn setup_signal_handlers() {
    extern "C" fn handle_signal(_: libc::c_int) {
        SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
        // 注意：在信号处理器中只能做最少的工作
        // 主程序会检查SHUTDOWN_REQUESTED并正常退出
    }

    // 注册SIGINT (Ctrl+C) 和 SIGTERM (kill) 处理器
    // SAFETY: signal 注册信号处理器。handle_signal 是 extern "C" 函数，
    // 符合 sighandler_t 的签名要求。处理器只做最小工作（设置原子标志），是信号安全的。
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
    }
}

#[cfg(not(unix))]
fn setup_signal_handlers() {
    // Windows平台暂不支持
}

fn register_font_family(
    fonts: &mut egui::FontDefinitions,
    family: egui::FontFamily,
    font_name: &str,
    prepend: bool,
) {
    if let Some(entries) = fonts.families.get_mut(&family) {
        if entries.iter().any(|entry| entry == font_name) {
            return;
        }

        if prepend {
            entries.insert(0, font_name.to_owned());
        } else {
            entries.push(font_name.to_owned());
        }
    }
}

fn load_font_from_path(
    fonts: &mut egui::FontDefinitions,
    loaded_paths: &mut HashMap<String, String>,
    path: &str,
    font_name: &str,
    families: &[egui::FontFamily],
    prepend: bool,
) -> bool {
    let registered_name = if let Some(existing_name) = loaded_paths.get(path) {
        existing_name.clone()
    } else {
        let font_data = match font_file::read_font_file(std::path::Path::new(path)) {
            Ok(font_data) => font_data,
            Err(error) => {
                eprintln!("[Fonts] Skipping {}: {}", path, error);
                return false;
            }
        };

        fonts.font_data.insert(
            font_name.to_owned(),
            std::sync::Arc::new(egui::FontData::from_owned(font_data)),
        );
        loaded_paths.insert(path.to_owned(), font_name.to_owned());
        font_name.to_owned()
    };

    for family in families {
        register_font_family(fonts, family.clone(), &registered_name, prepend);
    }

    eprintln!("[Fonts] Loaded {} from {}", registered_name, path);
    true
}

#[cfg(target_os = "linux")]
fn fontconfig_match_file(family: &str) -> Option<String> {
    let output = jterm_core::helper::fc_match(&["-f", "%{file}\n", family]).ok()?;

    String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .map(str::trim)
        .find(|path| !path.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(target_os = "linux")]
fn fontconfig_match_bold_file(family: &str) -> Option<String> {
    let query = format!("{}:style=Bold", family);
    let output = jterm_core::helper::fc_match(&["-f", "%{file}\n", &query]).ok()?;

    let path = String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .map(str::trim)
        .find(|path| !path.is_empty())
        .map(ToOwned::to_owned)?;

    // Verify it's actually a bold variant (not the same as regular)
    if path.to_lowercase().contains("bold") {
        Some(path)
    } else {
        None
    }
}

#[cfg(not(target_os = "linux"))]
fn fontconfig_match_file(_family: &str) -> Option<String> {
    None
}

/// Resolve `family` to a file, but only when fontconfig really has that family.
///
/// `fc-match` always answers with *something*, so an unverified lookup for a
/// font that is not installed quietly returns the default monospace file. In a
/// fallback stack that is worse than no answer: the substitute takes the slot
/// the icon or CJK font was meant to fill, and every glyph it lacks falls
/// through to whatever fontconfig happens to sort next.
#[cfg(target_os = "linux")]
fn fontconfig_match_family_file(family: &str) -> Option<String> {
    let output = jterm_core::helper::fc_match(&["-f", "%{family}\t%{file}\n", family]).ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let (families, path) = stdout.lines().find_map(|line| line.split_once('\t'))?;
    let wanted = family.trim();
    families
        .split(',')
        .any(|candidate| candidate.trim().eq_ignore_ascii_case(wanted))
        .then(|| path.trim().to_owned())
        .filter(|path| !path.is_empty())
}

#[cfg(not(target_os = "linux"))]
fn fontconfig_match_family_file(_family: &str) -> Option<String> {
    None
}

#[cfg(not(target_os = "linux"))]
fn fontconfig_match_bold_file(_family: &str) -> Option<String> {
    None
}

fn create_font_backend(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cfg: &config::Config,
    font_bytes: &[u8],
    bold_font_data: Option<&[u8]>,
    fallback_font_data: &[Vec<u8>],
    font_size_px: f32,
) -> Box<dyn gpu::font_backend::FontBackend> {
    match cfg.font_backend {
        config::FontBackendType::Fontdue => Box::new(gpu::fontdue_backend::FontdueAtlas::new(
            device,
            queue,
            font_bytes,
            bold_font_data,
            fallback_font_data,
            font_size_px,
            cfg.font_weight,
            cfg.subpixel_rendering,
        )),
        config::FontBackendType::AbGlyph => Box::new(gpu::ab_glyph_backend::AbGlyphAtlas::new(
            device,
            queue,
            font_bytes,
            bold_font_data,
            fallback_font_data,
            font_size_px,
            cfg.font_weight,
        )),
    }
}

fn load_first_matching_font(
    fonts: &mut egui::FontDefinitions,
    loaded_paths: &mut HashMap<String, String>,
    family_candidates: &[&str],
    path_candidates: &[&str],
    font_name: &str,
    families: &[egui::FontFamily],
    prepend: bool,
) -> bool {
    let mut seen_paths = HashSet::new();
    let mut resolved_paths = Vec::new();

    for family in family_candidates {
        if let Some(path) = fontconfig_match_file(family) {
            if seen_paths.insert(path.clone()) {
                resolved_paths.push(path);
            }
        }
    }

    for path in path_candidates {
        let path = (*path).to_owned();
        if seen_paths.insert(path.clone()) {
            resolved_paths.push(path);
        }
    }

    for path in resolved_paths {
        if load_font_from_path(fonts, loaded_paths, &path, font_name, families, prepend) {
            return true;
        }
    }

    false
}

fn load_matching_fallback_fonts(
    fonts: &mut egui::FontDefinitions,
    loaded_paths: &mut HashMap<String, String>,
    family_candidates: &[&str],
    path_candidates: &[&str],
    font_name_prefix: &str,
    families: &[egui::FontFamily],
) -> Vec<String> {
    let mut seen_paths = HashSet::new();
    let mut resolved_paths = Vec::new();

    for family in family_candidates {
        // Verified: an unavailable fallback must drop out of the stack
        // rather than be replaced by a substitute that has none of the
        // glyphs this slot exists to supply.
        if let Some(path) = fontconfig_match_family_file(family) {
            if seen_paths.insert(path.clone()) {
                resolved_paths.push(path);
            }
        }
    }

    for path in path_candidates {
        let path = (*path).to_owned();
        if seen_paths.insert(path.clone()) {
            resolved_paths.push(path);
        }
    }

    let mut loaded_names = Vec::new();
    for (idx, path) in resolved_paths.iter().enumerate() {
        let font_name = format!("{}_{}", font_name_prefix, idx);
        if load_font_from_path(fonts, loaded_paths, path, &font_name, families, false) {
            if let Some(name) = loaded_paths.get(path) {
                if !loaded_names.iter().any(|loaded| loaded == name) {
                    loaded_names.push(name.clone());
                }
            }
        }
    }

    loaded_names
}

/// 从 PNG 数据中提取宽度和高度
#[cfg(test)]
fn extract_png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if data.len() < 24
        || !data.starts_with(PNG_SIGNATURE)
        || data.get(12..16) != Some(b"IHDR".as_slice())
    {
        return None;
    }

    // PNG 宽度在偏移 16-19，高度在 20-23
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

    if width == 0 || height == 0 {
        return None;
    }

    crate::debug_log!("[KITTY] PNG dimensions: {}x{}", width, height);
    Some((width, height))
}

#[cfg(test)]
fn next_kitty_paste_image_id() -> u32 {
    loop {
        let id = NEXT_KITTY_PASTE_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

/// Encode a PNG as standard Kitty direct-transfer APC chunks, followed by a
/// separate put command. Each base64 payload is at most 4096 bytes; the final
/// transfer chunk uses m=0 and retains normal RFC 4648 padding.
#[cfg(test)]
fn kitty_graphics_payload(mime_type: &str, data: &[u8]) -> Option<Vec<u8>> {
    crate::debug_log!(
        "[KITTY] generating payload: mime_type={}, data_size={}",
        mime_type,
        data.len()
    );

    if mime_type != "image/png" || extract_png_dimensions(data).is_none() {
        return None;
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(data);
    crate::debug_log!(
        "[KITTY] encoded data size (base64): {} bytes",
        encoded.len()
    );

    let image_id = next_kitty_paste_image_id();
    let chunks: Vec<&[u8]> = encoded
        .as_bytes()
        .chunks(KITTY_BASE64_CHUNK_BYTES)
        .collect();
    let mut output = Vec::with_capacity(encoded.len() + chunks.len() * 32 + 48);
    for (index, chunk) in chunks.iter().enumerate() {
        let more = u8::from(index + 1 < chunks.len());
        if index == 0 {
            output.extend_from_slice(format!("\x1b_Ga=t,f=100,i={image_id},m={more};").as_bytes());
        } else {
            output.extend_from_slice(format!("\x1b_Gm={more};").as_bytes());
        }
        output.extend_from_slice(chunk);
        output.extend_from_slice(b"\x1b\\");
    }
    output.extend_from_slice(format!("\x1b_Ga=p,i={image_id}\x1b\\").as_bytes());

    crate::debug_log!("[KITTY] final packet size: {} bytes", output.len());
    Some(output)
}

fn configure_fonts_and_gpu(
    ctx: &egui::Context,
    wgpu_render_state: Option<&egui_wgpu::RenderState>,
    cfg: &config::Config,
) {
    let mut fonts = egui::FontDefinitions::default();
    let mut loaded_font_paths = HashMap::new();

    let configured_mono_family = cfg.get_font_family();
    let mono_loaded = load_first_matching_font(
        &mut fonts,
        &mut loaded_font_paths,
        &[
            configured_mono_family,
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Noto Sans Mono",
            "Noto Mono",
        ],
        &[
            "/usr/share/fonts/opentype/noto/NotoMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/opentype/dejavu/DejaVuSansMono.otf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
        ],
        "monospace_unicode",
        &[egui::FontFamily::Monospace],
        true,
    );

    if !mono_loaded {
        eprintln!(
            "[Fonts] Warning: no monospace font file could be loaded for {}",
            configured_mono_family
        );
    }

    let cjk_fallbacks = load_matching_fallback_fonts(
        &mut fonts,
        &mut loaded_font_paths,
        &[
            // Prefer plain TTF/OTF fallbacks for the terminal GPU atlas. Some
            // rasterizers used below do not accept TTC collections directly.
            "Droid Sans Fallback",
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "AR PL UMing CN",
        ],
        &[
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/wenquanyi/wqy-zenhei.ttc",
            "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
        ],
        "cjk",
        &[egui::FontFamily::Monospace, egui::FontFamily::Proportional],
    );

    if cjk_fallbacks.is_empty() {
        eprintln!("[Fonts] Warning: no CJK fallback font file could be loaded");
    }

    let symbol_fallbacks = load_matching_fallback_fonts(
        &mut fonts,
        &mut loaded_font_paths,
        &[
            // Nerd Fonts' icon-only cuts exist to be somebody's fallback, so
            // they come first; the patched text fonts carry the same icon set
            // and stand in when no Symbols cut is installed.
            "Symbols Nerd Font Mono",
            "Symbols Nerd Font",
            "JetBrainsMono Nerd Font Mono",
            "SauceCodePro Nerd Font Mono",
            "JetBrainsMono Nerd Font",
            "SauceCodePro Nerd Font",
            "Noto Sans Symbols 2",
            "Noto Sans Symbols",
            "DejaVu Sans",
            "Noto Emoji",
        ],
        &[
            "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
            "/usr/share/fonts/NerdFonts/SymbolsNerdFontMono-Regular.ttf",
            "/usr/share/fonts/NerdFonts/SymbolsNerdFont-Regular.ttf",
            "/usr/share/fonts/TTF/SymbolsNerdFontMono-Regular.ttf",
            "/usr/share/fonts/TTF/SymbolsNerdFont-Regular.ttf",
            "/usr/share/fonts/truetype/nerd-fonts/SymbolsNerdFontMono-Regular.ttf",
            "/usr/share/fonts/truetype/nerd-fonts/SymbolsNerdFont-Regular.ttf",
        ],
        "symbols",
        &[egui::FontFamily::Monospace, egui::FontFamily::Proportional],
    );

    if symbol_fallbacks.is_empty() {
        eprintln!("[Fonts] Warning: no symbol fallback font file could be loaded");
    }

    let mono_font_data: Option<Vec<u8>> = fonts
        .font_data
        .get("monospace_unicode")
        .map(|fd| fd.font.to_vec());
    let fallback_font_data: Vec<Vec<u8>> = cjk_fallbacks
        .iter()
        .chain(symbol_fallbacks.iter())
        .filter_map(|font_name| fonts.font_data.get(font_name))
        .map(|fd| fd.font.to_vec())
        .collect();

    // Try to load bold variant of the configured monospace font
    let bold_font_data: Option<Vec<u8>> = {
        let family = cfg.get_font_family();
        let bold_candidates = [
            family,
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Noto Sans Mono",
            "Noto Mono",
        ];
        let mut data = None;
        for candidate in &bold_candidates {
            if let Some(path) = fontconfig_match_bold_file(candidate) {
                match font_file::read_font_file(std::path::Path::new(&path)) {
                    Ok(bytes) => {
                        eprintln!("[Fonts] Loaded bold font: {}", path);
                        data = Some(bytes);
                        break;
                    }
                    Err(error) => {
                        eprintln!("[Fonts] Skipping bold candidate {}: {}", path, error);
                    }
                }
            }
        }
        if data.is_none() {
            eprintln!(
                "[Fonts] Warning: no bold font variant found, bold text will use regular weight"
            );
        }
        data
    };

    ctx.set_fonts(fonts);

    if let (Some(render_state), Some(font_bytes)) = (wgpu_render_state, mono_font_data) {
        let font_size_px = cfg.font_size * ctx.pixels_per_point();
        let atlas = create_font_backend(
            &render_state.device,
            &render_state.queue,
            cfg,
            &font_bytes,
            bold_font_data.as_deref(),
            &fallback_font_data,
            font_size_px,
        );
        let pipeline =
            gpu::pipeline::GridPipeline::new(&render_state.device, render_state.target_format);

        let mut renderer = render_state.renderer.write();
        if let Some(gpu_res) = renderer
            .callback_resources
            .get_mut::<gpu::callback::GpuResources>()
        {
            gpu_res.replace_font_resources(atlas, pipeline);
        } else {
            let gpu_resources =
                gpu::callback::GpuResources::new(atlas, pipeline, &render_state.device);
            renderer.callback_resources.insert(gpu_resources);
        }

        eprintln!(
            "[GPU] Configured GPU terminal renderer (font_size_px={:.1})",
            font_size_px
        );
    } else {
        eprintln!("[Renderer] Using low-memory Glow renderer; terminal GPU callbacks disabled");
    }
}

fn apply_theme_visuals(ctx: &egui::Context, theme: &theme::Theme) {
    let ui = &theme.ui;
    let brightness =
        u16::from(ui.window_bg[0]) + u16::from(ui.window_bg[1]) + u16::from(ui.window_bg[2]);
    let mut visuals = if brightness > 382 {
        egui::Visuals::light()
    } else {
        egui::Visuals::dark()
    };

    visuals.window_fill = theme::Theme::rgb_to_color32(ui.window_bg);
    visuals.panel_fill = theme::Theme::rgb_to_color32(ui.panel_bg);
    visuals.extreme_bg_color = theme::Theme::rgb_to_color32(ui.panel_bg);
    visuals.override_text_color = Some(theme::Theme::rgb_to_color32(ui.text));
    visuals.widgets.noninteractive.bg_stroke.color = theme::Theme::rgb_to_color32(ui.border);
    visuals.widgets.inactive.bg_stroke.color = theme::Theme::rgb_to_color32(ui.border);
    visuals.widgets.active.bg_stroke.color = theme::Theme::rgb_to_color32(ui.border);
    visuals.widgets.hovered.bg_stroke.color = theme::Theme::rgb_to_color32(ui.border);

    ctx.set_visuals(visuals);
}

fn main() -> Result<(), eframe::Error> {
    // Keep operational failures observable by default while allowing
    // `RUST_LOG=ember=debug` (or any normal env_logger filter) to opt into
    // deeper diagnostics.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .try_init();

    // 设置panic hook，记录panic信息
    // 注意：panic时Drop可能不会被调用，但我们依赖PR_SET_PDEATHSIG确保子进程退出
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("[PANIC] ember panicked: {}", panic_info);
        eprintln!("[PANIC] Child jsh processes should exit due to PR_SET_PDEATHSIG");
    }));

    // 设置信号处理，确保收到SIGINT/SIGTERM时能正常清理
    setup_signal_handlers();

    // Shared jterm_core modules brand themselves per app (env prefixes,
    // prompt strings) from this identity.
    jterm_core::identity::init(jterm_core::identity::AppIdentity {
        app_name: "ember",
        app_id: "io.github.beamiter.ember",
        // Reported to every child shell as TERM_PROGRAM_VERSION by
        // jterm_core::child_env, so it has to be this crate's version and not
        // the core library's.
        app_version: env!("CARGO_PKG_VERSION"),
    });

    // Load configuration
    let cfg = config::Config::load();

    let renderer = match cfg.app_renderer {
        config::AppRendererType::Glow => eframe::Renderer::Glow,
        config::AppRendererType::Wgpu => eframe::Renderer::Wgpu,
    };
    // Ask for an alpha-capable surface on both backends. Glow used to be
    // excluded, which silently turned the whole `opacity` feature into a no-op
    // whenever the config (or the wgpu fallback) picked it: the clear color
    // and background alpha are only visible through a transparent window.
    let transparent = true;

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([cfg.initial_width, cfg.initial_height])
        // The desktop shell matches a window to its .desktop entry by app_id,
        // so this has to be the entry's basename or the shell shows an
        // unbranded, unpinnable window instead of ember. egui sets it as
        // winit's Wayland window name, which winit also reports as the X11
        // WM_CLASS general class, so `StartupWMClass` in the entry is this same
        // string rather than the program name a GTK app would report.
        .with_app_id(WINDOW_APP_ID)
        .with_transparent(transparent);

    // The .desktop entry only covers windows the shell can match to it. Without
    // an icon of its own the window has no _NET_WM_ICON at all, so anything
    // else — a bare `cargo run`, a session where the entry is not installed —
    // falls back to a blank placeholder in the dock and the window switcher.
    match eframe::icon_data::from_png_bytes(WINDOW_ICON_PNG) {
        Ok(icon) => viewport = viewport.with_icon(icon),
        Err(err) => log::warn!("Failed to decode the embedded window icon: {err}"),
    }

    // Request dual-source blending when the adapter offers it: the terminal
    // foreground pipeline then applies per-channel (LCD subpixel) glyph alpha
    // over transparent default-background cells. Wrap egui's default
    // device_descriptor so its limits/feature choices are preserved.
    // eframe's winit integration fills in the display handle automatically
    // when it is left empty.
    let mut wgpu_setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    let base_device_descriptor = std::sync::Arc::clone(&wgpu_setup.device_descriptor);
    wgpu_setup.device_descriptor = std::sync::Arc::new(move |adapter| {
        let mut descriptor = base_device_descriptor(adapter);
        if adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING)
        {
            descriptor.required_features |= wgpu::Features::DUAL_SOURCE_BLENDING;
        }
        descriptor
    });

    let options = eframe::NativeOptions {
        viewport,
        renderer,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
            ..Default::default()
        },
        ..Default::default()
    };

    let cfg = std::sync::Arc::new(cfg);

    eframe::run_native(
        "Ember",
        options,
        Box::new(move |cc| {
            let cfg_clone = cfg.clone();
            // Set UI scale: use config value if provided, otherwise use native DPI
            let scale = cfg_clone
                .ui_scale
                .unwrap_or_else(|| cc.egui_ctx.native_pixels_per_point().unwrap_or(1.0));
            cc.egui_ctx.set_pixels_per_point(scale);
            // Font/UI zoom is owned by the configurable `font:*` commands and
            // Ctrl+wheel. egui's built-in keyboard zoom must be off: its
            // shortcut matching tolerates extra modifiers, so it would also
            // fire on Ctrl+Alt+=/- (the opacity chords) and rescale the UI.
            cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
            configure_fonts_and_gpu(&cc.egui_ctx, cc.wgpu_render_state.as_ref(), &cfg_clone);
            let initial_theme = theme::Theme::get_theme(&cfg_clone.theme).unwrap_or_default();
            apply_theme_visuals(&cc.egui_ctx, &initial_theme);

            TerminalApp::new(
                &cfg_clone,
                cc.egui_ctx.clone(),
                cc.wgpu_render_state.clone(),
            )
            .map(|app| Box::new(app) as Box<dyn eframe::App>)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        }),
    )
}

// TerminalApp struct definition moved to app::state module
use app::state::TerminalApp;

// normalize_terminal_shortcut_events moved to app::events module

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PasteOrigin {
    Clipboard,
    PromptInsert,
}

#[derive(Debug)]
enum PasteRequestError {
    Unsafe(crate::review_text::ReviewTextError),
    Write(PasteWriteError),
}

#[derive(Debug)]
enum PasteWriteError {
    /// An older session-scoped input edge/reply still owns delivery order.
    Busy,
    Write(crate::shell::ShellWriteError),
}

impl PasteWriteError {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::Busy) || matches!(self, Self::Write(error) if error.is_backpressure())
    }
}

impl std::fmt::Display for PasteWriteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => formatter.write_str("older terminal input is still pending"),
            Self::Write(error) => error.fmt(formatter),
        }
    }
}

fn ensure_direct_paste_route_available(
    direct_input_blocked: bool,
    pending_input: bool,
) -> Result<(), PasteWriteError> {
    if direct_input_blocked || pending_input {
        Err(PasteWriteError::Busy)
    } else {
        Ok(())
    }
}

impl std::fmt::Display for PasteRequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsafe(error) => write!(formatter, "unsafe prompt text: {error}"),
            Self::Write(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PasteRequestError {}

fn paste_requires_confirmation(
    origin: PasteOrigin,
    configured: bool,
    risk: &jterm_core::pty_input::PasteRisk,
    had_visual_spoofing: bool,
) -> bool {
    (origin == PasteOrigin::Clipboard && had_visual_spoofing)
        || (configured
            && jterm_core::pty_input::should_confirm(
                risk,
                crate::app::state::PASTE_CONFIRM_THRESHOLD_BYTES,
            ))
}

/// ember's side of the shared paste policy.
///
/// `SendVerbatim` is this app's long-standing behaviour: a multiline payload is
/// never truncated, because the confirmation modal — not silent mangling — is
/// what stands between the clipboard and the shell.
///
/// The pinned core's `prompt_insert` policy preserves control bytes, so it is
/// a trusted *post-validation* boundary, never an input classifier. The Files
/// sidebar reaches it only after `sanitize_prompt_payload` strips C0/C1 and
/// rejects hidden Unicode; Agent commands reach it only after strict proposal
/// validation. Clipboard/search bytes use the de-fanging policy as well.
fn paste_policy(submit_after_paste: bool) -> jterm_core::pty_input::PastePolicy {
    use jterm_core::pty_input::{PastePolicy, UnbracketedMultiline::SendVerbatim};
    if submit_after_paste {
        PastePolicy {
            submit: true,
            ..PastePolicy::prompt_insert(SendVerbatim)
        }
    } else {
        PastePolicy::clipboard(SendVerbatim)
    }
}

/// The payload exactly as the child will receive it, minus the framing: line
/// endings folded and every embedded ESC[200~/ESC[201~ removed, however many
/// passes that takes.
///
/// `bracketed: false` with `SendVerbatim` is the combination that neither frames
/// nor truncates, so this is normalization only. It is what the confirmation
/// modal previews and what gets framed on delivery.
///
/// The loop is the load-bearing part. `jterm_core`'s de-fanging deletes a marker
/// and resumes *after* it, so deleting an inner marker can splice the bytes
/// around it into a new one: `ESC [` + `ESC [ 2 0 1 ~` + `2 0 1 ~` becomes a
/// live `ESC[201~` after a single pass. Under a policy that keeps control bytes
/// (`prompt_insert`: the sidebar `cd`, command replay, an approved agent
/// command) that reconstituted terminator would then be framed and would close
/// the frame early — the very bracketed-paste injection this encoder exists to
/// prevent. Nesting can be arbitrarily deep, so run to a fixed point.
fn defanged_paste_body(text: &str, policy: jterm_core::pty_input::PastePolicy) -> String {
    use jterm_core::pty_input::{encode_paste, PasteModes};
    let unframed = PasteModes { bracketed: false };
    let mut body = encode_paste(text, unframed, policy).echo_text;
    // Every pass that changes anything deletes at least six bytes, so the
    // original length bounds the iteration count; the cap only guards against a
    // future encoder whose passes are not strictly shrinking.
    for _ in 0..body.len() {
        let next = encode_paste(&body, unframed, policy).echo_text;
        if next == body {
            break;
        }
        body = next;
    }
    body
}

/// [`defanged_paste_body`] under the policy ember uses for a clipboard paste
/// (or, when `submit_after_paste`, for a command this app composed itself).
fn normalized_paste_body(text: &str, submit_after_paste: bool) -> String {
    defanged_paste_body(text, paste_policy(submit_after_paste))
}

/// Bytes for a command this app submits on the user's behalf — currently the
/// Agent panel's approved suggestion.
///
/// Goes through the shared encoder for the same reason every other writer does:
/// the text is model output that a human only skimmed, so an `ESC[201~` in it
/// must not be able to close the frame and turn the remainder into typed
/// commands. The submitting CR lands outside the frame.
fn encode_submitted_command(command: &str, bracketed: bool) -> Vec<u8> {
    let policy = paste_policy(true);
    jterm_core::pty_input::encode_paste(
        &defanged_paste_body(command, policy),
        jterm_core::pty_input::PasteModes { bracketed },
        policy,
    )
    .bytes
}

/// Encode `text` for the child's *current* bracketed-paste mode and atomically
/// admit it to the shell writer. A caller-provided session ordering gate keeps
/// this direct delivery behind older mouse/protocol traffic.
///
/// DECSET 2004 is read here, at delivery time, and nowhere else. The
/// confirmation modal can stay open for arbitrarily many frames, and a shell
/// that enters or leaves bracketed-paste mode while it is up (finishing a
/// `vim`, starting one) would otherwise have its payload framed for the mode
/// that was advertised when the dialog opened: an unframed body pasted with
/// ESC[200~ in front of it lands as literal garbage, and a framed body sent
/// unframed executes every line.
fn write_paste_to_session(
    session: &mut Session,
    text: &str,
    submit_after_paste: bool,
    direct_input_blocked: bool,
) -> Result<bool, PasteWriteError> {
    // Do not pre-encode into pending_input: DECSET 2004 can change while an
    // OSC paste worker or an older user-input retry owns the route. Keeping
    // normalized source in the confirmation flow lets retry encode against
    // the mode that is actually live at delivery time.
    ensure_direct_paste_route_available(direct_input_blocked, !session.pending_input.is_empty())?;
    let bracketed = {
        let terminal = session.terminal.lock();
        terminal.is_bracketed_paste_enabled()
    };
    let policy = paste_policy(submit_after_paste);
    // De-fanged to a fixed point *before* framing, so nothing this function
    // frames can still contain a terminator. Callers hand us an already-stable
    // body, which makes this pass a no-op for them; doing it here anyway is what
    // keeps the guarantee a property of the writer rather than of its callers.
    let paste = jterm_core::pty_input::encode_paste(
        &defanged_paste_body(text, policy),
        jterm_core::pty_input::PasteModes { bracketed },
        policy,
    );
    if paste.is_empty() {
        return Ok(false);
    }
    session
        .shell
        .write(&paste.bytes)
        .map_err(PasteWriteError::Write)?;
    let mut terminal = session.terminal.lock();
    terminal.note_user_input(&paste.bytes);
    terminal.scroll_to_bottom();
    drop(terminal);
    session.projection_view_state.scroll_to_bottom();
    Ok(true)
}

fn paste_text_into_session(
    session: &mut Session,
    text: String,
    paste_confirm: bool,
    origin: PasteOrigin,
    submit_after_paste: bool,
    direct_input_blocked: bool,
    pending_paste_confirm: &mut Option<crate::app::state::PendingPasteConfirm>,
) -> Result<bool, PasteRequestError> {
    // Classify the clipboard as it arrived: `should_confirm` also trips on an
    // embedded paste marker, which the encoder below defuses but the user still
    // deserves to be told about.
    let risk = jterm_core::pty_input::classify_paste(&text);
    let (max_bytes, disposition) = match origin {
        PasteOrigin::Clipboard => (
            usize::MAX,
            crate::review_text::VisualSpoofDisposition::PreserveForConfirmation,
        ),
        PasteOrigin::PromptInsert => (
            crate::review_text::MAX_PROMPT_INSERT_BYTES,
            crate::review_text::VisualSpoofDisposition::Reject,
        ),
    };
    let sanitized = crate::review_text::sanitize_prompt_payload(&text, max_bytes, disposition)
        .map_err(PasteRequestError::Unsafe)?;
    let body = normalized_paste_body(&sanitized.text, submit_after_paste);
    if body.is_empty() {
        return Ok(false);
    }
    let session_id = session.metadata.session_id.clone();

    if paste_requires_confirmation(origin, paste_confirm, &risk, sanitized.had_visual_spoofing) {
        *pending_paste_confirm = Some(crate::app::state::PendingPasteConfirm {
            decision_armed: false,
            text: body,
            session_id,
            risk,
            submit_after_paste,
            had_visual_spoofing: sanitized.had_visual_spoofing,
        });
        return Ok(true);
    }

    // Retain the normalized source until the all-or-nothing shell enqueue
    // succeeds. On transient backpressure the confirmation flow becomes a
    // durable retry surface even when confirmations were otherwise off.
    match write_paste_to_session(session, &body, submit_after_paste, direct_input_blocked) {
        Ok(delivered) => Ok(delivered),
        Err(error) => {
            if error.is_retryable() {
                *pending_paste_confirm = Some(crate::app::state::PendingPasteConfirm {
                    decision_armed: false,
                    text: body,
                    session_id,
                    risk,
                    submit_after_paste,
                    had_visual_spoofing: sanitized.had_visual_spoofing,
                });
            }
            Err(PasteRequestError::Write(error))
        }
    }
}

fn osc_5522_packet(metadata: &str, payload: Option<&str>) -> Vec<u8> {
    let mut packet = Vec::new();
    packet.extend_from_slice(b"\x1b]5522;");
    packet.extend_from_slice(metadata.as_bytes());
    if let Some(payload) = payload {
        packet.extend_from_slice(b";");
        packet.extend_from_slice(payload.as_bytes());
    }
    packet.extend_from_slice(b"\x1b\\");
    packet
}

const OSC_5522_DATA_CHUNK_BYTES: usize = 4096;
const MAX_OSC_5522_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

struct ClipboardRequestGuard(Arc<std::sync::atomic::AtomicBool>);

impl Drop for ClipboardRequestGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn enqueue_terminal_protocol_response(
    response_tx: &ProtocolResponseSender,
    terminal: &Arc<ParkingMutex<TerminalState>>,
    response: Vec<u8>,
    context: &str,
) {
    if let Err((error, mut response)) = response_tx.try_enqueue_critical(response) {
        if error == ProtocolResponseQueueError::Full {
            // The parser pump stops while the per-session queue is non-empty,
            // so this bounded batch is retried before any newer PTY requests
            // are accepted. Prepend to preserve response order.
            let mut terminal = terminal.lock();
            response.append(&mut terminal.output_buffer);
            terminal.output_buffer = response;
        } else {
            log::debug!("{context} cancelled: {error}");
        }
    }
}

/// Worker responses wait for space in the bounded per-session protocol FIFO,
/// so transient shell-writer pressure cannot lose an accepted query. The
/// clipboard reader is globally single-flight and closing a session wakes the
/// waiter. Only a permanently oversized response is replaced with the small
/// protocol-specific failure response.
fn enqueue_worker_protocol_response(
    response_tx: &ProtocolResponseSender,
    response: Vec<u8>,
    fallback: Vec<u8>,
    context: &str,
) {
    match response_tx.enqueue_blocking(response) {
        Ok(()) => {}
        Err(ProtocolResponseQueueError::Closed) => {
            log::debug!("{context} target session closed before its response was queued");
        }
        Err(error) => {
            log::warn!("{context} response replaced by bounded fallback: {error}");
            if let Err(fallback_error) = response_tx.enqueue_blocking(fallback) {
                log::debug!("{context} fallback was cancelled: {fallback_error}");
            }
        }
    }
}

fn send_osc5522_worker_response(response_tx: &ProtocolResponseSender, response: Vec<u8>) {
    enqueue_worker_protocol_response(
        response_tx,
        response,
        osc_5522_packet("type=read:status=EBUSY", None),
        "OSC 5522",
    );
}

fn service_osc5522_clipboard_requests(
    clipboard_available: bool,
    in_flight: &Arc<AtomicBool>,
    terminal: Arc<ParkingMutex<TerminalState>>,
    response_tx: ProtocolResponseSender,
    mut requests: Vec<terminal::ClipboardReadRequest>,
) {
    if requests.is_empty() {
        return;
    }
    if !clipboard_available {
        for _ in &requests {
            enqueue_terminal_protocol_response(
                &response_tx,
                &terminal,
                osc_5522_packet("type=read:status=ENOSYS", None),
                "OSC 5522 ENOSYS response",
            );
        }
        return;
    }
    if in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        for _ in &requests {
            enqueue_terminal_protocol_response(
                &response_tx,
                &terminal,
                osc_5522_packet("type=read:status=EBUSY", None),
                "OSC 5522 busy response",
            );
        }
        return;
    }

    let in_flight_for_thread = Arc::clone(in_flight);
    let error_tx = response_tx.clone();
    let error_terminal = Arc::clone(&terminal);
    let request_count = requests.len();
    let spawn_result = std::thread::Builder::new()
        .name("clipboard-request-handler".to_string())
        .spawn(move || {
            let _guard = ClipboardRequestGuard(in_flight_for_thread);
            let Ok(clipboard) = ClipboardManager::new() else {
                crate::debug_log!("[OSC5522] Failed to create clipboard manager");
                for _ in 0..request_count {
                    send_osc5522_worker_response(
                        &response_tx,
                        osc_5522_packet("type=read:status=ENOSYS", None),
                    );
                }
                return;
            };

            for request in requests.drain(..) {
                let terminal::ClipboardReadKind::MimeData(mime_type) = request.kind;
                let data = clipboard.read_mime(&mime_type).unwrap_or_default();
                let response = if data.is_empty() {
                    osc_5522_packet("type=read:status=ENOSYS", None)
                } else {
                    clipboard_5522_response_for_mime(&mime_type, &data)
                };
                crate::debug_log!(
                    "[OSC5522] responding to mime request mime={} bytes={}",
                    mime_type,
                    data.len()
                );
                send_osc5522_worker_response(&response_tx, response);
            }
        });
    if let Err(error) = spawn_result {
        in_flight.store(false, Ordering::Release);
        log::warn!("failed to spawn OSC 5522 clipboard handler: {error}");
        for _ in 0..request_count {
            enqueue_terminal_protocol_response(
                &error_tx,
                &error_terminal,
                osc_5522_packet("type=read:status=ENOSYS", None),
                "OSC 5522 spawn-error response",
            );
        }
    }
}

fn clipboard_5522_response_for_mime(mime_type: &str, data: &[u8]) -> Vec<u8> {
    clipboard_5522_response_for_mime_with_limit(mime_type, data, MAX_OSC_5522_RESPONSE_BYTES)
}

fn clipboard_5522_response_for_mime_with_limit(
    mime_type: &str,
    data: &[u8],
    max_bytes: usize,
) -> Vec<u8> {
    if data.len() > max_bytes {
        // OSC 5522 has no dedicated "too large" read error. EPERM is the
        // closest truthful response: terminal policy denied this read.
        return osc_5522_packet("type=read:status=EPERM", None);
    }

    let encoded_mime = base64::engine::general_purpose::STANDARD.encode(mime_type.as_bytes());
    let mut output = Vec::new();
    output.extend_from_slice(&osc_5522_packet("type=read:status=OK", None));
    for chunk in data.chunks(OSC_5522_DATA_CHUNK_BYTES) {
        let encoded_data = base64::engine::general_purpose::STANDARD.encode(chunk);
        output.extend_from_slice(&osc_5522_packet(
            &format!("type=read:status=DATA:mime={}", encoded_mime),
            Some(&encoded_data),
        ));
    }
    output.extend_from_slice(&osc_5522_packet("type=read:status=DONE", None));
    output
}

fn link_at_pointer(
    links: &[link::Link],
    pointer: egui::Pos2,
    content_rect: egui::Rect,
    char_width: f32,
    line_height: f32,
    viewport: &terminal::ProjectedViewport,
) -> Option<link::Link> {
    let cols = viewport.columns();
    let rows = viewport.rows();
    if !content_rect.contains(pointer) || cols == 0 || rows == 0 {
        return None;
    }
    let (row, col) =
        grid_position_from_content(pointer, content_rect, char_width, line_height, cols, rows);
    if viewport.is_transformed()
        && viewport
            .raw_anchor_at(terminal::DisplayPoint::new(row, col))
            .is_none()
    {
        return None;
    }
    links
        .iter()
        .find(|link| link.line == row && col >= link.col_start && col < link.col_end)
        .cloned()
}

fn projected_viewport_for_session(
    session: &mut Session,
    renderer: &TerminalRenderer,
) -> terminal::ProjectedViewport {
    let terminal = Arc::clone(&session.terminal);
    let policy = &session.projection_policy;
    let view_state = &mut session.projection_view_state;
    let mut terminal = terminal.lock();
    renderer.projected_viewport_with_state(&mut terminal, policy, view_state)
}

fn application_cell_at_pointer(
    pointer: egui::Pos2,
    content_rect: egui::Rect,
    char_width: f32,
    line_height: f32,
    viewport: &terminal::ProjectedViewport,
) -> Option<(usize, usize)> {
    let cols = viewport.columns();
    let rows = viewport.rows();
    if cols == 0 || rows == 0 {
        return None;
    }
    let (row, col) =
        grid_position_from_content(pointer, content_rect, char_width, line_height, cols, rows);
    viewport.application_cell(terminal::DisplayPoint::new(row, col))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AppMouseFrameRoute {
    /// A pointer cell that may produce press, motion or wheel reports.
    lossy_cell: Option<(usize, usize)>,
    /// The last real raw cell is retained solely to pair an accepted press with
    /// its release when the pointer leaves the app surface or its session.
    release_cell: Option<(usize, usize)>,
}

fn app_mouse_press_reports_from_snapshot(
    route_requested: bool,
    application_cell: Option<(usize, usize)>,
) -> bool {
    route_requested && application_cell.is_some()
}

fn app_mouse_frame_route(
    uses_active_projection: bool,
    projected_pointer_cell: Option<Option<(usize, usize)>>,
    captured_last_cell: Option<(usize, usize)>,
) -> AppMouseFrameRoute {
    if !uses_active_projection {
        return AppMouseFrameRoute {
            lossy_cell: None,
            release_cell: captured_last_cell,
        };
    }

    let lossy_cell = projected_pointer_cell.flatten();
    AppMouseFrameRoute {
        lossy_cell,
        release_cell: lossy_cell.or(captured_last_cell),
    }
}

const LINK_ACTIVATION_DRAG_THRESHOLD: f32 = 4.0;

fn link_activation_dragged(origin: egui::Pos2, current: egui::Pos2) -> bool {
    current.distance(origin) > LINK_ACTIVATION_DRAG_THRESHOLD
}

fn link_activation_release_allowed(
    pressed_session: &str,
    active_session: &str,
    cancelled: bool,
    multiple_click: bool,
) -> bool {
    pressed_session == active_session && !cancelled && !multiple_click
}

fn link_activation_ready(released_at: Option<f64>, now: f64, double_click_delay: f64) -> bool {
    released_at.is_some_and(|released_at| {
        now.is_finite()
            && released_at.is_finite()
            && now - released_at >= double_click_delay.max(0.0)
    })
}

fn mouse_press_reports_to_app(
    mouse_enabled: bool,
    shift_bypass: bool,
    app_surface: bool,
    host_link_override: bool,
) -> bool {
    mouse_enabled && !shift_bypass && app_surface && !host_link_override
}

fn mouse_capture_accepts_new_press(capture_active: bool) -> bool {
    !capture_active
}

fn mouse_cell_for_current_dimensions(
    pointer: Option<egui::Pos2>,
    fallback: Option<(usize, usize)>,
    content_rect: egui::Rect,
    char_width: f32,
    line_height: f32,
    cols: usize,
    rows: usize,
) -> (usize, usize) {
    pointer
        .map(|pos| {
            grid_position_from_content(pos, content_rect, char_width, line_height, cols, rows)
        })
        .or_else(|| {
            fallback.map(|(row, col)| {
                (
                    row.min(rows.saturating_sub(1)),
                    col.min(cols.saturating_sub(1)),
                )
            })
        })
        .unwrap_or((0, 0))
}

#[derive(Debug)]
pub(crate) struct DesktopNotification {
    title: String,
    body: String,
}

const DESKTOP_NOTIFICATION_QUEUE_CAPACITY: usize = 8;

fn desktop_notification_channel() -> (
    crossbeam_channel::Sender<DesktopNotification>,
    crossbeam_channel::Receiver<DesktopNotification>,
) {
    crossbeam_channel::bounded(DESKTOP_NOTIFICATION_QUEUE_CAPACITY)
}

fn spawn_desktop_notification_worker() -> Option<crossbeam_channel::Sender<DesktopNotification>> {
    let (tx, rx) = desktop_notification_channel();
    std::thread::Builder::new()
        .name("desktop-notification-worker".to_string())
        .spawn(move || {
            while let Ok(notification) = rx.recv() {
                // `helper::notify_send` resolves a trusted absolute binary and
                // owns the helper's process group under one deadline, so a
                // stuck D-Bus bridge cannot strand the worker.
                if let Err(error) =
                    jterm_core::helper::notify_send(&notification.title, &notification.body)
                {
                    log::debug!("desktop notification unavailable: {error}");
                }
            }
        })
        .ok()?;
    Some(tx)
}

/// Shared rate window for every desktop-notification source (OSC 9/777 and
/// long-command toasts), so their combined output stays bounded.
const NOTIFICATION_RATE_WINDOW: Duration = Duration::from_secs(5);
const MAX_NOTIFICATIONS_PER_WINDOW: usize = 4;

/// Restart the shared notification rate window once it has elapsed. Callers
/// then check `notifications_in_window` against [`MAX_NOTIFICATIONS_PER_WINDOW`]
/// and count their own successful sends.
fn roll_notification_rate_window(
    window_started: &mut std::time::Instant,
    notifications_in_window: &mut usize,
) {
    let now = std::time::Instant::now();
    if now.duration_since(*window_started) >= NOTIFICATION_RATE_WINDOW {
        *window_started = now;
        *notifications_in_window = 0;
    }
}

fn show_desktop_notification(
    notification_tx: Option<&crossbeam_channel::Sender<DesktopNotification>>,
    window_started: &mut std::time::Instant,
    notifications_in_window: &mut usize,
    title: String,
    body: String,
) {
    roll_notification_rate_window(window_started, notifications_in_window);
    if *notifications_in_window >= MAX_NOTIFICATIONS_PER_WINDOW {
        return;
    }
    let Some(notification_tx) = notification_tx else {
        return;
    };
    if notification_tx
        .try_send(DesktopNotification { title, body })
        .is_ok()
    {
        *notifications_in_window += 1;
    }
}

/// anvil-parity gate for the long-command desktop toast
/// (`block_view`'s `notify_long_blocks` check): the command must be a real
/// foreground command (anvil skips background blocks, whose command line is
/// empty), the config flag must be on, and the measured duration must reach
/// the threshold. ember adds the egui window focus state: a completion the
/// user just watched on screen needs no toast.
fn should_notify_long_command(
    config: &config::Config,
    command: Option<&str>,
    duration_ms: Option<u64>,
    watched: bool,
) -> bool {
    if !config.notify_long_blocks || watched {
        return false;
    }
    if command.map(str::trim).is_none_or(str::is_empty) {
        return false;
    }
    duration_ms.is_some_and(|ms| ms >= config.notify_long_block_threshold_ms)
}

/// Post the long-command-finished notification when the gates above and the
/// shared rate window allow it. Free function over disjoint fields because the
/// completion sites hold a mutable borrow of the session manager.
fn maybe_notify_long_command(
    config: &config::Config,
    window_started: &mut std::time::Instant,
    notifications_in_window: &mut usize,
    completed: &crate::terminal::CompletedCommandEvent,
    watched: bool,
) {
    if !completed.is_trusted_completion() {
        return;
    }
    let Some(exit_code) = completed.exit_code else {
        // The notification API requires a concrete status. A bare `D` is an
        // observed end with Unknown outcome, not permission to report success.
        return;
    };
    if !should_notify_long_command(
        config,
        completed.command.as_deref(),
        completed.duration_ms,
        watched,
    ) {
        return;
    }
    // An untrusted PTY can claim any `duration=` in OSC 133;D, so this path
    // shares the OSC 9/777 rate window instead of trusting the threshold to
    // keep notify-send spawns rare.
    roll_notification_rate_window(window_started, notifications_in_window);
    if *notifications_in_window >= MAX_NOTIFICATIONS_PER_WINDOW {
        return;
    }
    *notifications_in_window += 1;
    let command = completed.command.as_deref().unwrap_or_default().trim();
    jterm_core::notify::long_block_finished(command, exit_code, completed.duration_ms.unwrap_or(0));
}

/// 把每条 shell 上报的已完成命令追加到家族共享的 JSONL 历史索引（与
/// anvil/forge/frost 同键同格式），供 Ctrl+Shift+H 选择器跨重启召回。
/// 写入走 jterm_core 的有界后台写入器；多行 heredoc 等不安全的重建文本
/// 直接跳过而不是 noisy 报错。只记录命令行/cwd/exit code/结束时间——
/// 绝不记录输出。与 [`maybe_notify_long_command`] 同为自由函数，因为
/// 两个完成路径都持有 session manager 的可变借用。
fn record_command_history(
    config: &config::Config,
    completed: &crate::terminal::CompletedCommandEvent,
) {
    if completed.completion_provenance != crate::block_mode::CompletionProvenance::ShellReported {
        // 边界推断的结束只用于释放本地 UI/Agent 生命周期；持久化它会把
        // 缺失的 OSC 证据变成跨会话的伪完成记录。
        return;
    }
    let Some(path) = config.resolved_command_history_path() else {
        return;
    };
    let Some(command) = completed
        .command
        .as_deref()
        .and_then(history_picker::sanitized_command)
    else {
        return;
    };
    let Some(exit_code) = completed.exit_code else {
        // 共享历史 schema 要求确切的状态。缺失 OSC 状态是 Unknown，绝不
        // 当成隐式成功。
        return;
    };
    let cwd = completed
        .cwd
        .as_deref()
        .filter(|cwd| history_picker::sanitized_cwd(cwd).is_some());
    if let Err(error) = jterm_core::command_history::prepare_path(&path, true) {
        log::warn!("unsafe command-history path rejected: {error}");
        return;
    }
    let end_time_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok());
    if let Err(error) = jterm_core::command_history::enqueue(
        &path,
        config.command_history_max_entries as usize,
        command,
        cwd,
        exit_code,
        end_time_ms,
    ) {
        log::warn!("command history: {error}");
    }
}

fn reported_capture_button(capture: Option<(bool, u8)>) -> Option<u8> {
    capture.and_then(|(reported_to_app, button)| reported_to_app.then_some(button))
}

const MAX_MOUSE_WHEEL_REPORTS_PER_FRAME: isize = 64;

fn bounded_wheel_step_accumulate(current: isize, delta: f32, multiplier: usize) -> isize {
    let multiplier = isize::try_from(multiplier).unwrap_or(isize::MAX);
    current
        .saturating_add((delta.round() as isize).saturating_mul(multiplier))
        .clamp(
            -MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
            MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
        )
}

fn captured_release_button(
    capture: Option<(bool, u8)>,
    released: &[u8],
    pointer_any_down: bool,
) -> Option<u8> {
    reported_capture_button(capture).filter(|button| released.contains(button) || !pointer_any_down)
}

fn queue_mouse_control(
    queue: &mut std::collections::VecDeque<crate::app::state::PendingMouseControl>,
    kind: crate::app::state::PendingMouseControlKind,
    bytes: Vec<u8>,
) {
    if !queue.iter().any(|pending| pending.kind == kind) {
        queue.push_back(crate::app::state::PendingMouseControl { kind, bytes });
    }
}

fn flush_pending_mouse_controls<E>(
    queue: &mut std::collections::VecDeque<crate::app::state::PendingMouseControl>,
    press_accepted: &mut bool,
    mut send: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    while let Some(pending) = queue.front() {
        send(&pending.bytes)?;
        let accepted = queue.pop_front().expect("front existed above");
        if accepted.kind == crate::app::state::PendingMouseControlKind::Press {
            *press_accepted = true;
        }
    }
    Ok(())
}

fn flush_mouse_controls(
    capture: &mut crate::app::state::TerminalMouseCapture,
    protocol_input_barriers: &crate::session_manager::SessionInputBarriers,
) -> Result<bool, crate::shell::ShellWriteError> {
    if mouse_protocol_input_is_blocked(
        &capture.session_id,
        protocol_input_barriers,
        &capture.protocol_responses,
    ) {
        return Ok(false);
    }
    let write_tx = capture.write_tx.clone();
    flush_pending_mouse_controls(
        &mut capture.pending_controls,
        &mut capture.press_accepted,
        |bytes| write_tx.try_send(bytes.to_vec()),
    )?;
    Ok(true)
}

fn mouse_protocol_input_is_blocked(
    session_id: &str,
    protocol_input_barriers: &crate::session_manager::SessionInputBarriers,
    protocol_responses: &crate::session_manager::ProtocolResponseSender,
) -> bool {
    crate::session_manager::user_input_flush_block(
        session_id,
        None,
        protocol_input_barriers,
        protocol_responses,
    )
    .is_some()
}

fn mouse_lossy_reports_allowed(protocol_input_blocked: bool, sequence_allows: bool) -> bool {
    !protocol_input_blocked && sequence_allows
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrimaryCopyRoute {
    None,
    CapturedLocal,
    Generic,
    SuppressCaptured,
}

fn primary_copy_route(
    capture: Option<(bool, bool, u8)>,
    capture_finished: bool,
    primary_released: bool,
) -> PrimaryCopyRoute {
    match capture {
        Some((reported_to_app, local_selection_cancelled, 0)) if capture_finished => {
            if !reported_to_app && !local_selection_cancelled {
                PrimaryCopyRoute::CapturedLocal
            } else {
                PrimaryCopyRoute::SuppressCaptured
            }
        }
        _ if primary_released => PrimaryCopyRoute::Generic,
        _ => PrimaryCopyRoute::None,
    }
}

fn take_tagged_cursor_move(
    target: &mut Option<usize>,
    bytes: &mut Vec<u8>,
) -> Option<(usize, Vec<u8>)> {
    let target = target.take();
    let bytes = std::mem::take(bytes);
    if bytes.is_empty() {
        None
    } else {
        target.map(|target| (target, bytes))
    }
}

fn mouse_capture_allows_lossy(capture: Option<&crate::app::state::TerminalMouseCapture>) -> bool {
    capture.is_none_or(|capture| {
        mouse_sequence_allows_lossy(
            capture.reported_to_app,
            capture.press_accepted,
            capture.release_observed,
            capture.pending_controls.is_empty(),
        )
    })
}

fn mouse_capture_is_complete(capture: &crate::app::state::TerminalMouseCapture) -> bool {
    mouse_sequence_is_complete(
        capture.reported_to_app,
        capture.press_accepted,
        capture.release_observed,
        capture.pending_controls.is_empty(),
    )
}

fn mouse_sequence_allows_lossy(
    reported_to_app: bool,
    press_accepted: bool,
    release_observed: bool,
    controls_empty: bool,
) -> bool {
    !reported_to_app || (press_accepted && !release_observed && controls_empty)
}

fn mouse_sequence_is_complete(
    reported_to_app: bool,
    press_accepted: bool,
    release_observed: bool,
    controls_empty: bool,
) -> bool {
    release_observed && (!reported_to_app || (press_accepted && controls_empty))
}

fn spawn_osc52_clipboard_writer(
    clipboard_available: bool,
) -> Option<crossbeam_channel::Sender<String>> {
    if !clipboard_available {
        return None;
    }
    let (tx, rx) = crossbeam_channel::bounded::<String>(1);
    std::thread::Builder::new()
        .name("osc52-clipboard-writer".to_string())
        .spawn(move || {
            let Ok(clipboard) = ClipboardManager::new() else {
                return;
            };
            while let Ok(text) = rx.recv() {
                if let Err(error) = clipboard.copy(&text) {
                    log::warn!("OSC 52 clipboard write failed: {error}");
                }
            }
        })
        .ok()?;
    Some(tx)
}

fn enqueue_osc52_clipboard_write(
    tx: Option<&crossbeam_channel::Sender<String>>,
    window_started: &mut std::time::Instant,
    writes_in_window: &mut usize,
    text: String,
) {
    const WINDOW: Duration = Duration::from_secs(1);
    const MAX_WRITES_PER_WINDOW: usize = 2;

    let now = std::time::Instant::now();
    if now.duration_since(*window_started) >= WINDOW {
        *window_started = now;
        *writes_in_window = 0;
    }
    let Some(tx) = tx else {
        return;
    };
    if *writes_in_window >= MAX_WRITES_PER_WINDOW {
        return;
    }
    if tx.try_send(text).is_ok() {
        *writes_in_window += 1;
    }
}

const MAX_OSC52_CLIPBOARD_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const OSC52_READ_RATE_WINDOW: Duration = Duration::from_secs(1);
const MAX_OSC52_READS_PER_WINDOW: usize = 2;

fn osc52_clipboard_response_with_limit(content: &str, max_response_bytes: usize) -> Vec<u8> {
    const PREFIX: &[u8] = b"\x1b]52;c;";
    const TERMINATOR: &[u8] = b"\x1b\\";
    let overhead = PREFIX.len() + TERMINATOR.len();
    if max_response_bytes < overhead {
        return Vec::new();
    }

    let encoded_len = content
        .len()
        .checked_add(2)
        .map(|length| length / 3)
        .and_then(|length| length.checked_mul(4));
    let content = if encoded_len.is_some_and(|length| length <= max_response_bytes - overhead) {
        content
    } else {
        // OSC 52 has no standardized error response. Replying with an empty
        // selection is bounded and lets a querying application stop waiting.
        ""
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(content.as_bytes());
    let mut response = Vec::with_capacity(overhead + encoded.len());
    response.extend_from_slice(PREFIX);
    response.extend_from_slice(encoded.as_bytes());
    response.extend_from_slice(TERMINATOR);
    response
}

fn osc52_read_rate_limit_allows(
    now: std::time::Instant,
    window_started: &mut std::time::Instant,
    reads_in_window: &mut usize,
) -> bool {
    if now.duration_since(*window_started) >= OSC52_READ_RATE_WINDOW {
        *window_started = now;
        *reads_in_window = 0;
    }
    if *reads_in_window >= MAX_OSC52_READS_PER_WINDOW {
        return false;
    }
    *reads_in_window += 1;
    true
}

fn service_osc52_clipboard_query(
    clipboard_available: bool,
    in_flight: &Arc<AtomicBool>,
    terminal: Arc<ParkingMutex<TerminalState>>,
    response_tx: ProtocolResponseSender,
    window_started: &mut std::time::Instant,
    reads_in_window: &mut usize,
) {
    let empty_response =
        || osc52_clipboard_response_with_limit("", MAX_OSC52_CLIPBOARD_RESPONSE_BYTES);
    if !osc52_read_rate_limit_allows(std::time::Instant::now(), window_started, reads_in_window)
        || !clipboard_available
    {
        enqueue_terminal_protocol_response(
            &response_tx,
            &terminal,
            empty_response(),
            "OSC 52 empty response",
        );
        return;
    }
    if in_flight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        enqueue_terminal_protocol_response(
            &response_tx,
            &terminal,
            empty_response(),
            "OSC 52 busy response",
        );
        return;
    }

    let in_flight_for_thread = Arc::clone(in_flight);
    let error_tx = response_tx.clone();
    let spawn_result = std::thread::Builder::new()
        .name("osc52-clipboard-reader".to_string())
        .spawn(move || {
            let _guard = ClipboardRequestGuard(in_flight_for_thread);
            let content = ClipboardManager::new()
                .and_then(|clipboard| clipboard.paste())
                .unwrap_or_default();
            let response =
                osc52_clipboard_response_with_limit(&content, MAX_OSC52_CLIPBOARD_RESPONSE_BYTES);
            let fallback =
                osc52_clipboard_response_with_limit("", MAX_OSC52_CLIPBOARD_RESPONSE_BYTES);
            enqueue_worker_protocol_response(&response_tx, response, fallback, "OSC 52");
        });
    if let Err(error) = spawn_result {
        in_flight.store(false, Ordering::Release);
        log::warn!("failed to spawn OSC 52 clipboard reader: {error}");
        enqueue_terminal_protocol_response(
            &error_tx,
            &terminal,
            empty_response(),
            "OSC 52 spawn-error response",
        );
    }
}

// key_to_string and build_keybinding_string moved to app::events module

/// 文件树右键菜单收集到的动作（闭包结束后统一执行，见 draw_tree_node）。
#[derive(Clone, Debug)]
enum FsMenuAction {
    /// 目标父目录。
    NewFile(std::path::PathBuf),
    NewFolder(std::path::PathBuf),
    /// 被重命名的源路径。
    Rename(std::path::PathBuf),
    /// 批量删除目标（单选时长度为 1）。
    Delete {
        paths: Vec<(std::path::PathBuf, bool)>,
    },
    Copy {
        paths: Vec<(std::path::PathBuf, bool)>,
    },
    Cut {
        paths: Vec<(std::path::PathBuf, bool)>,
    },
    /// 粘贴目标目录。
    Paste(std::path::PathBuf),
    /// 把条目完整路径复制到系统剪贴板（多选时换行连接；远程行是纯路径）。
    CopyPath(Vec<std::path::PathBuf>),
    /// 重新扫描该目录（若已加载）。
    Refresh(std::path::PathBuf),
}

/// 点击时的选择模式（修饰键语义）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FsSelectMode {
    /// 单选（无修饰键）。
    Single,
    /// ctrl+点击：切换该行。
    Toggle,
    /// shift+点击：锚点到该行的可见范围。
    Range,
}

/// 粘贴菜单项的状态：空剪贴板禁用；同位置直接粘贴（copy/rename 探针）；
/// 跨位置走流式传输（下载/上传/中转）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FsPasteState {
    Empty,
    /// 同位置。
    Ready,
    /// 远程 → 本地。
    Download,
    /// 本地 → 远程。
    Upload,
    /// 远程 i → 远程 j（经本地临时文件中转）。
    Relay,
}

fn snapshot_age_label(age: std::time::Duration) -> String {
    let seconds = age.as_secs();
    match seconds {
        0..=1 => "刚刚".to_string(),
        2..=59 => format!("{seconds} 秒前"),
        60..=3_599 => format!("{} 分钟前", seconds / 60),
        3_600..=86_399 => format!("{} 小时前", seconds / 3_600),
        _ => format!("{} 天前", seconds / 86_400),
    }
}

/// 文件树名称输入对话框（New File / New Folder / Rename 共用）。
#[derive(Clone, Debug)]
struct FsNameDialog {
    kind: FsNameDialogKind,
    /// New*：目标父目录；Rename：被重命名的源路径。
    base: std::path::PathBuf,
    input: String,
    error: Option<String>,
    /// Exact Files root/location that produced this delayed intent.
    context: sidebar::FilesIntentContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FsNameDialogKind {
    NewFile,
    NewFolder,
    Rename,
}

/// 文件树删除确认对话框（支持多选批量）。
#[derive(Clone, Debug)]
struct FsDeleteDialog {
    paths: Vec<std::path::PathBuf>,
    /// 其中目录的数量（递归删除警告）。
    dir_count: usize,
    /// Exact Files root/location that produced this delayed intent.
    context: sidebar::FilesIntentContext,
}

impl TerminalApp {
    /// Route click-to-cursor arrows produced by the previous render before
    /// collecting any input from this frame. The terminal pointer is a stable
    /// identity tag while the session owns its `Arc`; stale/unroutable bytes
    /// are discarded instead of being delivered to a replacement PTY.
    fn stage_prior_cursor_moves(&mut self, terminal_input_blocked: bool) -> (bool, bool) {
        let mut tagged_moves = Vec::with_capacity(self.pane_renderers.len() + 1);
        if let Some(tagged) = take_tagged_cursor_move(
            &mut self.renderer.cursor_move_terminal_ptr,
            &mut self.renderer.cursor_move_input,
        ) {
            tagged_moves.push(tagged);
        }
        for renderer in &mut self.pane_renderers {
            if let Some(tagged) = take_tagged_cursor_move(
                &mut renderer.cursor_move_terminal_ptr,
                &mut renderer.cursor_move_input,
            ) {
                tagged_moves.push(tagged);
            }
        }

        if terminal_input_blocked {
            return (false, false);
        }

        let mut had_input = false;
        let mut overflow = false;
        for (target, bytes) in tagged_moves {
            // `data_ptr`, not `Arc::as_ptr`: the tag is taken at render time
            // from the `&mut TerminalState` the renderer holds, which is the
            // mutex's *contents*. Comparing it against the address of the
            // `Mutex` itself never matched, and every click-to-cursor move was
            // silently dropped here.
            let routed = self
                .session_manager
                .sessions_mut()
                .iter_mut()
                .find(|session| {
                    session.terminal.data_ptr() as usize == target
                        && session.purpose != crate::session::SessionPurpose::RetainedCommand
                });
            if let Some(session) = routed {
                had_input = true;
                overflow |= !session.queue_input(&bytes);
            }
        }
        (had_input, overflow)
    }

    fn new(
        cfg: &config::Config,
        repaint_ctx: egui::Context,
        wgpu_render_state: Option<egui_wgpu::RenderState>,
    ) -> std::result::Result<Self, String> {
        let (cols, rows) = clamp_terminal_dimensions(cfg.cols, cfg.rows);
        crate::debug_log!(
            "[INIT] terminal dimensions cfg=({}, {}) clamped=({}, {})",
            cfg.cols,
            cfg.rows,
            cols,
            rows
        );

        // 尝试获取实例锁，成功表示没有其他实例在运行
        let lock_file = session_persistence::try_acquire_instance_lock();
        let is_first_instance = lock_file.is_some();

        let mut session_restore_notice = None;
        let mut session_persistence_blocked = false;

        // 仅在首个实例且配置允许时恢复会话。损坏文件先移到旁路备份；
        // 若备份失败则禁止本进程保存，绝不让新建的单会话覆盖原始证据。
        let saved_snapshot = if cfg.restore_session && is_first_instance {
            match cfg.resolved_session_history_path() {
                Ok(path) => {
                    match session_persistence::SessionsSnapshot::load_with_warnings(&path) {
                        Ok((snapshot, warnings)) => {
                            if !warnings.is_empty() {
                                for warning in &warnings {
                                    eprintln!("[SessionPersistence] WARNING: {warning}");
                                }
                                session_restore_notice = Some(format!(
                                    "Session restore adjusted {} unsafe value(s)",
                                    warnings.len()
                                ));
                            }
                            (!snapshot.sessions.is_empty()).then_some(snapshot)
                        }
                        Err(error) => {
                            eprintln!(
                                "[SessionPersistence] Failed to load {}: {error}",
                                path.display()
                            );
                            // ember donated this idea to the family; core's copy
                            // is a strict superset (a *dangling* symlink at a
                            // backup name no longer counts as free), so there is
                            // one scheme, not two.
                            match jterm_core::snapshot_file::quarantine_corrupt(&path) {
                                Ok(backup) => {
                                    session_restore_notice = Some(format!(
                                        "Session restore failed; original moved to {}",
                                        backup.display()
                                    ));
                                }
                                Err(backup_error) => {
                                    session_persistence_blocked = true;
                                    session_restore_notice = Some(format!(
                                    "Session restore failed and was not overwritten: {backup_error}"
                                ));
                                }
                            }
                            None
                        }
                    }
                }
                Err(error) => {
                    session_persistence_blocked = true;
                    session_restore_notice =
                        Some(format!("Session persistence is unavailable: {error}"));
                    None
                }
            }
        } else {
            if !is_first_instance {
                eprintln!("[SessionPersistence] Another instance is running, starting fresh");
            }
            None
        };

        // 创建首个会话，使用保存的 cwd 和 session_id（如果有）
        let first_cwd = saved_snapshot
            .as_ref()
            .and_then(|s| s.sessions.first()?.cwd.as_deref().map(String::from));
        let first_session_id = saved_snapshot
            .as_ref()
            .and_then(|s| s.sessions.first()?.session_id.as_deref().map(String::from))
            .filter(|id| session::is_valid_jsh_session_id(id))
            .unwrap_or_else(session::generate_session_id);
        let saved_active_index = saved_snapshot.as_ref().and_then(|s| s.active_index);
        let saved_tab_layouts: Vec<session_persistence::LayoutSnapshot> = saved_snapshot
            .as_ref()
            .map(|snapshot| snapshot.tabs.clone())
            .unwrap_or_default();
        let saved_active_tab = saved_snapshot.as_ref().and_then(|s| s.active_tab);
        let terminal = TerminalState::new(cols, rows);

        let configured_shell = std::env::var("EMBER_SHELL").ok().or(cfg.shell.clone());

        let shell = match ShellSession::new_with_cwd(
            cols,
            rows,
            first_cwd.as_deref(),
            Some(&first_session_id),
            configured_shell.as_deref(),
            None,
            repaint_ctx.clone(),
        ) {
            Ok(session) => {
                eprintln!("✓ Shell session started successfully");
                session
            }
            Err(e) => {
                eprintln!(
                    "✗ Failed to start shell with saved cwd, falling back: {}",
                    e
                );
                match ShellSession::new_with_cwd(
                    cols,
                    rows,
                    None,
                    Some(&first_session_id),
                    configured_shell.as_deref(),
                    None,
                    repaint_ctx.clone(),
                ) {
                    Ok(session) => session,
                    Err(e2) => {
                        return Err(format!(
                            "Cannot create shell session: {} (after fallback from: {})",
                            e2, e
                        ));
                    }
                }
            }
        };

        let session = Session::with_default_name_and_session_id(
            0,
            Arc::new(ParkingMutex::new(terminal)),
            shell,
            first_session_id,
        );
        let mut session_manager = SessionManager::new(session, repaint_ctx, configured_shell);

        // 恢复额外的会话（包括 restorable commands 回放）
        if let Some(snap) = saved_snapshot {
            session_manager.restore_from_snapshots(snap.sessions, saved_active_index);
            eprintln!(
                "[SessionPersistence] Restored {} sessions",
                session_manager.len()
            );
        }

        for session in session_manager.sessions_mut() {
            let mut terminal = session.terminal.lock();
            terminal.set_max_scrollback(cfg.scrollback_lines);
        }

        let clipboard = ClipboardManager::new().ok();
        let osc52_clipboard_write_tx = spawn_osc52_clipboard_writer(clipboard.is_some());

        let keybindings = match keybindings::KeyBindings::load() {
            Ok(bindings) => bindings,
            Err(error) => {
                eprintln!("[Keybindings] Failed to load keybindings.toml; using defaults: {error}");
                keybindings::KeyBindings::default()
            }
        };

        // Load theme
        let current_theme = theme::Theme::get_theme(&cfg.theme).unwrap_or_default();

        let mut renderer = TerminalRenderer::new(
            cfg.font_size,
            cfg.padding,
            cfg.line_spacing,
            cfg.scrollbar_visibility.clone(),
            current_theme.clone(),
        );
        renderer.opacity = cfg.opacity;
        renderer.font_ligatures = cfg.font_ligatures;
        renderer.click_moves_cursor = cfg.click_moves_cursor;
        renderer.block_mode = cfg.block_mode;
        renderer.block_compact = cfg.block_compact;
        renderer.gpu_rendering = cfg.gpu_rendering;
        renderer.wgpu_render_state = wgpu_render_state.clone();

        // 布局引用稳定 session ID，因此某个 shell 恢复失败时可以只折叠
        // 对应分支；旧版/损坏快照则从真正的活跃 session 回退到单 pane。
        let restored_session_ids: Vec<String> = session_manager
            .sessions()
            .iter()
            .map(|session| session.metadata.session_id.clone())
            .collect();
        let tabs = tab_manager::TabManager::restore(
            &saved_tab_layouts,
            &restored_session_ids,
            session_manager.active_index(),
            saved_active_tab,
        );
        if let Some(focused_idx) = tabs.active_focused_session() {
            session_manager.switch_session(focused_idx);
        }

        // Multi-pane renderers are allocated on demand and trimmed when panes
        // close, avoiding four full GPU/text caches in the common one-pane case.
        let pane_renderers = Vec::new();

        // 命令面板/搜索历史:启动时一次性读盘,失败回 Default(load 已吞日志)。
        let history = config::Config::ui_history_path()
            .map(|p| history_persistence::HistorySnapshot::load(&p))
            .unwrap_or_default();

        let mut command_palette = command_palette::CommandPalette::new();
        command_palette.restore_recent_commands(history.recent_commands);
        let mut search_state = search::SearchState::new();
        for entry in history.search_history.into_iter().take(50) {
            search_state.history.push_back(entry);
        }
        // 配置读不出来时优先报告它:此后所有保存都被拒绝(见 Config::load_error),
        // 不说的话用户只会看到"设置改了但重启就没了"。
        let startup_notice = match &cfg.load_error {
            Some(error) => Some(format!("配置未生效,也不会被覆盖：{error}")),
            None => session_restore_notice,
        };
        let initial_status_expires_at = startup_notice
            .as_ref()
            .map(|_| std::time::Instant::now() + Duration::from_secs(10));
        let initial_status_message = startup_notice.unwrap_or_default();

        Ok(TerminalApp {
            session_manager,
            renderer,
            clipboard,
            clipboard_request_in_flight: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            osc_paste_input_barriers: crate::session_manager::SessionInputBarriers::default(),
            osc52_clipboard_write_tx,
            osc52_write_window_started: std::time::Instant::now(),
            osc52_writes_in_window: 0,
            osc52_read_window_started: std::time::Instant::now(),
            osc52_reads_in_window: 0,
            cols,
            rows,
            next_cursor_blink_time: std::time::Instant::now() + Duration::from_millis(1000),
            cursor_visible: true,
            last_activity_time: std::time::Instant::now(),
            status_message: initial_status_message,
            status_expires_at: initial_status_expires_at,
            last_window_title: String::new(),
            hovered_tab_index: None,
            dragging_tab: None,
            dragging_tab_session_id: None,
            tab_drag_pointer_origin: None,
            tab_drag_hover_session_id: None,
            tab_drag_hover_started_at: None,
            tab_drag_origin_active_session_id: None,
            drag_start_pos: None,
            tab_drag_origin: None,
            current_mouse_x: 0.0,
            tab_scroll_offset: 0.0,
            renaming_tab: None,
            search_state,
            sidebar: {
                let mut sb = sidebar::Sidebar::new();
                sb.visible = false; // 默认隐藏，opt-in 切换

                // 三个视图在两种 tab 栏布局下都可用：Top 模式下侧边栏的
                // Sessions 列表与顶部 tab 栏并存(与 frost 一致)。
                sb.view = sidebar::effective_view(cfg.sidebar_view, cfg.experimental_task_sidebar);
                sb
            },
            sidebar_name_dialog: None,
            sidebar_delete_dialog: None,
            sidebar_drop_rect: None,
            last_pointer_pos: None,
            ssh_files_follow: Default::default(),
            active_session_epoch: 1,
            command_sidebar: Default::default(),
            task_sidebar: Default::default(),
            block_selection: None,
            block_search: Default::default(),
            block_bookmarks: Default::default(),
            pending_session_export: None,
            search_replace_panel: search_replace_panel::SearchReplacePanel::new(),
            link_detector: link::LinkDetector::new(link::LinkDetectionConfig::default()),
            hovered_link: None,
            pending_link_activation: None,
            cached_links: Vec::new(),
            cached_links_projection_key: None,
            cached_links_terminal_ptr: usize::MAX,
            keybindings,
            command_palette,
            history_picker: None,
            workflow_picker: None,
            workflow_args: None,
            workflow_refusals: Vec::new(),
            force_resize_session: false,
            current_theme,
            tabs,
            pane_renderers,
            dragging_divider: None,
            pane_status_cache: pane_header::PaneStatusCache::new(),
            git_strip_cache: pane_header::GitStripCache::new(),
            pane_drag: None,
            tab_bar_drop_rects: Vec::new(),
            help_panel: help::HelpPanel::new(),
            remote_picker: Default::default(),
            config_panel: config_panel::ConfigPanel::new(),
            debug_panel: debug_panel::DebugPanel::new(),
            agent_panel: agent_panel::AgentPanel::new(),
            command_correction: command_correction::CorrectionMonitor::default(),
            // 与会话快照同一所有权规则：只有持实例锁的主窗口写共享聊天库文件，
            // 后开的窗口可以浏览/对话但落盘由主窗口负责。
            ai_chat_panel: ai_chat_panel::AiChatPanel::with_persistence_owner(is_first_instance),
            ai_command_suggestion: None,
            agent_diff: agent::AgentDiffPanel::new(),
            task_manager: agent::TaskManager::new(),
            agent_runtime: agent::AgentRuntimeManager::new(),
            jsh_notice: jsh_ui::JshNotice::default(),
            config: cfg.clone(),
            config_save_pending: false,
            config_save_deadline: std::time::Instant::now(),
            session_save_pending: !session_persistence_blocked,
            session_save_deadline: std::time::Instant::now() + std::time::Duration::from_secs(1),
            session_persistence_blocked,
            _lock_file: lock_file,
            mouse_scroll_accumulator: 0.0,
            terminal_mouse_capture: None,
            last_terminal_mouse_motion: None,
            font_size_accumulator: 0.0,
            had_ctrl_scroll_last_frame: false,
            frame_events: Vec::new(),
            paste_key_state: Default::default(),
            keyboard_input_buffer: Vec::new(),
            adaptive_frame_budget: 65536, // 初始值 64KB
            config_last_check: std::time::Instant::now(),
            smooth_scroll_velocity: 0.0,
            smooth_scroll_pixel_offset: 0.0,
            pending_paste_confirm: None,
            paste_dont_ask_again: false,
            notification_tx: spawn_desktop_notification_worker(),
            notification_window_started: std::time::Instant::now(),
            notifications_in_window: 0,
        })
    }

    fn apply_runtime_config(&mut self, ctx: &egui::Context) {
        if !self.config.block_mode {
            self.clear_block_selection();
        }
        let effective_sidebar_view =
            sidebar::effective_view(self.sidebar.view, self.config.experimental_task_sidebar);
        if effective_sidebar_view != self.sidebar.view {
            self.sidebar.note_files_user_intent();
            self.sidebar.view = effective_sidebar_view;
        }
        // Apply UI scale: use config value if provided, otherwise use native DPI
        let scale = self
            .config
            .ui_scale
            .unwrap_or_else(|| ctx.native_pixels_per_point().unwrap_or(1.0));
        ctx.set_pixels_per_point(scale);

        // eframe chooses Glow vs WGPU when the native window is created; that
        // backend cannot be swapped by hot reload. Keep the newly configured
        // value for the next launch, but apply all other settings against the
        // renderer that is actually alive in this process.
        let runtime_uses_wgpu = self.renderer.wgpu_render_state.is_some();
        let mut runtime_config = self.config.clone();
        runtime_config.app_renderer = if runtime_uses_wgpu {
            config::AppRendererType::Wgpu
        } else {
            config::AppRendererType::Glow
        };
        configure_fonts_and_gpu(
            ctx,
            self.renderer.wgpu_render_state.as_ref(),
            &runtime_config,
        );
        apply_theme_visuals(ctx, &self.current_theme);

        self.renderer.font_size = self.config.font_size;
        self.renderer.padding = self.config.padding;
        self.renderer.line_spacing = self.config.line_spacing;
        self.renderer.scrollbar_visibility = self.config.scrollbar_visibility.clone();
        self.renderer.theme = self.current_theme.clone();
        self.renderer.opacity = self.config.opacity;
        self.renderer.font_ligatures = self.config.font_ligatures;
        self.renderer.click_moves_cursor = self.config.click_moves_cursor;
        self.renderer.block_mode = self.config.block_mode;
        self.renderer.block_compact = self.config.block_compact;
        self.renderer.gpu_rendering = runtime_uses_wgpu && self.config.gpu_rendering;
        self.renderer.sync_font_metrics(ctx);
        self.renderer.invalidate_font_cache();

        for renderer in &mut self.pane_renderers {
            renderer.font_size = self.config.font_size;
            renderer.padding = self.config.padding;
            renderer.line_spacing = self.config.line_spacing;
            renderer.scrollbar_visibility = self.config.scrollbar_visibility.clone();
            renderer.theme = self.current_theme.clone();
            renderer.opacity = self.config.opacity;
            renderer.font_ligatures = self.config.font_ligatures;
            renderer.click_moves_cursor = self.config.click_moves_cursor;
            renderer.block_mode = self.config.block_mode;
            renderer.block_compact = self.config.block_compact;
            renderer.gpu_rendering = runtime_uses_wgpu && self.config.gpu_rendering;
            renderer.sync_font_metrics(ctx);
            renderer.invalidate_font_cache();
        }

        for session in self.session_manager.sessions_mut() {
            let mut terminal = session.terminal.lock();
            terminal.set_max_scrollback(self.config.scrollback_lines);
        }

        ctx.request_repaint();
    }

    /// Surgical apply for font-size-only changes (Ctrl+wheel / font:* keys).
    /// The full apply_runtime_config path re-resolves fonts through fontconfig
    /// (subprocess per candidate), re-reads the font files from disk and swaps
    /// the whole egui font atlas — none of which a size change needs, and all
    /// of which makes interactive zooming stutter. Resize the live GPU atlas in
    /// place and refresh the renderers' cell metrics instead.
    fn apply_font_size_change(&mut self, ctx: &egui::Context) {
        if let Some(render_state) = self.renderer.wgpu_render_state.as_ref() {
            let font_size_px = self.config.font_size * ctx.pixels_per_point();
            let mut renderer = render_state.renderer.write();
            if let Some(gpu_res) = renderer
                .callback_resources
                .get_mut::<gpu::callback::GpuResources>()
            {
                gpu_res.atlas.set_font_size_px(
                    &render_state.device,
                    &render_state.queue,
                    font_size_px,
                );
            }
        }

        self.renderer.font_size = self.config.font_size;
        self.renderer.sync_font_metrics(ctx);
        self.renderer.invalidate_font_cache();
        for renderer in &mut self.pane_renderers {
            renderer.font_size = self.config.font_size;
            renderer.sync_font_metrics(ctx);
            renderer.invalidate_font_cache();
        }

        ctx.request_repaint();
    }

    fn create_session_with_current_config(
        &mut self,
        name: Option<String>,
        tags: Option<Vec<String>>,
    ) -> usize {
        let (cols, rows) = clamp_terminal_dimensions(self.cols, self.rows);
        let old_len = self.session_manager.len();
        let new_idx =
            self.session_manager
                .new_session(name, tags, cols, rows, self.config.scrollback_lines);
        if self.session_manager.len() > old_len {
            // 会话索引是全局的，所以插入要在每个 tab 的树里重编号，不只是
            // 当前 tab。新会话本身还没有归属，由调用方决定是分屏还是开新 tab。
            self.tabs.on_session_inserted(new_idx);
        }
        new_idx
    }

    /// Files-header Local action: open a fresh interactive tab at the exact
    /// tree root. This is explicit rather than inheriting the active PTY's cwd,
    /// because the user may have browsed the tree independently.
    fn open_local_terminal_at_sidebar_root(&mut self, cwd: &std::path::Path) {
        let (cols, rows) = clamp_terminal_dimensions(self.cols, self.rows);
        let created = match self.session_manager.new_session_in_cwd(
            None,
            None,
            cwd,
            cols,
            rows,
            self.config.scrollback_lines,
        ) {
            Ok(created) => created,
            Err(error) => {
                self.set_status_for(
                    format!("无法从文件树目录新建终端：{error}"),
                    Duration::from_secs(5),
                );
                return;
            }
        };
        self.tabs.on_session_inserted(created.session_index);
        self.tabs.insert_tab_after_active(created.session_index);
        self.activate_session(created.session_index);
        self.force_resize_session = true;
        self.schedule_session_save();
        let display = jterm_core::review_input::safe_inline_display(&cwd.to_string_lossy(), 256);
        self.set_status_for(
            format!("已在 {display} 新建本地终端"),
            Duration::from_secs(5),
        );
    }

    fn active_ssh_files_observation(&self) -> ssh_files_follow::Observation {
        self.session_manager
            .sessions()
            .get(self.session_manager.active_index())
            .map(ssh_files_follow::observe_session)
            .unwrap_or(ssh_files_follow::Observation::None)
    }

    fn active_session_allows_local_files_cwd_follow(&self) -> bool {
        let Some(session) = self
            .session_manager
            .sessions()
            .get(self.session_manager.active_index())
        else {
            return false;
        };
        session.purpose == crate::session::SessionPurpose::Interactive
            && matches!(
                ssh_files_follow::observe_session(session),
                ssh_files_follow::Observation::None
            )
    }

    /// Advance the single-flight hand-written SSH observer. Probe completion
    /// is gated before *any* status/tree/sidebar mutation; a stale success or
    /// failure is silently discarded because the user's newer UI/session
    /// intent owns the surface.
    fn update_ssh_files_follow(&mut self, ctx: &egui::Context, frame_start_files_user_intent: u64) {
        while let Some(result) = self.ssh_files_follow.try_result() {
            if self
                .ssh_files_follow
                .pending
                .as_ref()
                .is_none_or(|pending| pending.token != result.token)
            {
                continue;
            }
            let pending = self
                .ssh_files_follow
                .pending
                .take()
                .expect("matching pending SSH Files probe");
            let observation = self.active_ssh_files_observation();
            let observation_epoch = self.ssh_files_follow.sync_observation(&observation);
            let sidebar_ui = ssh_files_follow::SidebarUiSnapshot::capture(
                &self.sidebar,
                self.sidebar_name_dialog.is_some() || self.sidebar_delete_dialog.is_some(),
            );
            let sidebar_ui_epoch = self.ssh_files_follow.sync_sidebar_ui(&sidebar_ui);
            let files_context_current =
                self.sidebar.files_intent_is_current(&pending.files_context)
                    && !self.sidebar.has_pending_op();
            if !ssh_files_follow::result_is_current(
                &pending,
                &observation,
                observation_epoch,
                self.active_session_epoch,
                self.sidebar.files_user_intent_generation(),
                sidebar_ui_epoch,
                files_context_current,
                &self.sidebar.current_dir,
                &sidebar_ui,
            ) {
                // A user file operation or Files UI change that happened while
                // the network probe was in flight owns the surface and
                // consumes this observation. Only a real process/focus
                // authority change re-arms it; ordinary UI cancellation must
                // not turn the same argv into a surprise automatic retry.
                if ssh_files_follow::stale_probe_should_rearm(
                    &pending,
                    &observation,
                    observation_epoch,
                    self.active_session_epoch,
                    self.sidebar.files_user_intent_generation(),
                    sidebar_ui_epoch,
                    files_context_current,
                    &self.sidebar.current_dir,
                    &sidebar_ui,
                ) {
                    // Focus/process ABA is not a user rejection of Files
                    // following. The old staged result is invalid, but when
                    // this exact session becomes active again it deserves a
                    // fresh probe rather than remaining deduped forever.
                    self.ssh_files_follow.rearm_after_stale_probe(&pending.key);
                }
                continue;
            }

            let label = crate::config::remote_host_runtime_label(&pending.profile);
            match pending.commit {
                ssh_files_follow::FollowCommit::RebindCurrentOverlay => {
                    match self
                        .sidebar
                        .finish_probed_execution_overlay(pending.overlay, result.outcome)
                    {
                        Ok(()) => {
                            self.ssh_files_follow.clear_failure();
                            self.set_status_for(
                                format!("Files 已验证并切换 SSH 连接：{label}"),
                                Duration::from_secs(5),
                            );
                        }
                        Err(error) => {
                            self.ssh_files_follow.record_failure(pending.key);
                            let error = jterm_core::review_input::safe_inline_display(&error, 320);
                            self.set_status_for(
                                format!(
                                    "无法更新远程 Files 连接：{error}。旧文件树和旧连接保持不变。非交互 BatchMode 连接需要可用的 SSH key、agent，或可复用的 ControlMaster/ControlPath socket；配置后可点击重试。"
                                ),
                                Duration::from_secs(15),
                            );
                        }
                    }
                }
                ssh_files_follow::FollowCommit::ReplaceLocation => match result.outcome {
                    Ok(home) => {
                        let Some(location) = pending
                            .authority
                            .current_location(&pending.profile, &self.config.remote_hosts)
                        else {
                            self.ssh_files_follow.record_failure(pending.key);
                            self.set_status_for(
                                "无法自动打开远程 Files：匹配的 saved profile 在连接期间被修改、删除或变得不唯一。可点击重试。",
                                Duration::from_secs(12),
                            );
                            continue;
                        };
                        match self
                            .sidebar
                            .commit_probed_location(location, pending.overlay, home)
                        {
                            Ok(scan_error) => {
                                self.ssh_files_follow.clear_failure();
                                if let Some(error) = scan_error {
                                    self.set_status_for(
                                        format!("已连接 {label}，但文件树读取失败：{error}"),
                                        Duration::from_secs(7),
                                    );
                                } else {
                                    self.set_status_for(
                                        format!("Files 已跟随 SSH：{label}"),
                                        Duration::from_secs(5),
                                    );
                                }
                            }
                            Err(error) => {
                                self.ssh_files_follow.record_failure(pending.key);
                                self.set_status_for(
                                    format!("无法自动打开远程 Files：{error}。可点击重试。"),
                                    Duration::from_secs(12),
                                );
                            }
                        }
                    }
                    Err(error) => {
                        self.ssh_files_follow.record_failure(pending.key);
                        let error = jterm_core::review_input::safe_inline_display(&error, 320);
                        self.set_status_for(
                            format!(
                                "无法自动打开远程 Files：{error}。非交互 BatchMode 连接需要可用的 SSH key、agent，或可复用的 ControlMaster/ControlPath socket；配置后可点击重试。"
                            ),
                            Duration::from_secs(15),
                        );
                    }
                },
            }
        }

        let observation = self.active_ssh_files_observation();
        let observation_epoch = self.ssh_files_follow.sync_observation(&observation);
        let sidebar_ui = ssh_files_follow::SidebarUiSnapshot::capture(
            &self.sidebar,
            self.sidebar_name_dialog.is_some() || self.sidebar_delete_dialog.is_some(),
        );
        let sidebar_ui_epoch = self.ssh_files_follow.sync_sidebar_ui(&sidebar_ui);
        match observation {
            ssh_files_follow::Observation::None => {
                // SSH exiting only re-arms future observations; the transient
                // Files tree remains exactly where the user left it.
                self.ssh_files_follow.mark_observation_absent();
            }
            ssh_files_follow::Observation::Unsupported { key, reason } => {
                if self.ssh_files_follow.pending.is_none()
                    && !self.ssh_files_follow.was_handled(&key)
                {
                    self.ssh_files_follow.mark_handled(key);
                    self.set_status_for(
                        format!("无法自动打开远程 Files：{reason}"),
                        Duration::from_secs(7),
                    );
                }
            }
            ssh_files_follow::Observation::Target {
                key,
                profile,
                overlay,
            } => {
                let retry = self.ssh_files_follow.retry_requested_for(&key);
                if ssh_files_follow::same_frame_files_intent_suppresses_new_observation(
                    frame_start_files_user_intent,
                    self.sidebar.files_user_intent_generation(),
                ) {
                    // This frame's user action predates the first time the
                    // observation can be staged. Treat it exactly like an
                    // in-flight gate failure instead of letting end-of-frame
                    // ordering force-reveal Remote Files on a later callback.
                    self.ssh_files_follow.suppress_for_files_intent(&key);
                    return;
                }
                // Never let an automatic backend switch cancel work the user
                // had already started. This explicit Files intent consumes the
                // current observation; only the persistent Retry control can
                // ask for another probe of the same live process.
                if self.sidebar.has_pending_op()
                    || self.sidebar_name_dialog.is_some()
                    || self.sidebar_delete_dialog.is_some()
                {
                    self.ssh_files_follow.suppress_for_files_intent(&key);
                    return;
                }
                if self.ssh_files_follow.pending.is_some() {
                    return;
                }
                if self.ssh_files_follow.was_handled(&key) && !retry {
                    return;
                }
                let same_target_with_valid_tree = ssh_files_follow::location_matches_observed(
                    self.sidebar.location(),
                    &self.config.remote_hosts,
                    &profile,
                ) && self.sidebar.root.is_some()
                    && self.sidebar.current_dir.is_absolute()
                    && self.sidebar.location_error().is_none();
                let commit = match ssh_files_follow::same_target_action(
                    same_target_with_valid_tree,
                    self.sidebar.execution_overlay(),
                    &overlay,
                ) {
                    ssh_files_follow::SameTargetAction::RevealExisting => {
                        self.ssh_files_follow.mark_handled(key);
                        self.ssh_files_follow.clear_failure();
                        return;
                    }
                    ssh_files_follow::SameTargetAction::ProbeOverlayUpgrade => {
                        ssh_files_follow::FollowCommit::RebindCurrentOverlay
                    }
                    ssh_files_follow::SameTargetAction::DifferentLocation => {
                        ssh_files_follow::FollowCommit::ReplaceLocation
                    }
                };
                if let Err(problem) = crate::config::validate_remote_host(&profile) {
                    self.ssh_files_follow.mark_handled(key.clone());
                    self.ssh_files_follow.record_failure(key);
                    self.set_status_for(
                        format!("无法自动打开远程 Files：临时 SSH profile 不安全：{problem}"),
                        Duration::from_secs(7),
                    );
                    return;
                }

                let authority =
                    ssh_files_follow::target_authority(&profile, &self.config.remote_hosts);
                if let Err(error) = remote_fs::validate_execution_endpoint(
                    &remote_fs::FsLocation::Transient(authority.profile().clone()),
                    &[],
                    &overlay,
                ) {
                    self.ssh_files_follow.mark_handled(key.clone());
                    self.ssh_files_follow.record_failure(key);
                    self.set_status_for(
                        format!("无法自动打开远程 Files：SSH execution overlay 不安全：{error}"),
                        Duration::from_secs(12),
                    );
                    return;
                }

                let files_context = self.sidebar.files_intent_context();
                let root = self.sidebar.current_dir.clone();
                if retry {
                    self.ssh_files_follow.consume_retry_for(&key);
                }
                self.ssh_files_follow.mark_handled(key.clone());
                let pending = ssh_files_follow::PendingProbe {
                    token: 0,
                    observation_epoch,
                    active_session_epoch: self.active_session_epoch,
                    files_user_intent_generation: self.sidebar.files_user_intent_generation(),
                    sidebar_ui_epoch,
                    key: key.clone(),
                    authority,
                    profile: *profile,
                    overlay,
                    commit,
                    files_context,
                    root,
                    sidebar_ui,
                };
                if let Err(error) = self.ssh_files_follow.begin_probe(pending, ctx.clone()) {
                    self.ssh_files_follow.record_failure(key);
                    self.set_status_for(
                        format!("无法自动打开远程 Files：{error}。可点击重试。"),
                        Duration::from_secs(12),
                    );
                }
            }
        }
    }

    /// 顶部水平 tab 栏是否应显示：Top 模式下始终显示。
    /// 即便只有一个会话也保留，因为栏内含有侧边栏 toggle 控件。
    fn show_top_tab_bar(&self) -> bool {
        matches!(self.config.tab_bar_position, config::TabBarPosition::Top)
    }

    /// 切换标签栏位置(顶部 ⇄ 侧边栏)，并同步侧边栏视图与配置。
    /// 由顶栏内的位置切换按钮调用(两种模式下均可触发)。
    fn toggle_tab_bar_position(&mut self) {
        self.sidebar.note_files_user_intent();
        self.config.tab_bar_position = match self.config.tab_bar_position {
            config::TabBarPosition::Top => config::TabBarPosition::Sidebar,
            config::TabBarPosition::Sidebar => config::TabBarPosition::Top,
        };
        if !matches!(self.config.tab_bar_position, config::TabBarPosition::Top) {
            // 标签移入侧边栏：恢复上次记住的视图并确保侧边栏可见，否则标签不可达
            self.sidebar.view = sidebar::effective_view(
                self.config.sidebar_view,
                self.config.experimental_task_sidebar,
            );
            self.sidebar.visible = true;
            if self.sidebar.view == sidebar::SidebarView::Files {
                if let Some(error) = self.sidebar.refresh() {
                    self.set_status_for(format!("文件树刷新失败：{error}"), Duration::from_secs(5));
                }
            }
        }
        self.config_panel.sync_from_config(&self.config);
        self.schedule_config_save();
    }

    /// 渲染左侧文件树侧边栏。必须在 CentralPanel 之前调用，
    /// 否则中央区域不会正确收缩。
    #[allow(deprecated)]
    fn render_sidebar(&mut self, root_ui: &mut egui::Ui) {
        // Configuration can change while the sidebar is hidden (and a Files
        // dialog can still be open). Reconcile before the visibility early
        // return so stale remote paths never retain a live dispatch context.
        if let Some(message) = self.sidebar.set_remote_hosts(&self.config.remote_hosts) {
            self.set_status_for(message, Duration::from_secs(6));
        }
        if let Some(error) = self.sidebar.poll_scan_results().into_iter().last() {
            self.set_status_for(format!("文件树读取失败：{error}"), Duration::from_secs(5));
        }
        // 文件操作 worker 的结果（新建/重命名/删除/粘贴、远程起始目录解析）。
        for message in self.sidebar.poll_op_results() {
            self.set_status_for(message, Duration::from_secs(5));
        }
        if self.sidebar.visible && self.sidebar.view == sidebar::SidebarView::Files {
            for error in self.sidebar.auto_revalidate_visible() {
                self.set_status_for(
                    format!("Remote Files 后台重验失败：{error}"),
                    Duration::from_secs(5),
                );
            }
            if self.sidebar.location().is_remote() {
                root_ui.ctx().request_repaint_after(Duration::from_secs(5));
            }
        }
        if !self.sidebar.visible {
            // 展开按钮统一由顶部栏内的 ☰ 负责(Top 模式在 tab 栏，Sidebar 模式在精简顶部栏)，
            // 不再使用浮动按钮，避免覆盖终端内容。后台结果仍需在上方收割，
            // 否则隐藏侧边栏会让自动 SSH follow 永久误判为“仍有操作”。
            self.sidebar_drop_rect = None;
            return;
        }
        // 拖拽导入的帧级输入：落下的文件（raw_input_hook 已按面板区域放行）、
        // OS 拖悬停状态与指针位置。只在 Files 视图消费。
        let (dropped_paths, hover_files_active, pointer_pos) =
            if self.sidebar.view == sidebar::SidebarView::Files {
                root_ui.ctx().input(|input| {
                    (
                        input
                            .raw
                            .dropped_files
                            .iter()
                            .map(|file| file.path().to_path_buf())
                            .collect::<Vec<std::path::PathBuf>>(),
                        !input.raw.hovered_files.is_empty(),
                        input
                            .pointer
                            .interact_pos()
                            .or_else(|| input.pointer.latest_pos()),
                    )
                })
            } else {
                (Vec::new(), false, None)
            };

        // Follow the shell's authoritative OSC 7 cwd (or the local process
        // cwd fallback) instead of guessing that a queued `cd` succeeded.
        // This also keeps the file tree correct after users type `cd` by hand.
        // 仅在浏览本机时跟随：本地 shell 的 cwd 对远程文件系统没有意义。
        if self.sidebar.view == sidebar::SidebarView::Files
            && matches!(self.sidebar.location(), remote_fs::FsLocation::Local)
            && self.ssh_files_follow.pending.is_none()
            && self.ssh_files_follow.handled_observation.is_none()
            && self.active_session_allows_local_files_cwd_follow()
        {
            let reported_cwd = {
                let session = self.session_manager.get_active_session_mut();
                let osc7 = session.terminal.lock().current_working_dir.clone();
                osc7.or_else(|| jterm_core::process::process_cwd(session.get_shell_pid()))
            };
            let changed_directory = reported_cwd
                .map(std::path::PathBuf::from)
                .filter(|path| self.sidebar.current_dir != *path);
            if let Some(path) = changed_directory {
                if let Some(error) = self.sidebar.set_current_dir(path) {
                    self.set_status_for(
                        format!("文件树目录切换失败：{error}"),
                        Duration::from_secs(5),
                    );
                }
            }
        }

        // 树遍历期间只收集动作，闭包结束后再 mutate，规避借用冲突
        let mut toggle_path: Option<std::path::PathBuf> = None;
        let mut select_action: Option<(std::path::PathBuf, bool, FsSelectMode)> = None;
        let mut cd_path: Option<std::path::PathBuf> = None;
        let mut show_more_path: Option<std::path::PathBuf> = None;
        let mut retry_path: Option<std::path::PathBuf> = None;
        let mut navigate_back = false;
        let mut navigate_forward = false;
        let mut open_path_entry = false;
        let mut submit_path_entry = false;
        let mut cancel_path_entry = false;
        let mut do_refresh = false;
        let ssh_retry_available = if self.sidebar.view == sidebar::SidebarView::Files {
            let observation = self.active_ssh_files_observation();
            self.ssh_files_follow
                .retry_available_for_observation(&observation)
        } else {
            false
        };
        let mut retry_ssh_files = false;
        let mut view_changed = false;
        let mut location_changed: Option<remote_fs::FsLocation> = None;
        let mut files_terminal_target: Option<sidebar::FilesTerminalTarget> = None;
        let mut fs_menu_action: Option<FsMenuAction> = None;
        // Every menu action is stamped before rendering. If a location/root
        // change is also applied later in this frame, dispatch fails closed.
        let fs_intent_context = self.sidebar.files_intent_context();
        let mut cancel_transfer = false;
        // 右键点在选中集之外：选中集先收缩为该行（闭包结束后写回）。
        let mut selection_apply: Option<std::collections::BTreeMap<std::path::PathBuf, bool>> =
            None;
        // 过滤开关本帧刚打开时给输入行焦点。
        let mut filter_request_focus = false;
        let mut filter_interacted = false;
        let mut filter_has_focus = false;
        let mut path_entry_has_focus = false;
        let mut tree_has_focus = false;
        let mut show_hidden_changed: Option<bool> = None;
        let mut files_popup_open = false;
        // 本帧渲染出的文件树行矩形（拖放落点命中测试用，帧级、不持久）。
        let mut tree_row_rects: Vec<(egui::Rect, std::path::PathBuf, bool)> = Vec::new();
        // 粘贴状态：同位置直接粘贴；跨位置走流式传输（下载/上传/中转）。
        let paste_state = match &self.sidebar.clipboard {
            None => FsPasteState::Empty,
            Some(clipboard)
                if remote_fs::same_files_namespace(
                    &clipboard.loc,
                    self.sidebar.location(),
                    &self.config.remote_hosts,
                ) =>
            {
                FsPasteState::Ready
            }
            Some(clipboard) => match (
                clipboard.loc.is_remote(),
                self.sidebar.location().is_remote(),
            ) {
                (true, false) => FsPasteState::Download,
                (false, true) => FsPasteState::Upload,
                (true, true) => FsPasteState::Relay,
                (false, false) => FsPasteState::Ready,
            },
        };

        let panel_bg = theme::Theme::rgb_to_color32(self.current_theme.ui.panel_bg);
        let panel_response = egui::Panel::left("file_tree")
            .resizable(true)
            .default_size(self.sidebar.width)
            .frame(egui::Frame::NONE.fill(panel_bg).inner_margin(6.0))
            .show(root_ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    // Sessions 在两种 tab 栏布局下都可选：Top 模式下它是顶部
                    // tab 栏之外的一份纵向标签列表，而非替代品。
                    if ui
                        .selectable_label(
                            self.sidebar.view == sidebar::SidebarView::Sessions,
                            egui::RichText::new("Sessions").strong(),
                        )
                        .clicked()
                    {
                        self.sidebar.view = sidebar::SidebarView::Sessions;
                        view_changed = true;
                    }
                    if ui
                        .selectable_label(
                            self.sidebar.view == sidebar::SidebarView::Files,
                            egui::RichText::new("Files").strong(),
                        )
                        .clicked()
                    {
                        self.sidebar.view = sidebar::SidebarView::Files;
                        view_changed = true;
                    }
                    if ui
                        .selectable_label(
                            self.sidebar.view == sidebar::SidebarView::Commands,
                            egui::RichText::new("Commands").strong(),
                        )
                        .clicked()
                    {
                        self.sidebar.view = sidebar::SidebarView::Commands;
                        view_changed = true;
                    }
                    if self.config.experimental_task_sidebar {
                        let attention = self.task_manager.attention_count();
                        let label = if attention == 0 {
                            "Tasks".to_string()
                        } else {
                            format!("Tasks {attention}")
                        };
                        if ui
                            .selectable_label(
                                self.sidebar.view == sidebar::SidebarView::Tasks,
                                egui::RichText::new(label).strong(),
                            )
                            .clicked()
                        {
                            self.sidebar.view = sidebar::SidebarView::Tasks;
                            view_changed = true;
                        }
                    }
                    if self.sidebar.view == sidebar::SidebarView::Files
                        && ui.button("⟳").on_hover_text("Refresh").clicked()
                    {
                        do_refresh = true;
                    }
                    if self.sidebar.view == sidebar::SidebarView::Files {
                        if self.sidebar.location().is_remote() {
                            if ui
                                .add_enabled(
                                    self.sidebar.can_navigate_back(),
                                    egui::Button::new("←"),
                                )
                                .on_hover_text("Remote Files 后退（Alt+Left）")
                                .clicked()
                            {
                                navigate_back = true;
                            }
                            if ui
                                .add_enabled(
                                    self.sidebar.can_navigate_forward(),
                                    egui::Button::new("→"),
                                )
                                .on_hover_text("Remote Files 前进（Alt+Right）")
                                .clicked()
                            {
                                navigate_forward = true;
                            }
                            let parent = self.sidebar.parent_dir();
                            let up = ui
                                .add_enabled(parent.is_some(), egui::Button::new("↑"))
                                .on_hover_text("Remote Files 上级目录（Alt+Up）");
                            if up.clicked() {
                                cd_path = parent;
                            }
                            let home = self.sidebar.home_dir().map(std::path::Path::to_path_buf);
                            let at_home = home
                                .as_ref()
                                .is_none_or(|home| *home == self.sidebar.current_dir);
                            let home_button = ui
                                .add_enabled(!at_home, egui::Button::new("⌂"))
                                .on_hover_text("Remote Files Home（Alt+Home）");
                            if home_button.clicked() {
                                cd_path = home;
                            }
                            if ui
                                .button("路径")
                                .on_hover_text("输入绝对路径（Ctrl+L）")
                                .clicked()
                            {
                                open_path_entry = true;
                            }
                        }
                        if ssh_retry_available
                            && ui
                                .button("Retry SSH Files")
                                .on_hover_text(
                                    "Retry the failed Files probe for the still-active SSH process",
                                )
                                .clicked()
                        {
                            retry_ssh_files = true;
                        }
                        // 浏览位置选择器：本机 + config.remote_hosts 里的
                        // SSH 主机 / Docker 容器。每帧从配置重建，设置面板
                        // 的增删改立即生效。
                        let hosts = &self.config.remote_hosts;
                        let current = self.sidebar.location().clone();
                        let location_picker =
                            egui::ComboBox::from_id_salt("sidebar-fs-location")
                            .selected_text(current.label(hosts))
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(
                                        matches!(current, remote_fs::FsLocation::Local),
                                        "Local",
                                    )
                                    .clicked()
                                {
                                    location_changed = Some(remote_fs::FsLocation::Local);
                                }
                                for (index, _host) in
                                    hosts.iter().take(config::MAX_REMOTE_HOSTS).enumerate()
                                {
                                    let location = remote_fs::FsLocation::Remote(index);
                                    if ui
                                        .selectable_label(
                                            current == location,
                                            location.label(hosts),
                                        )
                                        .on_hover_text(location.detail(hosts))
                                        .clicked()
                                    {
                                        location_changed = Some(location);
                                    }
                                }
                                if matches!(current, remote_fs::FsLocation::Transient(_)) {
                                    // A transient process-observed profile is a
                                    // real current choice, but never becomes a
                                    // config row implicitly. Keep it visible in
                                    // the open dropdown so selection state does
                                    // not appear to point at no item.
                                    ui.selectable_label(true, current.label(hosts)).on_hover_text(
                                        format!(
                                            "{}\nTemporary profile observed from the active SSH process",
                                            current.detail(hosts)
                                        ),
                                    );
                                }
                            });
                        if location_picker.inner.is_some() {
                            files_popup_open = true;
                        }
                        location_picker
                            .response
                            .on_hover_text(current.detail(hosts));
                        let terminal_target = self.sidebar.files_terminal_target();
                        let (terminal_label, terminal_hint, terminal_enabled) =
                            match &terminal_target {
                                Some(sidebar::FilesTerminalTarget::Local(path)) => (
                                    "Open terminal here",
                                    format!(
                                        "Open a new local terminal tab in {}",
                                        path.display()
                                    ),
                                    true,
                                ),
                                Some(sidebar::FilesTerminalTarget::Remote {
                                    index,
                                    overlay,
                                }) => {
                                    let display = self
                                        .config
                                        .remote_hosts
                                        .get(*index)
                                        .map(|host| {
                                            crate::config::remote_host_display_name(host, *index)
                                        })
                                        .unwrap_or_else(|| format!("remote host #{}", index + 1));
                                    match crate::config::validate_remote_host_at(
                                        &self.config.remote_hosts,
                                        *index,
                                    ) {
                                        Ok(_) if !overlay.is_empty() => (
                                            "Connect terminal (SSH login)",
                                            format!(
                                                "Open {display} as a plain interactive SSH login using the live Files connection, not the current Files path"
                                            ),
                                            true,
                                        ),
                                        Ok(_) => (
                                            "Connect terminal (profile default)",
                                            format!(
                                                "Open {display} in a new terminal tab using the profile default directory, not the current Files path"
                                            ),
                                            true,
                                        ),
                                        Err(problem) => (
                                            "Connect terminal (profile default)",
                                            format!("{display} is unavailable: {problem}"),
                                            false,
                                        ),
                                    }
                                }
                                Some(sidebar::FilesTerminalTarget::Transient {
                                    host,
                                    overlay: _,
                                }) => {
                                    let display =
                                        crate::config::remote_host_runtime_label(host);
                                    match crate::config::validate_remote_host(host) {
                                        Ok(()) => (
                                            "Connect terminal (SSH login)",
                                            format!(
                                                "Open {display} in a new terminal tab using the observed SSH connection options and its default login directory, not the current Files path"
                                            ),
                                            true,
                                        ),
                                        Err(problem) => (
                                            "Connect terminal (SSH login)",
                                            format!("{display} is unavailable: {problem}"),
                                            false,
                                        ),
                                    }
                                }
                                None => (
                                    "Open terminal here",
                                    "Wait for the local Files root to become available".to_string(),
                                    false,
                                ),
                            };
                        let terminal_response = ui.add_enabled(
                            terminal_enabled,
                            egui::Button::new(terminal_label),
                        );
                        let terminal_clicked = terminal_response.clicked();
                        if terminal_enabled {
                            terminal_response.on_hover_text(terminal_hint);
                        } else {
                            terminal_response.on_disabled_hover_text(terminal_hint);
                        }
                        if terminal_clicked {
                            files_terminal_target = terminal_target;
                        }
                        // 树内过滤开关（客户端过滤已加载的树，不触发新扫描）。
                        let filter_icon =
                            if self.sidebar.filter_open && !self.sidebar.filter.is_empty() {
                                egui::RichText::new("🔍").strong()
                            } else {
                                egui::RichText::new("🔍")
                            };
                        if ui
                            .button(filter_icon)
                            .on_hover_text("树内过滤（名称子串；Esc 或再次点击关闭）")
                            .clicked()
                        {
                            filter_interacted = true;
                            self.sidebar.filter_open = !self.sidebar.filter_open;
                            if !self.sidebar.filter_open {
                                self.sidebar.filter.clear();
                            } else {
                                filter_request_focus = true;
                            }
                        }
                        if ui
                            .selectable_label(self.sidebar.show_hidden(), "Hidden")
                            .on_hover_text("显示/隐藏以点开头的文件；切换会安全刷新当前树")
                            .clicked()
                        {
                            show_hidden_changed = Some(!self.sidebar.show_hidden());
                        }
                    }
                });
                ui.separator();

                match self.sidebar.view {
                    sidebar::SidebarView::Sessions => self.render_sidebar_sessions(ui),
                    sidebar::SidebarView::Commands => self.render_sidebar_commands(ui),
                    sidebar::SidebarView::Tasks => self.render_sidebar_tasks(ui),
                    sidebar::SidebarView::Files => {
                        if self.sidebar.location().is_remote() {
                            if self.sidebar.path_entry_open {
                                ui.horizontal(|ui| {
                                    ui.label("路径:");
                                    let response = ui.add(
                                        egui::TextEdit::singleline(
                                            &mut self.sidebar.path_entry,
                                        )
                                        .id_salt("remote-files-path-entry")
                                        .hint_text("/absolute/path")
                                        .desired_width(f32::INFINITY),
                                    );
                                    path_entry_has_focus = response.has_focus();
                                    if response.has_focus()
                                        && ui.input(|input| {
                                            input.key_pressed(egui::Key::Enter)
                                        })
                                    {
                                        submit_path_entry = true;
                                    }
                                    if response.has_focus()
                                        && ui.input(|input| {
                                            input.key_pressed(egui::Key::Escape)
                                        })
                                    {
                                        cancel_path_entry = true;
                                    }
                                });
                            } else {
                                let breadcrumbs = self.sidebar.breadcrumbs();
                                ui.horizontal_wrapped(|ui| {
                                    for (index, (label, path)) in
                                        breadcrumbs.into_iter().enumerate()
                                    {
                                        if index > 0 {
                                            ui.label(egui::RichText::new("›").weak());
                                        }
                                        let at_current = path == self.sidebar.current_dir;
                                        if ui
                                            .add_enabled(
                                                !at_current,
                                                egui::Button::new(label).frame(false),
                                            )
                                            .on_hover_text(path.display().to_string())
                                            .clicked()
                                        {
                                            cd_path = Some(path);
                                        }
                                    }
                                });
                            }
                            if let Some(target) = self.sidebar.navigation_pending_target() {
                                ui.horizontal(|ui| {
                                    ui.spinner();
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "正在验证 {}（当前树保持可用）",
                                            target.display()
                                        ))
                                        .weak()
                                        .small(),
                                    );
                                });
                            }
                        }
                        // 远程位置的起始目录解析/失败提示。
                        if self.sidebar.is_starting() {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("正在验证新位置…（当前文件树保持可用）");
                            });
                        } else if let Some(error) = self.sidebar.location_error() {
                            ui.colored_label(
                                ui.visuals().error_fg_color,
                                format!("无法进入该位置：{error}"),
                            );
                        }
                        let (pending_scans, queued_scans) = self.sidebar.scan_activity();
                        if pending_scans > 1 || queued_scans > 0 {
                            ui.label(
                                egui::RichText::new(format!(
                                    "目录请求：{pending_scans} 个（排队 {queued_scans} 个）"
                                ))
                                .weak()
                                .small(),
                            );
                        }
                        if let Some(timing) = self.sidebar.last_scan_timing() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "最近读取：{} ms（排队 {} ms）",
                                    timing.run_time.as_millis(),
                                    timing.queue_delay.as_millis()
                                ))
                                .weak()
                                .small(),
                            )
                            .on_hover_text(timing.path.display().to_string());
                        }
                        // 传输忙碌行：进度原地更新，✕ 取消（仅传输可取消）。
                        if let Some(status) = self.sidebar.transfer_status() {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                let amount = match status.total {
                                    Some(total) => format!(
                                        "{} / {}",
                                        remote_fs::format_bytes(status.bytes),
                                        remote_fs::format_bytes(total)
                                    ),
                                    None => remote_fs::format_bytes(status.bytes),
                                };
                                ui.label(format!(
                                    "正在{} {}… {}",
                                    status.direction, status.name, amount
                                ));
                                if ui.small_button("✕").on_hover_text("取消传输").clicked() {
                                    cancel_transfer = true;
                                }
                            });
                        }
                        // 树内过滤输入行：客户端过滤已加载的树，不触发新扫描。
                        if self.sidebar.filter_open {
                            ui.horizontal(|ui| {
                                ui.label("过滤:");
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut self.sidebar.filter)
                                        .hint_text("名称子串，Esc 关闭")
                                        .desired_width(f32::INFINITY),
                                );
                                if resp.changed() {
                                    filter_interacted = true;
                                }
                                if filter_request_focus {
                                    resp.request_focus();
                                }
                                filter_has_focus = resp.has_focus();
                                if resp.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Escape))
                                {
                                    filter_interacted = true;
                                    self.sidebar.filter.clear();
                                    self.sidebar.filter_open = false;
                                }
                            });
                        }
                        let filter_query = if self.sidebar.filter_open {
                            self.sidebar.filter.trim().to_lowercase()
                        } else {
                            String::new()
                        };
                        if !self.sidebar.current_dir.as_os_str().is_empty() {
                            if let Some(dir) = self
                                .sidebar
                                .current_dir
                                .file_name()
                                .and_then(|n| n.to_str())
                            {
                                // 根目录行也挂上下文菜单：在根里新建/粘贴/刷新。
                                // 行矩形计入拖放命中表（落点 = 当前根目录）。
                                let root_dir = self.sidebar.current_dir.clone();
                                let snapshot_hint = self
                                    .sidebar
                                    .root
                                    .as_ref()
                                    .and_then(sidebar::FileTreeNode::snapshot_age)
                                    .map(snapshot_age_label)
                                    .map(|age| format!("\n上次成功快照：{age}"))
                                    .unwrap_or_default();
                                let label_resp = ui
                                    .label(egui::RichText::new(dir).weak().small())
                                    .on_hover_text(format!(
                                        "{} {snapshot_hint}",
                                        root_dir.display()
                                    ));
                                tree_has_focus |= label_resp.has_focus();
                                tree_row_rects.push((label_resp.rect, root_dir.clone(), true));
                                label_resp.context_menu(|ui| {
                                    Self::fs_context_menu(
                                        ui,
                                        None,
                                        &[],
                                        &root_dir,
                                        &mut fs_menu_action,
                                        paste_state,
                                    );
                                });
                                files_popup_open |= label_resp.context_menu_opened();
                            }
                        }
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if let Some(root) = &self.sidebar.root {
                                // 过滤视图：命中项 + 祖先（强制展开），纯客户端、不动原树。
                                let filtered_root;
                                let root = if filter_query.is_empty() {
                                    Some(root)
                                } else {
                                    filtered_root = root.filtered(&filter_query);
                                    filtered_root.as_ref()
                                };
                                let Some(root) = root else {
                                    ui.label("无匹配项");
                                    return;
                                };
                                let snapshot_age = root.snapshot_age().map(snapshot_age_label);
                                if root.is_refreshing() {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label(match &snapshot_age {
                                            Some(age) => {
                                                format!("正在刷新…（显示 {age} 快照）")
                                            }
                                            None => "正在刷新…（显示上次结果）".to_string(),
                                        });
                                    });
                                } else if root.is_loading() {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label("正在读取目录…");
                                    });
                                } else if let Some(error) = root.load_error() {
                                    let retry_cooldown = self.sidebar.retry_cooldown(&root.path);
                                    ui.horizontal(|ui| {
                                        ui.colored_label(
                                            ui.visuals().error_fg_color,
                                            if root.has_stale_error() {
                                                match &snapshot_age {
                                                    Some(age) => format!(
                                                        "刷新失败，显示 {age} 快照：{error}"
                                                    ),
                                                    None => format!(
                                                        "刷新失败，显示上次结果：{error}"
                                                    ),
                                                }
                                            } else {
                                                format!("无法读取目录：{error}")
                                            },
                                        );
                                        let retryable = root.load_error_retryable();
                                        let retry_label = retry_cooldown.map_or_else(
                                            || {
                                                if retryable {
                                                    "重试".to_string()
                                                } else {
                                                    "重新验证".to_string()
                                                }
                                            },
                                            |remaining| {
                                                format!(
                                                    "重试（后台冷却 {}s）",
                                                    remaining.as_secs().max(1)
                                                )
                                            },
                                        );
                                        if ui
                                            .small_button(retry_label)
                                            .on_hover_text(if retryable {
                                                "显式重试可旁路一次后台冷却；已有内容会保留"
                                            } else {
                                                "路径、权限或协议故障：修复后重新验证"
                                            })
                                            .clicked()
                                        {
                                            retry_path = Some(root.path.clone());
                                        }
                                    });
                                }
                                for child in root.visible_children() {
                                    Self::draw_tree_node(
                                        ui,
                                        child,
                                        &self.sidebar.selection,
                                        &mut toggle_path,
                                        &mut select_action,
                                        &mut cd_path,
                                        &mut retry_path,
                                        &mut show_more_path,
                                        &mut fs_menu_action,
                                        paste_state,
                                        &mut tree_row_rects,
                                        &mut selection_apply,
                                        &mut files_popup_open,
                                        &mut tree_has_focus,
                                    );
                                }
                                let remaining = root.remaining_children();
                                if remaining > 0
                                    && ui
                                        .button(format!("显示更多（剩余 {remaining} 项）"))
                                        .clicked()
                                {
                                    show_more_path = Some(root.path.clone());
                                }
                                if root.entries_truncated() {
                                    ui.colored_label(
                                        ui.visuals().warn_fg_color,
                                        format!(
                                            "目录过大：仅显示前 {} 项",
                                            sidebar::MAX_DIRECTORY_ENTRIES
                                        ),
                                    );
                                }
                            }
                        });
                    }
                }
            });

        // File-browser shortcuts require actual focus on a tree row as well as
        // pointer scope. Merely hovering Files must not steal keys from the PTY,
        // and a focused filter/path editor keeps ownership of its input.
        let pointer_over_files = root_ui
            .ctx()
            .pointer_hover_pos()
            .is_some_and(|pointer| panel_response.response.rect.contains(pointer));
        let files_keyboard_enabled = Self::files_tree_keyboard_enabled(
            self.sidebar.view,
            pointer_over_files,
            tree_has_focus,
            filter_has_focus || path_entry_has_focus,
            files_popup_open,
        );
        let files_f5_pressed = if files_keyboard_enabled {
            root_ui
                .ctx()
                .input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::F5))
        } else {
            false
        };
        if Self::files_panel_refresh_shortcut(
            self.sidebar.view,
            pointer_over_files,
            tree_has_focus,
            filter_has_focus || path_entry_has_focus,
            files_popup_open,
            files_f5_pressed,
        ) {
            do_refresh = true;
        }
        if self.sidebar.view == sidebar::SidebarView::Files
            && files_keyboard_enabled
            && self.sidebar.location().is_remote()
        {
            let (alt_up, alt_home, alt_left, alt_right, ctrl_l) =
                root_ui.ctx().input_mut(|input| {
                    (
                        input.consume_key(egui::Modifiers::ALT, egui::Key::ArrowUp),
                        input.consume_key(egui::Modifiers::ALT, egui::Key::Home),
                        input.consume_key(egui::Modifiers::ALT, egui::Key::ArrowLeft),
                        input.consume_key(egui::Modifiers::ALT, egui::Key::ArrowRight),
                        input.consume_key(egui::Modifiers::CTRL, egui::Key::L),
                    )
                });
            if alt_up && cd_path.is_none() {
                cd_path = self.sidebar.parent_dir();
            }
            if alt_home && cd_path.is_none() {
                cd_path = self.sidebar.home_dir().map(std::path::Path::to_path_buf);
            }
            if alt_left {
                navigate_back = true;
            }
            if alt_right {
                navigate_forward = true;
            }
            if ctrl_l {
                open_path_entry = true;
            }
        }
        if files_keyboard_enabled {
            let (up, down, left, right, enter) = root_ui.ctx().input_mut(|input| {
                (
                    input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    input.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            let selected_index = self.sidebar.selected_path.as_ref().and_then(|selected| {
                tree_row_rects
                    .iter()
                    .position(|(_, path, _)| path == selected)
            });
            if up || down {
                let next = match (up, selected_index) {
                    (true, Some(index)) => index.saturating_sub(1),
                    (true, None) => tree_row_rects.len().saturating_sub(1),
                    (false, Some(index)) => index
                        .saturating_add(1)
                        .min(tree_row_rects.len().saturating_sub(1)),
                    (false, None) => 0,
                };
                if let Some((_, path, is_dir)) = tree_row_rects.get(next) {
                    select_action = Some((path.clone(), *is_dir, FsSelectMode::Single));
                }
            } else if left {
                if let Some(selected) = self.sidebar.selected_path.as_ref() {
                    if self.sidebar.directory_expanded(selected) == Some(true) {
                        toggle_path = Some(selected.clone());
                    } else if let Some(parent) = selected.parent() {
                        if let Some((_, path, is_dir)) = tree_row_rects
                            .iter()
                            .find(|(_, path, _)| path.as_path() == parent)
                        {
                            select_action = Some((path.clone(), *is_dir, FsSelectMode::Single));
                        }
                    }
                }
            } else if right {
                if let Some(selected) = self.sidebar.selected_path.as_ref() {
                    match self.sidebar.directory_expanded(selected) {
                        Some(false) => toggle_path = Some(selected.clone()),
                        Some(true) => {
                            if let Some(index) = selected_index {
                                if let Some((_, path, is_dir)) = tree_row_rects.get(index + 1) {
                                    if path.parent() == Some(selected.as_path()) {
                                        select_action =
                                            Some((path.clone(), *is_dir, FsSelectMode::Single));
                                    }
                                }
                            }
                        }
                        None => {}
                    }
                }
            } else if enter {
                if let Some(selected) = self.sidebar.selected_path.as_ref() {
                    if self.sidebar.directory_expanded(selected).is_some() {
                        cd_path = Some(selected.clone());
                    }
                }
            }
        }

        // 闭包结束，安全 mutate
        let files_user_intent = toggle_path.is_some()
            || show_more_path.is_some()
            || retry_path.is_some()
            || navigate_back
            || navigate_forward
            || open_path_entry
            || submit_path_entry
            || cancel_path_entry
            || selection_apply.is_some()
            || select_action.is_some()
            || cd_path.is_some()
            || do_refresh
            || view_changed
            || location_changed.is_some()
            || files_terminal_target.is_some()
            || fs_menu_action.is_some()
            || cancel_transfer
            || filter_interacted
            || show_hidden_changed.is_some()
            || ssh_files_follow::ongoing_files_surface_is_user_intent(
                files_popup_open,
                hover_files_active,
            );
        if files_user_intent {
            self.sidebar.note_files_user_intent();
        }
        if open_path_entry {
            self.sidebar.open_path_entry();
            root_ui.ctx().memory_mut(|memory| {
                memory.request_focus(egui::Id::new("remote-files-path-entry"))
            });
        }
        if cancel_path_entry {
            self.sidebar.cancel_path_entry();
        }
        if submit_path_entry {
            if let Some(error) = self.sidebar.submit_path_entry() {
                self.set_status_for(
                    format!("Remote Files 路径无效：{error}"),
                    Duration::from_secs(6),
                );
            }
        }
        if navigate_back {
            if let Some(error) = self.sidebar.navigate_back() {
                self.set_status_for(
                    format!("Remote Files 后退失败：{error}"),
                    Duration::from_secs(5),
                );
            }
        } else if navigate_forward {
            if let Some(error) = self.sidebar.navigate_forward() {
                self.set_status_for(
                    format!("Remote Files 前进失败：{error}"),
                    Duration::from_secs(5),
                );
            }
        }
        self.execute_pending_command_sidebar_action();
        self.execute_pending_task_sidebar_action();
        if let Some(p) = toggle_path {
            if let Some(error) = self.sidebar.toggle_node(&p) {
                self.set_status_for(format!("文件树读取失败：{error}"), Duration::from_secs(5));
            }
        }
        if let Some(p) = retry_path {
            if let Some(error) = self.sidebar.retry_node(&p) {
                self.set_status_for(format!("文件树重试失败：{error}"), Duration::from_secs(5));
            }
        }
        if let Some(p) = show_more_path {
            self.sidebar.show_more(&p);
        }
        if let Some(selection) = selection_apply {
            // 右键点在选中集之外：选中集收缩为该行（锚点一并跟过去）。
            self.sidebar.selection = selection;
            self.sidebar.selected_path = self.sidebar.selection.keys().next().cloned();
        }
        if let Some((p, is_dir, mode)) = select_action {
            match mode {
                FsSelectMode::Single => self.sidebar.select_single(&p, is_dir),
                FsSelectMode::Toggle => self.sidebar.select_toggle(&p, is_dir),
                FsSelectMode::Range => {
                    // 范围选择的"可见行序"就是本帧渲染出的行（含根目录行）。
                    let row_order: Vec<(std::path::PathBuf, bool)> = tree_row_rects
                        .iter()
                        .map(|(_, path, is_dir)| (path.clone(), *is_dir))
                        .collect();
                    self.sidebar.select_range(&row_order, &p, is_dir);
                }
            }
        }
        if let Some(p) = cd_path.take() {
            if self.sidebar.location().is_remote() {
                if let Some(error) = self.sidebar.set_current_dir(p.clone()) {
                    self.set_status_for(
                        format!("Remote Files 导航失败：{error}"),
                        Duration::from_secs(5),
                    );
                } else {
                    self.set_status_for(
                        format!("Remote Files 正在验证：{}", p.display()),
                        Duration::from_secs(3),
                    );
                }
                // Remote Files is an independent execution endpoint. Never
                // inject its path into an unrelated local/SSH terminal PTY.
            } else {
                cd_path = Some(p);
            }
        }
        if let Some(p) = cd_path {
            let quoted = jterm_core::process::shell_quote_path(&p.to_string_lossy());
            let cmd = format!("cd {}\n", quoted);
            let active_session_id = self
                .session_manager
                .sessions()
                .get(self.session_manager.active_index())
                .map(|session| session.metadata.session_id.clone());
            let direct_input_blocked = active_session_id
                .as_deref()
                .is_none_or(|session_id| self.direct_input_is_blocked_for_session(session_id));
            let paste_result = {
                let session = self.session_manager.get_active_session_mut();
                paste_text_into_session(
                    session,
                    cmd,
                    self.config.paste_confirm,
                    PasteOrigin::PromptInsert,
                    true,
                    direct_input_blocked,
                    &mut self.pending_paste_confirm,
                )
            };
            match paste_result {
                Ok(true) if self.pending_paste_confirm.is_some() => {
                    self.status_message =
                        "请确认目录切换命令；文件树将在 shell 切换后同步".to_string();
                    self.status_expires_at =
                        Some(std::time::Instant::now() + Duration::from_secs(4));
                }
                Ok(true) => {
                    if let Some(session_id) = active_session_id {
                        self.clear_block_selection_for_session(&session_id);
                    }
                    self.status_message =
                        "目录切换命令已发送；文件树将跟随 shell 工作目录".to_string();
                    self.status_expires_at =
                        Some(std::time::Instant::now() + Duration::from_secs(4));
                }
                Ok(false) => {
                    self.status_message = "目录切换命令为空，未发送".to_string();
                    self.status_expires_at =
                        Some(std::time::Instant::now() + Duration::from_secs(4));
                }
                Err(error) => {
                    self.status_message = format!("目录切换命令发送失败：{error}");
                    self.status_expires_at =
                        Some(std::time::Instant::now() + Duration::from_secs(4));
                }
            }
        }
        if do_refresh {
            if let Some(error) = self.sidebar.refresh() {
                self.set_status_for(format!("文件树刷新失败：{error}"), Duration::from_secs(5));
            }
        }
        if let Some(show_hidden) = show_hidden_changed {
            if let Some(error) = self.sidebar.set_show_hidden(show_hidden) {
                self.set_status_for(
                    format!("隐藏文件策略切换失败：{error}"),
                    Duration::from_secs(5),
                );
            }
        }
        if retry_ssh_files {
            self.ssh_files_follow.request_retry();
        }
        if let Some(location) = location_changed {
            if let Some(error) = self.sidebar.set_location(location) {
                self.set_status_for(format!("切换浏览位置失败:{error}"), Duration::from_secs(5));
            }
        }
        if let Some(target) = files_terminal_target {
            match target {
                sidebar::FilesTerminalTarget::Local(path) => {
                    self.open_local_terminal_at_sidebar_root(&path);
                }
                sidebar::FilesTerminalTarget::Remote { index, overlay } => {
                    self.connect_files_remote_host(index, overlay);
                }
                sidebar::FilesTerminalTarget::Transient { host, overlay } => {
                    self.connect_transient_remote_host(host, overlay);
                }
            }
        }
        if let Some(action) = fs_menu_action {
            self.apply_fs_menu_action(action, fs_intent_context);
        }
        if cancel_transfer && self.sidebar.cancel_transfers() > 0 {
            self.set_status("正在取消传输…");
        }
        if view_changed {
            // 记住用户选择的视图，下次默认沿用。
            self.config.sidebar_view = self.sidebar.view;
            self.schedule_config_save();
            if self.sidebar.view == sidebar::SidebarView::Files {
                if let Some(error) = self.sidebar.refresh() {
                    self.set_status_for(format!("文件树刷新失败：{error}"), Duration::from_secs(5));
                }
            }
        }

        // 拖拽导入：面板矩形 + 行命中测试 + 悬停提示 + 落下分派。
        let panel_rect = panel_response.response.rect;
        self.sidebar_drop_rect = (self.sidebar.view == sidebar::SidebarView::Files
            && !self.sidebar.current_dir.as_os_str().is_empty())
        .then_some(panel_rect);
        let drop_root = (!self.sidebar.current_dir.as_os_str().is_empty())
            .then_some(self.sidebar.current_dir.as_path());
        if hover_files_active && self.sidebar_drop_rect.is_some() {
            if let Some(pointer) = pointer_pos {
                if let Some(target) =
                    Self::resolve_drop_target(pointer, panel_rect, &tree_row_rects, drop_root)
                {
                    // 高亮命中的行 + 指针旁的悬停提示。
                    if let Some((rect, _, _)) = tree_row_rects
                        .iter()
                        .find(|(rect, _, _)| rect.contains(pointer))
                    {
                        root_ui.painter().rect_stroke(
                            *rect,
                            2.0,
                            egui::Stroke::new(1.5, root_ui.visuals().selection.stroke.color),
                            egui::StrokeKind::Inside,
                        );
                    }
                    let hint = match self.sidebar.location() {
                        remote_fs::FsLocation::Local => {
                            format!("松开以导入到 {}", target.display())
                        }
                        location => format!(
                            "松开以上传到 {} 的 {}",
                            location.label(&self.config.remote_hosts),
                            target.display()
                        ),
                    };
                    egui::Area::new(egui::Id::new("sidebar-drop-hint"))
                        .order(egui::Order::Foreground)
                        .interactable(false)
                        .fixed_pos(pointer + egui::vec2(12.0, 12.0))
                        .show(root_ui.ctx(), |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                ui.label(hint);
                            });
                        });
                }
            }
        }
        if !dropped_paths.is_empty() && self.sidebar_drop_rect.is_some() {
            if let Some(pointer) = pointer_pos {
                if let Some(target_dir) =
                    Self::resolve_drop_target(pointer, panel_rect, &tree_row_rects, drop_root)
                {
                    self.sidebar.note_files_user_intent();
                    match sidebar::plan_drop(&dropped_paths, &target_dir, self.sidebar.location()) {
                        Ok(plan) => self.execute_drop_plan(plan),
                        Err(reason) => self.set_status_for(reason, Duration::from_secs(5)),
                    }
                }
            }
        }
        if self.sidebar.has_pending_scan() || self.sidebar.has_pending_op() {
            root_ui
                .ctx()
                .request_repaint_after(Duration::from_millis(50));
        }
    }

    /// 拖放落点解析：指针在某行 → 目录行是它自己、文件行是它的父目录；
    /// 在面板空白处 → 当前根目录；面板外 → None（保持终端的拖放行为）。
    fn files_panel_refresh_shortcut(
        view: sidebar::SidebarView,
        panel_hovered: bool,
        tree_has_focus: bool,
        text_input_has_focus: bool,
        popup_open: bool,
        f5_pressed: bool,
    ) -> bool {
        Self::files_tree_keyboard_enabled(
            view,
            panel_hovered,
            tree_has_focus,
            text_input_has_focus,
            popup_open,
        ) && f5_pressed
    }

    fn files_tree_keyboard_enabled(
        view: sidebar::SidebarView,
        panel_hovered: bool,
        tree_has_focus: bool,
        text_input_has_focus: bool,
        popup_open: bool,
    ) -> bool {
        view == sidebar::SidebarView::Files
            && panel_hovered
            && tree_has_focus
            && !text_input_has_focus
            && !popup_open
    }

    fn resolve_drop_target(
        pointer: egui::Pos2,
        panel_rect: egui::Rect,
        rows: &[(egui::Rect, std::path::PathBuf, bool)],
        root: Option<&std::path::Path>,
    ) -> Option<std::path::PathBuf> {
        if !panel_rect.contains(pointer) {
            return None;
        }
        // 行矩形互不重叠，直接取命中的那一行。
        for (rect, path, is_dir) in rows {
            if rect.contains(pointer) {
                return Some(if *is_dir {
                    path.clone()
                } else {
                    path.parent()
                        .map(std::path::Path::to_path_buf)
                        .unwrap_or_else(|| path.clone())
                });
            }
        }
        root.map(std::path::Path::to_path_buf)
    }

    /// 执行拖放导入计划：Local → 递归复制 op；Remote → 传输上传
    /// （进度/取消/状态与粘贴一致，每项的完成都会刷新落点父目录）。
    fn execute_drop_plan(&mut self, plan: sidebar::DropPlan) {
        let mut dispatch_errors = 0usize;
        let item_count = plan.items.len();
        for item in plan.items {
            let error = match item {
                sidebar::DropPlanItem::Copy { src, dst, .. } => self
                    .sidebar
                    .request_fs_op(sidebar::FsOpKind::Copy { src, dst }, false),
                sidebar::DropPlanItem::Upload {
                    src,
                    dst_dir,
                    is_dir,
                } => self.sidebar.request_transfer(
                    sidebar::FsTransfer {
                        src_endpoint: remote_fs::FsEndpointSnapshot::new(
                            remote_fs::FsLocation::Local,
                            remote_fs::SshExecutionOverlay::default(),
                        ),
                        src,
                        src_is_dir: is_dir,
                        dst_endpoint: remote_fs::FsEndpointSnapshot::new(
                            self.sidebar.location().clone(),
                            self.sidebar.execution_overlay().clone(),
                        ),
                        dst_dir,
                        cut: false,
                    },
                    false,
                ),
            };
            if error.is_some() {
                dispatch_errors += 1;
            }
        }
        let mut parts = vec![format!(
            "开始导入 {item_count} 项（{}）",
            remote_fs::format_bytes(plan.total_bytes)
        )];
        if !plan.refused_existing.is_empty() {
            parts.push(format!(
                "{} 项因目标已存在被跳过",
                plan.refused_existing.len()
            ));
        }
        if dispatch_errors > 0 {
            parts.push(format!("{dispatch_errors} 项分派失败"));
        }
        self.set_status_for(parts.join("；"), Duration::from_secs(5));
    }

    /// 递归绘制文件树节点（关联函数，不持 &self 以避免借用冲突）。
    /// rows 收集本帧每行的矩形（拖放落点命中测试用）；selection 是多选集；
    /// 修饰键（ctrl/shift）按下时点击只改选中集（不展开/不 cd）。
    #[allow(clippy::too_many_arguments)]
    fn draw_tree_node(
        ui: &mut egui::Ui,
        node: &sidebar::FileTreeNode,
        selection: &std::collections::BTreeMap<std::path::PathBuf, bool>,
        toggle: &mut Option<std::path::PathBuf>,
        select: &mut Option<(std::path::PathBuf, bool, FsSelectMode)>,
        cd: &mut Option<std::path::PathBuf>,
        retry: &mut Option<std::path::PathBuf>,
        show_more: &mut Option<std::path::PathBuf>,
        menu: &mut Option<FsMenuAction>,
        paste: FsPasteState,
        rows: &mut Vec<(egui::Rect, std::path::PathBuf, bool)>,
        selection_apply: &mut Option<std::collections::BTreeMap<std::path::PathBuf, bool>>,
        files_popup_open: &mut bool,
        tree_has_focus: &mut bool,
    ) {
        let is_selected = selection.contains_key(&node.path);
        let modifiers = ui.input(|input| input.modifiers);
        let selection_only = modifiers.ctrl || modifiers.shift;
        let select_mode = if modifiers.ctrl {
            FsSelectMode::Toggle
        } else if modifiers.shift {
            FsSelectMode::Range
        } else {
            FsSelectMode::Single
        };
        if node.is_dir {
            let arrow = if node.expanded { "▼" } else { "▶" };
            let label = format!("{} {}/", arrow, node.name);
            let resp = ui.selectable_label(is_selected, label);
            *tree_has_focus |= resp.has_focus();
            rows.push((resp.rect, node.path.clone(), true));
            if resp.clicked() {
                if selection_only {
                    *select = Some((node.path.clone(), true, select_mode));
                } else {
                    *toggle = Some(node.path.clone());
                    *select = Some((node.path.clone(), true, FsSelectMode::Single));
                }
            }
            if resp.double_clicked() && !selection_only {
                *cd = Some(node.path.clone());
            }
            // 右键点在选中集之外：选中集先收缩为该行（菜单目标随之只有它）。
            if resp.secondary_clicked() && !selection.contains_key(&node.path) {
                *selection_apply = Some(std::collections::BTreeMap::from([(
                    node.path.clone(),
                    node.is_dir,
                )]));
            }
            resp.context_menu(|ui| {
                let (_, targets) =
                    sidebar::Sidebar::resolve_menu_targets(selection, &node.path, node.is_dir);
                Self::fs_context_menu(
                    ui,
                    Some((&node.path, node.is_dir)),
                    &targets,
                    &node.path,
                    menu,
                    paste,
                );
            });
            *files_popup_open |= resp.context_menu_opened();
            resp.on_hover_text("单击展开/折叠，双击进入目录；ctrl/shift 点击多选");
            if node.expanded {
                ui.indent(node.path.to_string_lossy(), |ui| {
                    let snapshot_age = node.snapshot_age().map(snapshot_age_label);
                    if node.is_refreshing() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(match &snapshot_age {
                                Some(age) => format!("正在刷新…（显示 {age} 快照）"),
                                None => "正在刷新…（显示上次结果）".to_string(),
                            });
                        });
                    } else if node.is_loading() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("正在读取…");
                        });
                    } else if let Some(error) = node.load_error() {
                        ui.horizontal(|ui| {
                            ui.colored_label(
                                ui.visuals().error_fg_color,
                                if node.has_stale_error() {
                                    match &snapshot_age {
                                        Some(age) => {
                                            format!("刷新失败，显示 {age} 快照：{error}")
                                        }
                                        None => {
                                            format!("刷新失败，显示上次结果：{error}")
                                        }
                                    }
                                } else {
                                    format!("无法读取：{error}")
                                },
                            );
                            let retryable = node.load_error_retryable();
                            if ui
                                .small_button(if retryable { "重试" } else { "重新验证" })
                                .on_hover_text(if retryable {
                                    "可重试故障：重新读取此目录，已有内容会保留"
                                } else {
                                    "路径、权限或协议故障：修复后重新验证"
                                })
                                .clicked()
                            {
                                *retry = Some(node.path.clone());
                            }
                        });
                    }
                    for child in node.visible_children() {
                        Self::draw_tree_node(
                            ui,
                            child,
                            selection,
                            toggle,
                            select,
                            cd,
                            retry,
                            show_more,
                            menu,
                            paste,
                            rows,
                            selection_apply,
                            files_popup_open,
                            tree_has_focus,
                        );
                    }
                    let remaining = node.remaining_children();
                    if remaining > 0
                        && ui
                            .button(format!("显示更多（剩余 {remaining} 项）"))
                            .clicked()
                    {
                        *show_more = Some(node.path.clone());
                    }
                    if node.entries_truncated() {
                        ui.colored_label(
                            ui.visuals().warn_fg_color,
                            format!("目录过大：仅显示前 {} 项", sidebar::MAX_DIRECTORY_ENTRIES),
                        );
                    }
                });
            }
        } else {
            let resp = ui.selectable_label(is_selected, format!("  {}", node.name));
            *tree_has_focus |= resp.has_focus();
            rows.push((resp.rect, node.path.clone(), false));
            if resp.clicked() {
                *select = Some((node.path.clone(), false, select_mode));
            }
            if resp.secondary_clicked() && !selection.contains_key(&node.path) {
                *selection_apply = Some(std::collections::BTreeMap::from([(
                    node.path.clone(),
                    node.is_dir,
                )]));
            }
            // 文件行的"新建/粘贴/刷新"作用于它所在的目录。
            let target_dir = node
                .path
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| node.path.clone());
            resp.context_menu(|ui| {
                let (_, targets) =
                    sidebar::Sidebar::resolve_menu_targets(selection, &node.path, node.is_dir);
                Self::fs_context_menu(
                    ui,
                    Some((&node.path, node.is_dir)),
                    &targets,
                    &target_dir,
                    menu,
                    paste,
                );
            });
            *files_popup_open |= resp.context_menu_opened();
        }
    }

    /// 文件树右键菜单（目录行/文件行/根目录行共用）。只收集动作、不做任何
    /// mutate：实际执行在树遍历闭包结束后由 apply_fs_menu_action 完成。
    /// `entry` 是被右键的条目自身（Rename 的目标），根目录行传 None ——
    /// 不允许对浏览根做删除/重命名这类动作。`targets` 是批量动作
    /// （Delete/Copy/Cut/复制路径）的目标集：点在选中集内时为整个选中集。
    fn fs_context_menu(
        ui: &mut egui::Ui,
        entry: Option<(&std::path::Path, bool)>,
        targets: &[(std::path::PathBuf, bool)],
        target_dir: &std::path::Path,
        menu: &mut Option<FsMenuAction>,
        paste: FsPasteState,
    ) {
        let multi = targets.len() > 1;
        // 多选下新建/重命名没有意义（都是单目标操作）。
        if ui
            .add_enabled(!multi, egui::Button::new("New File"))
            .clicked()
        {
            *menu = Some(FsMenuAction::NewFile(target_dir.to_path_buf()));
            ui.close();
        }
        if ui
            .add_enabled(!multi, egui::Button::new("New Folder"))
            .clicked()
        {
            *menu = Some(FsMenuAction::NewFolder(target_dir.to_path_buf()));
            ui.close();
        }
        if let Some((path, _)) = entry {
            if ui
                .add_enabled(targets.len() == 1, egui::Button::new("Rename"))
                .clicked()
            {
                *menu = Some(FsMenuAction::Rename(path.to_path_buf()));
                ui.close();
            }
            let delete_label = if multi {
                format!("删除 {} 项", targets.len())
            } else {
                "Delete".to_string()
            };
            if ui.button(delete_label).clicked() {
                *menu = Some(FsMenuAction::Delete {
                    paths: targets.to_vec(),
                });
                ui.close();
            }
            ui.separator();
            if ui.button("Copy").clicked() {
                *menu = Some(FsMenuAction::Copy {
                    paths: targets.to_vec(),
                });
                ui.close();
            }
            if ui.button("Cut").clicked() {
                *menu = Some(FsMenuAction::Cut {
                    paths: targets.to_vec(),
                });
                ui.close();
            }
        }
        let paste_label = match paste {
            FsPasteState::Download => "Paste（下载）",
            FsPasteState::Upload => "Paste（上传）",
            FsPasteState::Relay => "Paste（中转）",
            FsPasteState::Ready | FsPasteState::Empty => "Paste",
        };
        let paste_button =
            ui.add_enabled(paste != FsPasteState::Empty, egui::Button::new(paste_label));
        match paste {
            FsPasteState::Empty => {
                paste_button.on_disabled_hover_text("剪贴板为空：先 Copy 或 Cut");
            }
            _ => {
                if paste_button.clicked() {
                    *menu = Some(FsMenuAction::Paste(target_dir.to_path_buf()));
                    ui.close();
                }
            }
        }
        ui.separator();
        // 复制路径：多选时换行连接；本地与远程行都是完整路径文本（不带前缀）。
        // 复制动作是纯 UI 行为，当场完成；菜单动作只负责状态栏提示。
        let copy_paths: Vec<std::path::PathBuf> = if entry.is_some() {
            targets.iter().map(|(path, _)| path.clone()).collect()
        } else {
            vec![target_dir.to_path_buf()]
        };
        if ui.button("复制路径").clicked() {
            let payload = copy_paths
                .iter()
                .map(|path| Self::fs_copy_path_payload(path))
                .collect::<Vec<_>>()
                .join("\n");
            ui.ctx().copy_text(payload);
            *menu = Some(FsMenuAction::CopyPath(copy_paths));
            ui.close();
        }
        ui.separator();
        if ui.button("Refresh").clicked() {
            *menu = Some(FsMenuAction::Refresh(target_dir.to_path_buf()));
            ui.close();
        }
    }

    /// "复制路径"的剪贴板载荷：本地与远程行都是完整路径文本（远程不带前缀）。
    fn fs_copy_path_payload(path: &std::path::Path) -> String {
        path.to_string_lossy().into_owned()
    }

    /// 执行右键菜单收集到的动作（对话框/剪贴板/操作 worker 分派）。
    fn apply_fs_menu_action(&mut self, action: FsMenuAction, context: sidebar::FilesIntentContext) {
        if !self.sidebar.files_intent_is_current(&context) {
            self.set_status_for(
                "文件树位置已变化；已取消旧位置的操作",
                Duration::from_secs(5),
            );
            return;
        }
        let uses_stale_snapshot = match &action {
            FsMenuAction::NewFile(path)
            | FsMenuAction::NewFolder(path)
            | FsMenuAction::Rename(path)
            | FsMenuAction::Paste(path) => self.sidebar.path_uses_stale_snapshot(path),
            FsMenuAction::Delete { paths }
            | FsMenuAction::Copy { paths }
            | FsMenuAction::Cut { paths } => paths
                .iter()
                .any(|(path, _)| self.sidebar.path_uses_stale_snapshot(path)),
            // Retry/Refresh must remain available, and copying already-visible
            // text cannot mutate or later dereference the remote pathname.
            FsMenuAction::CopyPath(_) | FsMenuAction::Refresh(_) => false,
        };
        if uses_stale_snapshot {
            self.set_status_for(
                "该目录显示的是上次结果；请先重试刷新，再执行文件操作",
                Duration::from_secs(5),
            );
            return;
        }
        match action {
            FsMenuAction::NewFile(dir) => {
                self.sidebar_name_dialog = Some(FsNameDialog {
                    kind: FsNameDialogKind::NewFile,
                    base: dir,
                    input: String::new(),
                    error: None,
                    context,
                });
            }
            FsMenuAction::NewFolder(dir) => {
                self.sidebar_name_dialog = Some(FsNameDialog {
                    kind: FsNameDialogKind::NewFolder,
                    base: dir,
                    input: String::new(),
                    error: None,
                    context,
                });
            }
            FsMenuAction::Rename(src) => {
                let input = src
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_string();
                self.sidebar_name_dialog = Some(FsNameDialog {
                    kind: FsNameDialogKind::Rename,
                    base: src,
                    input,
                    error: None,
                    context,
                });
            }
            FsMenuAction::Delete { paths } => {
                let dir_count = paths.iter().filter(|(_, is_dir)| *is_dir).count();
                self.sidebar_delete_dialog = Some(FsDeleteDialog {
                    paths: paths.into_iter().map(|(path, _)| path).collect(),
                    dir_count,
                    context,
                });
            }
            FsMenuAction::Copy { paths } => self.set_fs_clipboard(paths, false),
            FsMenuAction::Cut { paths } => self.set_fs_clipboard(paths, true),
            FsMenuAction::Paste(target_dir) => self.paste_fs_clipboard(&target_dir),
            FsMenuAction::CopyPath(paths) => {
                if paths.len() == 1 {
                    self.set_status(format!("已复制路径：{}", paths[0].display()));
                } else {
                    self.set_status(format!("已复制 {} 个路径", paths.len()));
                }
            }
            FsMenuAction::Refresh(dir) => {
                if let Some(error) = self.sidebar.refresh_loaded_node(&dir) {
                    self.set_status_for(format!("文件树刷新失败：{error}"), Duration::from_secs(5));
                }
            }
        }
    }

    /// Copy/Cut 进文件剪贴板（支持多选批量）。
    fn set_fs_clipboard(&mut self, paths: Vec<(std::path::PathBuf, bool)>, cut: bool) {
        let count = paths.len();
        self.sidebar.set_clipboard(remote_fs::FsClipboard {
            loc: self.sidebar.location().clone(),
            overlay: self.sidebar.execution_overlay().clone(),
            items: paths
                .into_iter()
                .map(|(path, is_dir)| remote_fs::FsClipboardItem { path, is_dir })
                .collect(),
            cut,
        });
        self.set_status(match (count, cut) {
            (1, false) => "已复制到文件剪贴板".to_string(),
            (1, true) => "已剪切到文件剪贴板".to_string(),
            (_, false) => format!("已复制 {count} 项到文件剪贴板"),
            (_, true) => format!("已剪切 {count} 项到文件剪贴板"),
        });
    }

    /// 粘贴：目标目录 + 源文件名。单项保持既有路径（同位置 rename/copy；
    /// 跨位置流式传输带进度/取消）；多项走批量任务（逐项、跳过失败、汇总，
    /// 跨位置逐项复用 transfer）。sidebar 会冻结这次粘贴的 Copy/Cut
    /// intent token；完成时仍匹配才清空/收缩剪贴板。
    fn paste_fs_clipboard(&mut self, target_dir: &std::path::Path) {
        let Some(clipboard) = self.sidebar.clipboard.clone() else {
            return;
        };
        if clipboard.items.is_empty() {
            return;
        }
        let current = self.sidebar.location().clone();
        let same_namespace =
            remote_fs::same_files_namespace(&clipboard.loc, &current, &self.config.remote_hosts);
        if clipboard.items.len() > 1 {
            // 多项：一个批量任务（同位置逐项 rename/copy，跨位置逐项 transfer）。
            let batch = sidebar::BatchIntent::Paste {
                src_endpoint: Box::new(remote_fs::FsEndpointSnapshot::new(
                    clipboard.loc.clone(),
                    clipboard.overlay.clone(),
                )),
                dst_endpoint: Box::new(remote_fs::FsEndpointSnapshot::new(
                    current,
                    self.sidebar.execution_overlay().clone(),
                )),
                dst_dir: target_dir.to_path_buf(),
                items: clipboard
                    .items
                    .iter()
                    .map(|item| (item.path.clone(), item.is_dir))
                    .collect(),
                cut: clipboard.cut,
            };
            if let Some(error) = self.sidebar.request_batch(batch, clipboard.cut) {
                self.set_status_for(format!("粘贴失败:{error}"), Duration::from_secs(5));
            }
            return;
        }
        let item = clipboard.items[0].clone();
        if same_namespace {
            // 同位置：cut → rename，copy → copy（探针的 17/AlreadyExists 兜底）。
            let Some(dst) = item.paste_destination(target_dir) else {
                self.set_status("无法粘贴：源路径没有文件名");
                return;
            };
            if clipboard.cut && item.path == dst {
                self.set_status("源与目标相同，未移动");
                return;
            }
            let (kind, clear_clipboard_on_success) = if clipboard.cut {
                (
                    sidebar::FsOpKind::Rename {
                        src: item.path.clone(),
                        dst,
                    },
                    true,
                )
            } else {
                (
                    sidebar::FsOpKind::Copy {
                        src: item.path.clone(),
                        dst,
                    },
                    false,
                )
            };
            let overlay = remote_fs::same_namespace_execution_overlay(
                &clipboard.overlay,
                self.sidebar.execution_overlay(),
            )
            .clone();
            if let Some(error) =
                self.sidebar
                    .request_fs_op_with_overlay(kind, clear_clipboard_on_success, overlay)
            {
                self.set_status_for(format!("粘贴失败:{error}"), Duration::from_secs(5));
            }
            return;
        }
        // 跨位置：FsOpService 上的流式传输（字节帽见 remote_fs::MAX_TRANSFER_BYTES）。
        if item.paste_destination(target_dir).is_none() {
            self.set_status("无法粘贴：源路径没有文件名");
            return;
        }
        let direction = match (clipboard.loc.is_remote(), current.is_remote()) {
            (true, false) => "下载",
            (false, true) => "上传",
            _ => "中转",
        };
        let cut = clipboard.cut;
        let transfer = sidebar::FsTransfer {
            src_endpoint: remote_fs::FsEndpointSnapshot::new(
                clipboard.loc.clone(),
                clipboard.overlay.clone(),
            ),
            src: item.path.clone(),
            src_is_dir: item.is_dir,
            dst_endpoint: remote_fs::FsEndpointSnapshot::new(
                current,
                self.sidebar.execution_overlay().clone(),
            ),
            dst_dir: target_dir.to_path_buf(),
            cut,
        };
        if let Some(error) = self.sidebar.request_transfer(transfer, cut) {
            self.set_status_for(format!("{direction}失败:{error}"), Duration::from_secs(5));
        } else {
            self.set_status(format!(
                "已开始{direction}（后台进行，大文件可能需要几分钟）"
            ));
        }
    }

    /// 名称对话框提交：校验已在对话框内做过，这里负责组装操作并分派。
    fn submit_fs_name_dialog(&mut self, dialog: FsNameDialog) {
        if !self.sidebar.files_intent_is_current(&dialog.context) {
            self.set_status_for(
                "文件树位置已变化；未执行旧位置的文件操作",
                Duration::from_secs(5),
            );
            return;
        }
        let name = dialog.input.trim().to_string();
        let (kind, verb) = match dialog.kind {
            FsNameDialogKind::NewFile => (
                sidebar::FsOpKind::CreateFile(dialog.base.join(&name)),
                "新建文件",
            ),
            FsNameDialogKind::NewFolder => (
                sidebar::FsOpKind::CreateDir(dialog.base.join(&name)),
                "新建文件夹",
            ),
            FsNameDialogKind::Rename => {
                let dst = dialog.base.with_file_name(&name);
                if dst == dialog.base {
                    // 名字没变：静默关闭，不打扰用户也不发操作。
                    return;
                }
                (
                    sidebar::FsOpKind::Rename {
                        src: dialog.base.clone(),
                        dst,
                    },
                    "重命名",
                )
            }
        };
        if let Some(error) = self.sidebar.request_fs_op(kind, false) {
            self.set_status_for(format!("{verb}失败:{error}"), Duration::from_secs(5));
        }
    }

    /// 文件树的模态对话框：名称输入（New File / New Folder / Rename 共用）
    /// 与删除确认。浮动窗口，仿 remote_picker 的模式。
    pub fn render_sidebar_fs_dialogs(&mut self, ctx: &egui::Context) {
        let mut files_dialog_interacted = false;
        let stale_name = self
            .sidebar_name_dialog
            .as_ref()
            .is_some_and(|dialog| !self.sidebar.files_intent_is_current(&dialog.context));
        let stale_delete = self
            .sidebar_delete_dialog
            .as_ref()
            .is_some_and(|dialog| !self.sidebar.files_intent_is_current(&dialog.context));
        if stale_name {
            self.sidebar_name_dialog = None;
        }
        if stale_delete {
            self.sidebar_delete_dialog = None;
        }
        if stale_name || stale_delete {
            self.set_status_for(
                "文件树位置已变化；已关闭旧位置的文件操作",
                Duration::from_secs(5),
            );
        }

        let mut submitted_dialog: Option<FsNameDialog> = None;
        if let Some(dialog) = &mut self.sidebar_name_dialog {
            let mut open = true;
            let mut cancel = false;
            let title = match dialog.kind {
                FsNameDialogKind::NewFile => "New File",
                FsNameDialogKind::NewFolder => "New Folder",
                FsNameDialogKind::Rename => "Rename",
            };
            egui::Window::new(title)
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    match dialog.kind {
                        FsNameDialogKind::NewFile | FsNameDialogKind::NewFolder => {
                            ui.label(format!("在 {} 中创建：", dialog.base.display()));
                        }
                        FsNameDialogKind::Rename => {
                            ui.label(format!("重命名 {}：", dialog.base.display()));
                        }
                    }
                    let response = ui.text_edit_singleline(&mut dialog.input);
                    if response.changed() {
                        files_dialog_interacted = true;
                        dialog.error = None;
                    }
                    if let Some(error) = &dialog.error {
                        ui.colored_label(ui.visuals().error_fg_color, error.as_str());
                    }
                    let mut submitted = response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            files_dialog_interacted = true;
                            submitted = true;
                        }
                        if ui.button("Cancel").clicked() {
                            files_dialog_interacted = true;
                            cancel = true;
                        }
                    });
                    if submitted {
                        match remote_fs::validate_new_name(dialog.input.trim()) {
                            Ok(()) => submitted_dialog = Some(dialog.clone()),
                            Err(error) => dialog.error = Some(error),
                        }
                    }
                });
            if !open || cancel || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
                files_dialog_interacted = true;
                self.sidebar_name_dialog = None;
            }
        }
        if let Some(dialog) = submitted_dialog {
            files_dialog_interacted = true;
            self.sidebar_name_dialog = None;
            self.submit_fs_name_dialog(dialog);
        }

        let mut confirmed_delete: Option<FsDeleteDialog> = None;
        if let Some(dialog) = &self.sidebar_delete_dialog {
            let mut open = true;
            let mut cancel = false;
            egui::Window::new("Delete")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    if dialog.paths.len() == 1 {
                        ui.label("确定删除以下路径吗？");
                    } else {
                        ui.label(format!("确定删除以下 {} 项吗？", dialog.paths.len()));
                    }
                    for path in dialog.paths.iter().take(5) {
                        ui.label(egui::RichText::new(path.display().to_string()).monospace());
                    }
                    if dialog.paths.len() > 5 {
                        ui.label(format!("… 等 {} 项", dialog.paths.len()));
                    }
                    if dialog.dir_count > 0 {
                        ui.colored_label(
                            ui.visuals().warn_fg_color,
                            if dialog.paths.len() == 1 {
                                "这是一个目录，其中的全部内容都会被递归删除。".to_string()
                            } else {
                                format!(
                                    "其中包含 {} 个目录，它们的全部内容都会被递归删除。",
                                    dialog.dir_count
                                )
                            },
                        );
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Delete").clicked() {
                            files_dialog_interacted = true;
                            confirmed_delete = Some(dialog.clone());
                        }
                        if ui.button("Cancel").clicked() {
                            files_dialog_interacted = true;
                            cancel = true;
                        }
                    });
                });
            if !open || cancel || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
                files_dialog_interacted = true;
                self.sidebar_delete_dialog = None;
            }
        }
        if let Some(dialog) = confirmed_delete {
            files_dialog_interacted = true;
            self.sidebar_delete_dialog = None;
            if !self.sidebar.files_intent_is_current(&dialog.context) {
                self.sidebar.note_files_user_intent();
                self.set_status_for(
                    "文件树位置已变化；未执行旧位置的删除操作",
                    Duration::from_secs(5),
                );
                return;
            }
            let paths = dialog.paths;
            if paths.len() == 1 {
                let path = paths.into_iter().next().expect("len == 1");
                if let Some(error) = self
                    .sidebar
                    .request_fs_op(sidebar::FsOpKind::Delete(path), false)
                {
                    self.set_status_for(format!("删除失败:{error}"), Duration::from_secs(5));
                }
            } else {
                // 多选删除：一个批量任务逐项删除、跳过失败、汇总上报。
                let batch = sidebar::BatchIntent::Delete {
                    endpoint: Box::new(remote_fs::FsEndpointSnapshot::new(
                        self.sidebar.location().clone(),
                        self.sidebar.execution_overlay().clone(),
                    )),
                    items: paths,
                };
                if let Some(error) = self.sidebar.request_batch(batch, false) {
                    self.set_status_for(format!("删除失败:{error}"), Duration::from_secs(5));
                }
            }
        }
        if files_dialog_interacted {
            self.sidebar.note_files_user_intent();
        }
    }

    #[allow(deprecated)]
    fn render_ui(&mut self, root_ui: &mut egui::Ui, frame_pointer_input_blocked: bool) {
        // egui 0.35 起 Panel/CentralPanel 都改成在 Ui 上 .show(ui, ...) 调用;
        // 但仍有部分代码(浮窗 Window、各种 input/viewport 操作)需要 &Context,
        // 这里克隆一份作为局部 ctx 供下游使用(Arc 引用计数,几乎零成本)。
        let ctx_owned = root_ui.ctx().clone();
        let ctx = &ctx_owned;

        // Bars register their pane-promotion targets before CentralPanel runs.
        // They are frame-local geometry and must never survive a resize or a
        // hidden sidebar. A release outside the viewport may not reach egui,
        // so lost focus (and a no-button frame without a release edge) must
        // cancel either gesture just like Escape does.
        self.tab_bar_drop_rects.clear();
        let workspace_drag_in_flight =
            self.dragging_tab_session_id.is_some() || self.pane_drag.is_some();
        let (escape_pressed, pointer_cancelled, window_focused) = ctx.input(|input| {
            let released = input.pointer.any_released();
            (
                input.key_pressed(egui::Key::Escape),
                workspace_drag_pointer_cancelled(
                    input.pointer.primary_down(),
                    released,
                    input.pointer.has_pointer(),
                ),
                input.viewport().focused.unwrap_or(true),
            )
        });
        if workspace_drag_in_flight && (escape_pressed || pointer_cancelled || !window_focused) {
            self.clear_workspace_drag();
        }

        let frame = egui::Frame::NONE.inner_margin(0.0);

        // 顶部栏(全宽)：必须在 render_sidebar 之前声明，egui 会把先声明的面板
        // 分配到容器边缘的完整范围 —— 因此顶栏横跨整个窗口，侧边栏落在其下方，
        // 而不是侧边栏贯穿到顶部。
        let mut close_requested = false;
        egui::Panel::top("top_bar")
            .frame(egui::Frame::NONE)
            .resizable(false)
            .show(root_ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                // Top 模式：完整水平标签栏(含 ☰ 与位置切换控件)。
                // Sidebar 模式：精简顶栏(仅 ☰ 与位置切换控件)，标签在侧边栏内。
                if self.show_top_tab_bar() {
                    if self.render_tab_bar(ui, ctx) {
                        close_requested = true;
                    }
                } else {
                    self.render_sidebar_mode_top_bar(ui, ctx);
                }
            });
        if close_requested {
            return;
        }

        // ember prefers jsh as its shell, so it is worth noticing when the
        // machine has none or an old one. The row draws nothing until the
        // background check has something actionable to offer.
        if self.render_jsh_notice(root_ui) {
            self.install_or_update_jsh();
        }

        // 底部状态栏(全宽)：同顶栏一样在侧边栏之前声明，因此它横跨整个
        // 窗口底边，侧边栏落在顶栏与它之间。
        self.render_bottom_bar(root_ui);

        // 侧边栏：在顶栏之后声明，占据顶栏下方区域的左侧。
        self.render_sidebar(root_ui);

        egui::CentralPanel::default()
            .frame(frame)
            .show(root_ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                self.render_terminal_content(ui, ctx, frame_pointer_input_blocked);
            });

        self.render_floating_panels(ctx);
    }

    // adjust_frame_budget moved to app::rendering module
    // Config and session save methods moved to app::window module
}

impl eframe::App for TerminalApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // Fully transparent clear color to support window-level opacity
        [0.0, 0.0, 0.0, 0.0]
    }

    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        // 记录最近一次指针位置：落下的拖放帧不一定带 PointerMoved。
        for event in &raw_input.events {
            match event {
                egui::Event::PointerMoved(pos) => self.last_pointer_pos = Some(*pos),
                egui::Event::PointerGone => self.last_pointer_pos = None,
                _ => {}
            }
        }
        // Modal/search/settings text fields need egui's semantic clipboard
        // events. They must bypass the terminal-specific Ctrl+C/X/V rewrite.
        let ui_owns_clipboard = self.terminal_input_blocked(ctx);
        // 落在文件树面板上的拖放留给侧边栏（raw.dropped_files 原样保留给
        // egui 与本帧的 render_sidebar）；其余维持今天的行为：图片按 payload
        // 粘进终端。
        let drop_targets_sidebar = self
            .sidebar_drop_rect
            .zip(self.last_pointer_pos)
            .is_some_and(|(rect, pos)| rect.contains(pos));
        let dropped_paths = if drop_targets_sidebar {
            Vec::new()
        } else {
            std::mem::take(&mut raw_input.dropped_files)
                .into_iter()
                .map(|file| file.path().to_path_buf())
                .collect::<Vec<_>>()
        };
        if !dropped_paths.is_empty() {
            if ui_owns_clipboard {
                self.set_status("图片拖放已忽略：当前面板正在接收输入");
            } else {
                match image_drop::prompt_payload(&dropped_paths) {
                    Ok(payload) => {
                        let accepted = paste_text_into_session(
                            self.session_manager.get_active_session_mut(),
                            payload,
                            self.config.paste_confirm,
                            PasteOrigin::PromptInsert,
                            false,
                            false,
                            &mut self.pending_paste_confirm,
                        );
                        if let Err(error) = accepted {
                            self.set_status(format!("图片拖放失败：{error}"));
                        }
                    }
                    Err(error) => self.set_status(format!("图片拖放已拒绝：{error}")),
                }
            }
        }
        let preserve_paste_event = {
            let terminal = self
                .session_manager
                .get_active_session_mut()
                .terminal
                .lock();
            terminal.is_paste_events_enabled()
        };

        // Egui-winit swallows Ctrl+V press when clipboard has no text (e.g. image only),
        // leaving only V's release. Track real/semantic V presses across frames so only
        // a genuinely orphaned release restores the application-facing Ctrl+V event.
        if raw_input.focused && !ui_owns_clipboard {
            restore_missing_image_paste_key_event(&mut raw_input.events, &mut self.paste_key_state);
        } else {
            self.paste_key_state.reset();
        }

        // Event::Paste has no per-event modifiers. Recover Shift from V's
        // release when a whole Ctrl+Shift+V chord lands in one input batch;
        // the batch-level modifier snapshot may already be empty by now.
        // egui 0.36 moved that snapshot into the event stream: the last
        // Event::ModifiersChanged of the batch is the state at drain time.
        let batch_modifiers = raw_input
            .events
            .iter()
            .rev()
            .find_map(|event| match event {
                egui::Event::ModifiersChanged(modifiers) => Some(*modifiers),
                _ => None,
            })
            .unwrap_or_default();
        let shortcut_modifiers = semantic_paste_modifiers(&raw_input.events, batch_modifiers);

        // egui-winit turns Ctrl/Cmd+C/X/V into semantic clipboard events and skips the
        // corresponding Key press. Restore those as Key events so the terminal can receive
        // control bytes, while still preventing egui's default text-edit shortcut behavior.
        let restore_shortcuts = should_restore_terminal_shortcut_event(ctx, shortcut_modifiers);

        normalize_terminal_shortcut_events(
            &mut raw_input.events,
            shortcut_modifiers,
            restore_shortcuts,
            preserve_paste_event,
            ui_owns_clipboard,
        );
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // eframe 0.35 起将原来的 App::update 拆成了 `logic` 和 `ui` 两段:
        // 这里把整个 update 的逻辑迁到 ui 中。许多下游代码(viewport 命令、输入查询、重绘请求)
        // 仍需要 &Context,从 root_ui 上 clone 一份(Arc 引用计数,几乎零成本)即可,
        // 与 root_ui 的可变借用互不冲突。
        let ctx_owned = root_ui.ctx().clone();
        let ctx = &ctx_owned;
        // Capture before any command, panel, dialog, or pane interaction in
        // this frame. SSH follow is drained only after rendering, so a first
        // observation must still yield to an explicit Files intent that ran
        // earlier in the same frame.
        let frame_start_files_user_intent = self.sidebar.files_user_intent_generation();

        // 检查是否收到退出信号（SIGINT/SIGTERM）
        if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
            crate::debug_log!("[SIGNAL] Shutdown requested, exiting gracefully");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        self.debug_panel.record_frame();
        self.poll_task_creation(ctx);
        self.poll_native_agent_runtime(ctx);
        self.poll_session_export(ctx);

        // A stateful mouse edge admitted in an earlier frame is older than
        // every keyboard/IME event arriving now. Retry it before any session
        // gets a chance to flush user input. If capacity is still unavailable,
        // keep accepting bytes only into that session's bounded retry buffer
        // and hold a one-frame admission barrier for the captured writer.
        let mut retire_mouse_capture = false;
        let mut prior_mouse_write_error = None;
        let protocol_input_barriers = self.osc_paste_input_barriers.clone();
        let prior_mouse_control_result = self
            .terminal_mouse_capture
            .as_mut()
            .filter(|capture| capture.reported_to_app && !capture.pending_controls.is_empty())
            .map(|capture| flush_mouse_controls(capture, &protocol_input_barriers));
        if let Some(Err(error)) = prior_mouse_control_result {
            retire_mouse_capture = !error.is_backpressure();
            prior_mouse_write_error = Some(error);
        }
        if self
            .terminal_mouse_capture
            .as_ref()
            .is_some_and(mouse_capture_is_complete)
        {
            retire_mouse_capture = true;
        }
        if retire_mouse_capture {
            self.terminal_mouse_capture = None;
            self.last_terminal_mouse_motion = None;
        }
        let mouse_input_barrier_session_id = self
            .terminal_mouse_capture
            .as_ref()
            .filter(|capture| capture.reported_to_app && !capture.pending_controls.is_empty())
            .map(|capture| capture.session_id.clone());
        if let Some(error) = prior_mouse_write_error {
            self.set_status_for(format!("鼠标报告发送失败：{error}"), Duration::from_secs(3));
            if error.is_backpressure() {
                ctx.request_repaint_after(Duration::from_millis(10));
            }
        }

        // Render-time click navigation is tagged with its originating
        // terminal and staged now, before this frame's IME, Ctrl commands, or
        // keyboard bytes. A modal that already owned input when the frame
        // began discards the stale navigation instead.
        let initially_blocked =
            self.terminal_input_blocked(ctx) || self.active_terminal_is_read_only();
        let (has_cursor_move_input, cursor_move_retry_overflow) =
            self.stage_prior_cursor_moves(initially_blocked);

        // Keep every PTY moving, not just the focused tab. All visible panes
        // receive priority, while hidden tabs rotate fairly. Background
        // parsing consumes at most half of the global adaptive byte budget;
        // whatever it does not use remains available to the active session.
        let visible_sessions: Vec<usize> = self.layout().session_indices();
        let background_budget = if self.session_manager.len() > 1 {
            self.adaptive_frame_budget / 2
        } else {
            0
        };
        let background_parse_started = std::time::Instant::now();
        let mut background_pump = self.session_manager.pump_inactive_sessions(
            background_budget,
            &visible_sessions,
            mouse_input_barrier_session_id.as_deref(),
            &self.osc_paste_input_barriers,
        );
        let mut terminal_parse_time = background_parse_started.elapsed();
        let window_focused = ctx.input(|input| input.viewport().focused.unwrap_or(true));
        for (session_idx, completed) in background_pump.completed_command_events.drain(..) {
            if let Some(session) = self.session_manager.sessions().get(session_idx) {
                self.agent_panel
                    .handle_completed(&session.metadata.session_id, &completed);
                self.command_correction.handle_completed(
                    &self.config,
                    self.agent_panel.session_active(),
                    &session.metadata.session_id,
                    &completed,
                );
                self.git_strip_cache
                    .mark_command_finished(&session.metadata.session_id);
            }
            // A visible split pane is watched exactly when the window itself
            // has focus; a hidden tab is never watched.
            maybe_notify_long_command(
                &self.config,
                &mut self.notification_window_started,
                &mut self.notifications_in_window,
                &completed,
                window_focused && visible_sessions.contains(&session_idx),
            );
            record_command_history(&self.config, &completed);
            if let Err(error) = execution_journal::submit(completed) {
                log::warn!("jsh execution output journal queue rejected an event: {error:?}");
            }
        }
        for (session_idx, error) in background_pump.errors.drain(..) {
            log::warn!("background session {}: {}", session_idx + 1, error);
        }
        for (session_idx, requests) in background_pump.clipboard_requests.drain(..) {
            let response_tx = self.session_manager.protocol_response_sender(session_idx);
            if let Some(session) = self.session_manager.get_session_mut(session_idx) {
                if let Some(response_tx) = response_tx {
                    service_osc5522_clipboard_requests(
                        self.clipboard.is_some(),
                        &self.clipboard_request_in_flight,
                        Arc::clone(&session.terminal),
                        response_tx,
                        requests,
                    );
                }
            }
        }
        if self.config.osc52_clipboard_write {
            for (_session_idx, text) in background_pump.osc52_writes.drain(..) {
                enqueue_osc52_clipboard_write(
                    self.osc52_clipboard_write_tx.as_ref(),
                    &mut self.osc52_write_window_started,
                    &mut self.osc52_writes_in_window,
                    text,
                );
            }
        }
        if self.config.osc52_clipboard_read {
            for session_idx in background_pump.osc52_queries.drain(..) {
                let response_route = self
                    .session_manager
                    .protocol_response_sender(session_idx)
                    .zip(
                        self.session_manager
                            .sessions()
                            .get(session_idx)
                            .map(|session| Arc::clone(&session.terminal)),
                    );
                if let Some((response_tx, terminal)) = response_route {
                    service_osc52_clipboard_query(
                        self.clipboard.is_some(),
                        &self.clipboard_request_in_flight,
                        terminal,
                        response_tx,
                        &mut self.osc52_read_window_started,
                        &mut self.osc52_reads_in_window,
                    );
                }
            }
        }
        for (_session_idx, title, body) in background_pump.notifications.drain(..) {
            show_desktop_notification(
                self.notification_tx.as_ref(),
                &mut self.notification_window_started,
                &mut self.notifications_in_window,
                title,
                body,
            );
        }
        // 按稳定 ID 而不是索引关闭:关掉一个会话会让它之后的索引整体左移,
        // 而关掉一个只剩一个窗格的 tab 还会连带关掉该 tab 的其他会话,索引
        // 可能往任意方向漂移。ID 查不到就说明它已经被前一次关闭带走了。
        // 在关闭 tab 前先把真实 child wait status 交给 TaskManager，否则
        // opaque Agent PTY 只能得到一个无意义的“消失了”状态。
        let exited_sessions: Vec<(String, Option<i32>)> = background_pump
            .exited_sessions
            .drain(..)
            .map(|exited| (exited.session_id, exited.exit_code))
            .collect();
        for (session_id, exit_code) in exited_sessions {
            let terminal_role = self.task_manager.terminal_role_for_session(&session_id);
            let task_terminal = self
                .task_manager
                .handle_terminal_session_exit(&session_id, exit_code)
                .is_some();
            let Some(session_idx) = self.session_manager.index_of(&session_id) else {
                continue;
            };
            if terminal_role == Some(crate::agent::TaskTerminalRole::Agent) {
                self.session_manager.retain_exited_command(&session_id);
                continue;
            }
            if task_terminal && self.session_manager.retain_exited_command(&session_id) {
                continue;
            }
            if self.session_manager.len() > 1 && session_idx != self.session_manager.active_index()
            {
                self.close_session_or_owning_tab(session_idx);
                self.schedule_session_save();
            }
        }
        let background_processed_bytes = background_pump.bytes_processed;
        let active_output_budget = self
            .adaptive_frame_budget
            .saturating_sub(background_processed_bytes)
            .max(1);
        let background_had_output = background_pump.had_output;
        let background_has_more = background_pump.has_more;

        // Collect events once per frame to avoid multiple clones
        self.frame_events.clear();
        ctx.input(|i| self.frame_events.extend(i.events.iter().cloned()));

        let mut ui_input_blocked = self.terminal_input_blocked(ctx);
        let terminal_input_blocked_at_frame_start =
            ui_input_blocked || self.active_terminal_is_read_only();
        let mut terminal_input_blocked = terminal_input_blocked_at_frame_start;
        let has_preedit = if terminal_input_blocked {
            // UI text fields and modal dialogs own IME/keyboard input while open.
            // In particular, an IME commit must never be mirrored into the PTY.
            self.session_manager
                .get_active_session_mut()
                .terminal
                .lock()
                .clear_preedit();
            false
        } else {
            self.handle_ime_events(ctx)
        };

        if !ui_input_blocked {
            self.handle_font_zoom(ctx);
        }

        // Step 2: 处理快捷键 - 使用可配置的快捷键系统。
        // 命令面板与帮助面板也通过 Command 派发，确保按键会被消费且
        // 帮助文案始终反映当前绑定。

        let (palette_requested_close, palette_owned_input) = self.handle_command_palette_input(ctx);
        if palette_requested_close {
            return;
        }

        // Block-search picker keys (Enter/Escape/arrows), routed like the
        // palette's so the overlay owns the whole frame's keyboard input.
        let block_search_owned_input = self.handle_block_search_input();
        // 历史命令选择器同理：浮层打开期间拥有整帧键盘输入。
        let history_picker_owned_input = self.handle_history_picker_input();
        // 工作流选择器与参数对话框同理。
        let workflow_picker_owned_input = self.handle_workflow_picker_input();
        let overlay_owned_input = palette_owned_input
            || block_search_owned_input
            || history_picker_owned_input
            || workflow_picker_owned_input;

        let (keybinding_requested_close, selection_postdates_terminal_input, accepted_ime_input) =
            self.handle_keybindings(
                ctx,
                ui_input_blocked || overlay_owned_input,
                terminal_input_blocked || overlay_owned_input,
            );
        if keybinding_requested_close {
            return;
        }

        // A command handled above may have opened or closed a modal. Re-evaluate
        // newly opened surfaces, but never release a frame that a UI surface
        // owned at its start: later events in the same OS batch must not escape
        // into the PTY after the modal-closing shortcut.
        ui_input_blocked = app::input::terminal_input_blocked_after_commands(
            ui_input_blocked,
            overlay_owned_input,
            self.terminal_input_blocked(ctx),
        );
        terminal_input_blocked = app::input::terminal_input_blocked_after_commands(
            terminal_input_blocked_at_frame_start,
            overlay_owned_input,
            self.terminal_input_blocked(ctx) || self.active_terminal_is_read_only(),
        );
        if terminal_input_blocked {
            let mut terminal = self
                .session_manager
                .get_active_session_mut()
                .terminal
                .lock();
            app::input::clear_terminal_preedit_for_ui_owner(&mut terminal, true);
        }

        // Route pointer input to the pane under the pointer before taking the
        // active-session borrow below. The renderer used to switch focus only
        // at the end of the frame, which sent a click (and mouse protocol
        // coordinates) to the previously focused PTY.
        let pointer_targets_terminal = !ui_input_blocked
            && !app::input::semantic_paste_precedes_mouse_input(&self.frame_events)
            && ctx.input(|input| {
                input.pointer.button_pressed(egui::PointerButton::Primary)
                    || input.pointer.button_pressed(egui::PointerButton::Secondary)
                    || input.pointer.button_pressed(egui::PointerButton::Middle)
                    || input
                        .events
                        .iter()
                        .any(|event| matches!(event, egui::Event::MouseWheel { .. }))
            });
        if pointer_targets_terminal && self.layout().panes().len() > 1 {
            if let Some(pos) =
                ctx.input(|input| input.pointer.interact_pos().or(input.pointer.hover_pos()))
            {
                let on_divider = self.layout().divider_at(pos).is_some();
                if !on_divider && self.layout_mut().focus_pane_at(pos).is_some() {
                    self.sync_active_session_to_focused_pane();
                }
            }
        }

        let active_session_idx = self.session_manager.active_index();
        let active_pane_renderer_idx = (self.layout().panes().len() > 1).then(|| {
            self.layout()
                .panes()
                .iter()
                .position(|pane| pane.id == self.layout().focused_pane_id)
        });
        let active_pane_renderer_idx = active_pane_renderer_idx.flatten();
        let active_terminal_content_rect = if let Some(index) = active_pane_renderer_idx {
            self.pane_renderers
                .get(index)
                .and_then(|renderer| renderer.last_content_rect)
        } else {
            self.renderer.last_content_rect
        };
        let pointer_over_active_terminal = ctx
            .input(|input| input.pointer.interact_pos().or(input.pointer.hover_pos()))
            .zip(active_terminal_content_rect)
            .is_some_and(|(pointer, rect)| rect.contains(pointer));

        // 获取当前活跃会话（在所有快捷键处理完后）
        let session_count_before = self.session_manager.len();
        let mut shell_exited = false;
        let mut shell_exit_observed = false;
        let mut shell_exit_code = None;
        let mut retain_exited_task_terminal = false;
        // A shell that dies before it could ever have shown a prompt is a
        // startup failure, not the user leaving. Closing the window on it
        // makes ember look like it "exits as soon as it runs", hiding the
        // real cause (bad `shell =` config, unusable cwd, wrong binary).
        let mut shell_startup_failed = false;

        // Step 2.5: 搜索面板事件处理
        if self.pending_paste_confirm.is_none() && !self.search_replace_panel.is_open {
            self.handle_search_panel_input();
        }

        if let Some(Err(error)) = self
            .session_manager
            .flush_protocol_responses(active_session_idx)
        {
            if !error.is_backpressure() {
                log::warn!("active protocol response queue stopped: {error}");
            }
        }
        let active_protocol_responses = self
            .session_manager
            .protocol_response_sender(active_session_idx)
            .expect("active session has an aligned protocol response queue");

        // Snapshot active session index before mutably borrowing session_manager;
        // we use it to tag pending-paste confirmations so they only deliver to
        // the same tab if the user hasn't switched away.
        let session = self.session_manager.get_active_session_mut();
        let active_session_id = session.metadata.session_id.clone();
        let active_task_terminal = self
            .task_manager
            .task_and_role_for_terminal_session(&active_session_id)
            .map(|(task, role)| (task.provider.display_name(), role));

        // Step 3: semantic application paste events. Host copy/paste keyboard
        // shortcuts are dispatched above through configurable commands.
        let events_copy =
            app::input::routed_terminal_events(&self.frame_events, terminal_input_blocked);
        let mut consumed_keys = std::collections::HashSet::new();

        let mut accepted_paste_input = false;
        let paste_confirmation_was_open = self.pending_paste_confirm.is_some();
        let mut semantic_paste_claims_rest = false;
        let mut rejected_mouse_prefix_allows_pointer = false;
        let saw_semantic_paste = events_copy.iter().any(|event| {
            if let egui::Event::Paste(_content) = event {
                crate::debug_log!(
                    "[EVENT] detected Paste event: {:?}",
                    if _content.is_empty() {
                        "empty"
                    } else {
                        "has content"
                    }
                );
                true
            } else {
                false
            }
        });

        if saw_semantic_paste {
            crate::debug_log!("[PASTE] ===== Semantic Paste triggered =====");
            if app::input::semantic_paste_has_mouse_prefix(&events_copy) {
                // Pointer input is dispatched later in the frame and cannot be
                // staged ahead of an asynchronous OSC 5522 notification. Keep
                // that older pointer input intact and reject this Paste. The
                // Paste and its keyboard/IME suffix remain claimed so retrying
                // cannot accidentally duplicate or submit them.
                semantic_paste_claims_rest = true;
                rejected_mouse_prefix_allows_pointer = true;
                self.status_message = "本帧已有更早的鼠标输入；请重试粘贴".to_string();
                self.status_expires_at = Some(std::time::Instant::now() + Duration::from_secs(3));
                consumed_keys.insert("PasteEvent".to_string());
            } else {
                let paste_events_enabled = {
                    let terminal = session.terminal.lock();
                    let paste_events_enabled = terminal.is_paste_events_enabled();
                    crate::debug_log!(
                        "[PASTE] terminal paste_events_enabled (mode 5522): {}",
                        paste_events_enabled
                    );
                    paste_events_enabled
                };

                if paste_events_enabled && self.clipboard.is_some() {
                    // An asynchronous paste notification may only establish its
                    // protocol barrier at a clean FIFO boundary. Otherwise an
                    // older Text/IME/key prefix (or pending input from a prior
                    // frame) would be forced behind the notification.
                    semantic_paste_claims_rest = true;
                    let route_blocked = crate::session_manager::user_input_flush_block(
                        &session.metadata.session_id,
                        mouse_input_barrier_session_id.as_deref(),
                        &self.osc_paste_input_barriers,
                        &active_protocol_responses,
                    )
                    .is_some();
                    let clean_route = app::input::osc_paste_route_is_clean(
                        !session.pending_input.is_empty(),
                        route_blocked,
                        &events_copy,
                    );
                    // MIME discovery is host clipboard I/O and build_paste_event
                    // replaces the terminal's single-use grant. Serialize it with
                    // OSC reads so concurrent Paste events cannot race tokens or
                    // create an unbounded helper/thread population.
                    if !clean_route {
                        self.status_message =
                            "终端仍有更早的输入；请在输入送达后重试粘贴".to_string();
                        self.status_expires_at =
                            Some(std::time::Instant::now() + Duration::from_secs(3));
                    } else if self
                        .clipboard_request_in_flight
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        crate::debug_log!(
                            "[PASTE] app supports paste events, building paste event"
                        );
                        let terminal = Arc::clone(&session.terminal);
                        let response_tx = active_protocol_responses.clone();
                        let in_flight = Arc::clone(&self.clipboard_request_in_flight);
                        let paste_input_barrier = self
                            .osc_paste_input_barriers
                            .acquire(active_session_id.clone());
                        let spawn_result = std::thread::Builder::new()
                            .name("paste-event-sender".to_string())
                            .spawn(move || {
                                let _guard = ClipboardRequestGuard(in_flight);
                                // Release only after enqueue_blocking returns. Until
                                // then this stable session's pending_input cannot
                                // overtake the asynchronous paste notification.
                                let _paste_input_barrier = paste_input_barrier;
                                let mime_types = ClipboardManager::new()
                                    .and_then(|clipboard| clipboard.available_mime_types())
                                    .unwrap_or_default();
                                crate::debug_log!("[PASTE] available MIME types: {:?}", mime_types);
                                let bytes = terminal.lock().build_paste_event(&mime_types);
                                crate::debug_log!(
                                    "[OSC5522] sending unsolicited paste MIME list ({} bytes)",
                                    bytes.len()
                                );
                                if let Err(error) = response_tx.enqueue_blocking(bytes) {
                                    log::debug!(
                                        "OSC 5522 unsolicited paste event cancelled: {error}"
                                    );
                                }
                            });
                        match spawn_result {
                            Ok(_) => {
                                accepted_paste_input = true;
                            }
                            Err(error) => {
                                self.clipboard_request_in_flight
                                    .store(false, Ordering::Release);
                                log::warn!("failed to spawn OSC 5522 paste event worker: {error}");
                                self.status_message = "剪贴板正忙，请重试粘贴".to_string();
                                self.status_expires_at =
                                    Some(std::time::Instant::now() + Duration::from_secs(3));
                            }
                        }
                    } else {
                        self.status_message = "剪贴板正忙，请稍后重试粘贴".to_string();
                        self.status_expires_at =
                            Some(std::time::Instant::now() + Duration::from_secs(3));
                    }
                    consumed_keys.insert("PasteEvent".to_string());
                } else {
                    if self.clipboard.is_none() {
                        crate::debug_log!("[PASTE] clipboard not available");
                    } else {
                        crate::debug_log!("[PASTE] app does NOT support paste events");
                    }
                    // 应用不支持粘贴事件协议，需要特殊处理不同类型的内容
                    crate::debug_log!(
                        "[PASTE] fallback: app doesn't support paste events, handling content directly"
                    );
                    if let Some(clipboard) = &self.clipboard {
                        if let Ok(content) = clipboard.paste_contents() {
                            match content {
                                ClipboardContent::Text(text) => {
                                    crate::debug_log!(
                                        "[PASTE] fallback: TEXT content ({} chars)",
                                        text.len()
                                    );
                                    match paste_text_into_session(
                                        session,
                                        text,
                                        self.config.paste_confirm,
                                        PasteOrigin::Clipboard,
                                        false,
                                        app::input::semantic_paste_direct_input_blocked(
                                            crate::session_manager::user_input_flush_block(
                                                &session.metadata.session_id,
                                                mouse_input_barrier_session_id.as_deref(),
                                                &self.osc_paste_input_barriers,
                                                &active_protocol_responses,
                                            )
                                            .is_some(),
                                            &events_copy,
                                        ),
                                        &mut self.pending_paste_confirm,
                                    ) {
                                        Ok(true) => {
                                            accepted_paste_input =
                                                self.pending_paste_confirm.is_none();
                                            consumed_keys.insert("PasteEvent".to_string());
                                        }
                                        Ok(false) => {
                                            crate::debug_log!("[PASTE] fallback: text is empty");
                                        }
                                        Err(error) => {
                                            self.status_message = format!("粘贴失败：{error}");
                                            self.status_expires_at = Some(
                                                std::time::Instant::now() + Duration::from_secs(4),
                                            );
                                        }
                                    }
                                }
                                ClipboardContent::Binary(_bytes) => {
                                    crate::debug_log!(
                                        "[PASTE] fallback: BINARY content ({} bytes)",
                                        _bytes.len()
                                    );
                                    crate::debug_log!(
                                        "[PASTE] refusing to send {} binary bytes as PTY input; app did not negotiate OSC 5522",
                                        _bytes.len()
                                    );
                                    self.status_message =
                                        "图像粘贴需要应用支持 OSC 5522".to_string();
                                    self.status_expires_at =
                                        Some(std::time::Instant::now() + Duration::from_secs(4));
                                    consumed_keys.insert("PasteEvent".to_string());
                                }
                            }
                        } else {
                            crate::debug_log!("[PASTE] fallback: failed to get clipboard content");
                        }
                    } else {
                        crate::debug_log!("[PASTE] fallback: clipboard not available");
                    }
                }
            }
            crate::debug_log!("[PASTE] ===== Semantic Paste finished =====");
        }

        if !paste_confirmation_was_open && self.pending_paste_confirm.is_some() {
            // Confirmation owns every event after the Paste that opened it,
            // including Enter and IME lifecycle/commit events.
            semantic_paste_claims_rest = true;
        }
        let semantic_paste_blocks_pointer = accepted_paste_input
            || (!paste_confirmation_was_open && self.pending_paste_confirm.is_some());
        let terminal_keyboard_events = app::input::terminal_events_before_semantic_paste_claim(
            &events_copy,
            semantic_paste_claims_rest,
        );
        if semantic_paste_claims_rest {
            terminal_input_blocked = true;
            app::input::clear_terminal_preedit_for_ui_owner(&mut session.terminal.lock(), true);
        }
        let mut terminal_pointer_input_blocked = app::input::semantic_paste_pointer_input_blocked(
            terminal_input_blocked,
            rejected_mouse_prefix_allows_pointer,
            semantic_paste_blocks_pointer,
        );
        // An exited task terminal is read-only at the PTY boundary, but its
        // local buffer must remain selectable and scrollable for review.
        terminal_pointer_input_blocked = app::input::retained_terminal_pointer_input_blocked(
            terminal_pointer_input_blocked,
            session.purpose == crate::session::SessionPurpose::RetainedCommand,
            ui_input_blocked,
            semantic_paste_blocks_pointer,
        );

        // Step 4: 处理普通键盘输入
        // 当搜索面板或配置面板打开时，不处理普通键盘输入（面板会处理输入）
        // 复用缓冲区减少内存分配
        self.keyboard_input_buffer.clear();
        if !terminal_input_blocked || semantic_paste_claims_rest {
            let (
                keyboard_enhancement_flags,
                report_all_keys_mode,
                xterm_modify_other_keys,
                xterm_format_other_keys,
                application_cursor_keys,
                alt_screen,
            ) = {
                let terminal = session.terminal.lock();
                (
                    terminal.keyboard_enhancement_flags(),
                    terminal.is_report_all_keys_enabled(),
                    terminal.xterm_modify_other_keys(),
                    terminal.xterm_format_other_keys(),
                    terminal.is_application_cursor_keys(),
                    terminal.is_alt_buffer_active(),
                )
            };
            // 转换 consumed_keys 为需要的格式（HashSet<&str>）
            let consumed_keys_refs: std::collections::HashSet<&str> =
                consumed_keys.iter().map(|s| s.as_str()).collect();
            let input_renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            input_renderer.handle_keyboard_input(
                ctx,
                &mut self.keyboard_input_buffer,
                &consumed_keys_refs,
                has_preedit,
                keyboard_enhancement_flags,
                report_all_keys_mode,
                xterm_modify_other_keys,
                xterm_format_other_keys,
                application_cursor_keys,
                alt_screen,
                &terminal_keyboard_events,
            );
        }

        let has_keyboard_input = !self.keyboard_input_buffer.is_empty();

        // The retry buffer is per-session and sent as one FIFO message. Do not
        // split arbitrary bytes into frame-sized chunks: terminal replies could
        // otherwise interleave inside a UTF-8/key escape/paste sequence.
        let mut terminal_write_error = None;
        let mut input_retry_overflow = cursor_move_retry_overflow;
        let mut keyboard_input_accepted = false;
        {
            if has_keyboard_input {
                keyboard_input_accepted = session.queue_input(&self.keyboard_input_buffer);
                input_retry_overflow |= !keyboard_input_accepted;
            }
            let user_input_flush_blocked = crate::session_manager::user_input_flush_block(
                &session.metadata.session_id,
                mouse_input_barrier_session_id.as_deref(),
                &self.osc_paste_input_barriers,
                &active_protocol_responses,
            )
            .is_some();
            if !user_input_flush_blocked && !session.pending_input.is_empty() {
                session.terminal.lock().scroll_to_bottom();
                session.projection_view_state.scroll_to_bottom();
                match session.shell.write(&session.pending_input) {
                    Ok(()) => session.pending_input.clear(),
                    Err(error) => {
                        if !error.is_backpressure() {
                            session.pending_input.clear();
                        }
                        terminal_write_error = Some(error);
                    }
                }
            }
        }
        let accepted_terminal_input =
            keyboard_input_accepted || accepted_ime_input || accepted_paste_input;

        // 本帧真正接受的终端输入（键盘、IME 或 paste）都遵循同一时序：
        // 只有输入之后又显式建立的新选区才能保留。Retry-buffer cap 拒绝
        // 是全-or-nothing，不得因为仅生成了 bytes 就误清 selection。
        if app::input::accepted_terminal_input_clears_block_selection(
            accepted_terminal_input,
            selection_postdates_terminal_input,
        ) {
            app::commands::clear_block_selection_state(
                &mut self.block_selection,
                &mut self.command_sidebar.selected,
            );
        }

        // 有输入活动时更新最后活动时间
        if accepted_terminal_input || has_cursor_move_input {
            self.last_activity_time = std::time::Instant::now();
        }
        if let Some(error) = terminal_write_error {
            if error.is_backpressure() {
                self.status_message = "终端输入繁忙，正在重试…".to_string();
                ctx.request_repaint_after(Duration::from_millis(10));
            } else {
                self.status_message = format!("终端输入失败：{error}");
            }
            self.status_expires_at = Some(std::time::Instant::now() + Duration::from_secs(3));
        }
        if input_retry_overflow {
            self.status_message = "终端输入重试缓冲区已满，新输入未发送".to_string();
            self.status_expires_at = Some(std::time::Instant::now() + Duration::from_secs(4));
        }
        if !session.pending_input.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(10));
        }

        // Force repaint if we have any keyboard/cursor input - ensures input renders immediately
        if accepted_terminal_input || has_cursor_move_input {
            ctx.request_repaint();
        }

        // Step 6: 处理 shell 事件
        // 关键：限制每帧处理的总字节数，防止大量 ANSI 数据阻塞 UI 线程导致假死。
        // 超出限制的数据保存到当前 session 自己的 pending_output，
        // 下一帧继续处理。绝不能放在 TerminalApp 全局缓冲中：帧间切 tab
        // 会把旧 session 的 ANSI 字节流喂给新 session，造成串屏和终端模式污染。
        // 使用自适应帧预算，根据帧时间动态调整
        let mut has_new_output = false;
        let max_bytes_per_frame = active_output_budget;
        let mut has_more_data = false;
        let mut active_processed_bytes = 0;

        // 先取回上一帧未处理完的数据
        let mut accumulated_data = std::mem::take(&mut session.pending_output);
        if !accumulated_data.is_empty() {
            has_new_output = true;
        }

        // 从 channel 中收集数据，直到达到字节上限
        let shell_events_are_live =
            session.purpose != crate::session::SessionPurpose::RetainedCommand;
        if shell_events_are_live && accumulated_data.len() < max_bytes_per_frame {
            loop {
                match session.shell.events().try_recv() {
                    Ok(ShellEvent::Output(data)) => {
                        accumulated_data.extend(data);
                        has_new_output = true;
                        if accumulated_data.len() >= max_bytes_per_frame {
                            has_more_data = true;
                            break;
                        }
                    }
                    Ok(ShellEvent::Exit(code)) => {
                        crate::debug_log!("[SHELL EXIT] shell exited with code: {}", code);
                        shell_exit_observed = true;
                        shell_exit_code = Some(code);
                        retain_exited_task_terminal = active_task_terminal.is_some();
                        let uptime = session.shell.uptime();
                        if let Some((provider, role)) = active_task_terminal {
                            self.status_message = match (role, code) {
                                (crate::agent::TaskTerminalRole::Agent, 0) => {
                                    format!("{provider} task finished")
                                }
                                (crate::agent::TaskTerminalRole::Agent, code) => {
                                    format!("{provider} task exited with code {code}")
                                }
                                (crate::agent::TaskTerminalRole::Validation, 0) => {
                                    "Task validation passed".to_string()
                                }
                                (crate::agent::TaskTerminalRole::Validation, code) => {
                                    format!("Task validation failed with code {code}")
                                }
                            };
                            self.status_expires_at =
                                Some(std::time::Instant::now() + Duration::from_secs(6));
                        } else if code != 0 && uptime < SHELL_STARTUP_GRACE {
                            shell_startup_failed = true;
                            self.status_message = format!(
                                "Shell failed to start (exit code {code}). Check the `shell` setting in the config panel."
                            );
                            self.status_expires_at = None;
                        } else {
                            self.status_message = format!("Shell exited with code: {}", code);
                            self.status_expires_at =
                                Some(std::time::Instant::now() + Duration::from_secs(6));
                        }
                        has_new_output = true;
                        shell_exited = true;
                        break;
                    }
                    Ok(ShellEvent::Error(e)) => {
                        self.status_message = format!("Error: {}", e);
                        self.status_expires_at =
                            Some(std::time::Instant::now() + Duration::from_secs(6));
                        has_new_output = true;
                        break;
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        shell_exit_observed = true;
                        retain_exited_task_terminal = active_task_terminal.is_some();
                        shell_exited = true;
                        break;
                    }
                }
            }
        } else if accumulated_data.len() >= max_bytes_per_frame {
            has_more_data = true;
        }

        // 如果累积数据超过帧限制，将多余部分保存到下一帧
        if accumulated_data.len() > max_bytes_per_frame {
            session.pending_output = accumulated_data.split_off(max_bytes_per_frame);
            has_more_data = true;
        }
        // 也检查 channel 中是否还有数据
        if shell_events_are_live && !has_more_data && !session.shell.events().is_empty() {
            has_more_data = true;
        }

        // 处理本帧的数据
        if !accumulated_data.is_empty() {
            if active_protocol_responses.has_pending() {
                // Preserve PTY byte order and stop accepting more protocol
                // requests until their older replies have entered the bounded
                // shell writer. The current chunk precedes the split-off tail.
                accumulated_data.append(&mut session.pending_output);
                session.pending_output = accumulated_data;
                has_more_data = true;
            } else {
                let mut terminal = session.terminal.lock();
                let active_parse_started = std::time::Instant::now();
                terminal.process_batch(&accumulated_data);
                terminal.check_sync_output_timeout();
                terminal_parse_time += active_parse_started.elapsed();
                active_processed_bytes = accumulated_data.len();
                let completed_outputs = terminal.take_completed_command_events();
                // 不再每帧清空 status_message:它由 set_status*/current_status_for_display
                // 按时长自动过期,否则任何快速输出都会把瞬时反馈瞬间吞掉。
                // 有输出时更新最后活动时间
                self.last_activity_time = std::time::Instant::now();
                drop(terminal);
                for completed in completed_outputs {
                    self.agent_panel
                        .handle_completed(&session.metadata.session_id, &completed);
                    self.command_correction.handle_completed(
                        &self.config,
                        self.agent_panel.session_active(),
                        &session.metadata.session_id,
                        &completed,
                    );
                    self.git_strip_cache
                        .mark_command_finished(&session.metadata.session_id);
                    // The active pane is on screen, so its completion was
                    // watched whenever the window itself had focus.
                    maybe_notify_long_command(
                        &self.config,
                        &mut self.notification_window_started,
                        &mut self.notifications_in_window,
                        &completed,
                        window_focused,
                    );
                    record_command_history(&self.config, &completed);
                    if let Err(error) = execution_journal::submit(completed) {
                        log::warn!(
                            "jsh execution output journal queue rejected an event: {error:?}"
                        );
                    }
                }
            }
        }

        let processed_bytes = background_processed_bytes.saturating_add(active_processed_bytes);
        let output_backlogged = background_has_more
            || has_more_data
            || processed_bytes >= self.adaptive_frame_budget.saturating_mul(3) / 4;
        self.adaptive_frame_budget = app::rendering::adapt_frame_budget(
            self.adaptive_frame_budget,
            processed_bytes,
            terminal_parse_time,
            output_backlogged,
        );

        // Step 7: 发送终端输出回 shell（DSR 响应等）
        {
            let mut terminal = session.terminal.lock();
            terminal.check_sync_output_timeout();
            let output = terminal.get_output();
            if let Err((error, mut output)) = active_protocol_responses.try_enqueue(output) {
                if error == ProtocolResponseQueueError::Full {
                    output.append(&mut terminal.output_buffer);
                    terminal.output_buffer = output;
                    has_more_data = true;
                } else {
                    log::warn!("terminal protocol response queue rejected output: {error}");
                }
            }
            let clipboard_requests = terminal.take_clipboard_read_requests();
            drop(terminal);
            if let Err(error) = active_protocol_responses.flush(&session.shell) {
                if !error.is_backpressure() {
                    log::warn!("terminal protocol response queue stopped: {error}");
                }
            }
            service_osc5522_clipboard_requests(
                self.clipboard.is_some(),
                &self.clipboard_request_in_flight,
                Arc::clone(&session.terminal),
                active_protocol_responses.clone(),
                clipboard_requests,
            );
        }

        // OSC 52 clipboard handling
        {
            let mut terminal = session.terminal.lock();
            if let Some(text) = terminal.take_osc52_clipboard_set() {
                if self.config.osc52_clipboard_write {
                    enqueue_osc52_clipboard_write(
                        self.osc52_clipboard_write_tx.as_ref(),
                        &mut self.osc52_write_window_started,
                        &mut self.osc52_writes_in_window,
                        text,
                    );
                }
            }
            let osc52_query = terminal.take_osc52_clipboard_query();
            drop(terminal);
            // Reading the clipboard exposes user data to a terminal program,
            // so it remains opt-in. Even when enabled, the external helper
            // and base64 encoding run only on the bounded background path.
            if osc52_query && self.config.osc52_clipboard_read {
                service_osc52_clipboard_query(
                    self.clipboard.is_some(),
                    &self.clipboard_request_in_flight,
                    Arc::clone(&session.terminal),
                    active_protocol_responses.clone(),
                    &mut self.osc52_read_window_started,
                    &mut self.osc52_reads_in_window,
                );
            }
        }

        // OSC 9/777 desktop notifications
        {
            let mut terminal = session.terminal.lock();
            let notifications: Vec<_> = terminal.pending_notifications.drain(..).collect();
            drop(terminal);
            for (title, body) in notifications {
                show_desktop_notification(
                    self.notification_tx.as_ref(),
                    &mut self.notification_window_started,
                    &mut self.notifications_in_window,
                    title,
                    body,
                );
            }
        }

        // Step 8: 光标闪烁
        // 优化逻辑：只有在完全空闲时才闪烁，有活动时保持常显
        let mut cursor_state_changed = false;
        {
            let terminal = session.terminal.lock();
            let app_wants_cursor_visible = terminal.is_cursor_visible();
            drop(terminal);

            if app_wants_cursor_visible {
                let now = std::time::Instant::now();
                let idle_duration = now.duration_since(self.last_activity_time);
                const IDLE_THRESHOLD: Duration = Duration::from_millis(1500); // 1.5秒空闲后才开始闪烁

                if idle_duration < IDLE_THRESHOLD {
                    // 有活动或刚有活动，光标保持常显
                    if !self.cursor_visible {
                        self.cursor_visible = true;
                        cursor_state_changed = true;
                    }
                    // 重置下次闪烁时间为空闲阈值后
                    self.next_cursor_blink_time = self.last_activity_time + IDLE_THRESHOLD;
                } else {
                    // 完全空闲，开始闪烁
                    if now >= self.next_cursor_blink_time {
                        self.cursor_visible = !self.cursor_visible;
                        cursor_state_changed = true;

                        debug_log!(
                            "[CURSOR] idle blink toggle: {}, next in 1000ms",
                            self.cursor_visible
                        );

                        // 计算下一次改变的时间
                        self.next_cursor_blink_time = now + Duration::from_millis(1000);
                    }
                }
            } else if self.cursor_visible {
                self.cursor_visible = false;
                cursor_state_changed = true;
            }
        }

        // Step 9: 滚动处理
        // 优化：批量处理键盘滚动，只获取一次锁
        let page_scroll_key = (!ui_input_blocked)
            .then(|| {
                ctx.input(|i| {
                    if i.key_pressed(egui::Key::PageUp) {
                        Some((egui::Key::PageUp, i.modifiers))
                    } else if i.key_pressed(egui::Key::PageDown) {
                        Some((egui::Key::PageDown, i.modifiers))
                    } else {
                        None
                    }
                })
            })
            .flatten();
        let scroll_amount = page_scroll_key.and_then(|(key, modifiers)| {
            let terminal = session.terminal.lock();
            let (_, rows) = terminal.get_dimensions();
            app::input::viewport_scroll_delta(key, modifiers, rows)
        });

        if let Some(amount) = scroll_amount {
            let renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            let viewport = projected_viewport_for_session(session, renderer);
            if viewport.is_transformed() {
                session.projection_view_state.scroll(amount, &viewport);
            } else {
                session.terminal.lock().scroll(amount);
            }
        }

        let scroll_delta = ctx.input(|i| i.smooth_scroll_delta.y);
        let ctrl_scroll_this_frame = self.frame_events.iter().any(
            |event| matches!(event, egui::Event::MouseWheel { modifiers, .. } if modifiers.ctrl),
        );

        // 检查是否启用鼠标报告
        let mouse_enabled = {
            let terminal = session.terminal.lock();
            session.purpose != crate::session::SessionPurpose::RetainedCommand
                && terminal.is_mouse_enabled()
        };
        let shift_mouse_bypass = ctx.input(|input| input.modifiers.shift);
        let pointer_pos =
            ctx.input(|input| input.pointer.interact_pos().or(input.pointer.hover_pos()));
        let mut mouse_projection_viewport = {
            let renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            projected_viewport_for_session(session, renderer)
        };

        // Apply velocity already admitted by an earlier frame before taking
        // ownership decisions for this frame. A wheel event admitted below is
        // intentionally applied on the next frame, so no viewport mutation can
        // split one pointer batch across two projection snapshots.
        let mut viewport_changed_by_smooth_scroll = false;
        if self.smooth_scroll_velocity.abs() > 0.1 {
            self.smooth_scroll_velocity *= 0.88;

            let line_h = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer)
                .line_height
                .max(1.0);

            // 抵达边界检测：在累积偏移前先看当前是否已到顶/到底(或处于备用屏幕)。
            // 若惯性继续往边界外推，会出现"跨行 → scroll 被钳制 → 偏移回弹"的逐帧抖动。
            let transformed_scroll = mouse_projection_viewport.is_transformed();
            let mut hit_boundary = if transformed_scroll {
                let offset = session.projection_view_state.offset_from_bottom();
                let at_top = offset >= mouse_projection_viewport.max_scroll_offset();
                let at_bottom = offset == 0;
                (self.smooth_scroll_velocity > 0.0 && at_top)
                    || (self.smooth_scroll_velocity < 0.0 && at_bottom)
            } else {
                let terminal = session.terminal.lock();
                let at_top = terminal.scroll_offset >= terminal.scrollback_len();
                let at_bottom = terminal.scroll_offset == 0;
                let alt = terminal.is_alt_buffer();
                alt || (self.smooth_scroll_velocity > 0.0 && at_top)
                    || (self.smooth_scroll_velocity < 0.0 && at_bottom)
            };

            if !hit_boundary {
                self.smooth_scroll_pixel_offset += self.smooth_scroll_velocity;
                let lines = (self.smooth_scroll_pixel_offset / line_h) as isize;
                if lines != 0 {
                    self.smooth_scroll_pixel_offset -= lines as f32 * line_h;
                    if transformed_scroll {
                        let before = session.projection_view_state.offset_from_bottom();
                        session
                            .projection_view_state
                            .scroll(lines, &mouse_projection_viewport);
                        let after = session.projection_view_state.offset_from_bottom();
                        viewport_changed_by_smooth_scroll = after != before;
                        let moved_fully = if lines > 0 {
                            after.saturating_sub(before) == lines as usize
                        } else {
                            before.saturating_sub(after) == lines.unsigned_abs()
                        };
                        if !moved_fully {
                            hit_boundary = true;
                        }
                    } else {
                        let mut terminal = session.terminal.lock();
                        let before = terminal.scroll_offset as isize;
                        terminal.scroll(lines);
                        viewport_changed_by_smooth_scroll =
                            terminal.scroll_offset as isize != before;
                        // 实际移动行数不等于请求行数 => 在本帧触及边界，立即停下惯性。
                        if terminal.scroll_offset as isize - before != lines {
                            hit_boundary = true;
                        }
                    }
                }
            }

            if hit_boundary {
                self.smooth_scroll_velocity = 0.0;
                self.smooth_scroll_pixel_offset = 0.0;
            }

            // 渲染偏移取负：shader 中 +offset 使内容上移，而 terminal.scroll(+lines)
            // 是向历史滚动(内容下移)。两者方向必须一致，否则每跨一行就会出现约
            // 2*行高的回跳——滚轮停下后的低速惯性阶段表现为上下抖动。
            if let Some(index) = active_pane_renderer_idx {
                if let Some(renderer) = self.pane_renderers.get_mut(index) {
                    renderer.scroll_pixel_offset = -self.smooth_scroll_pixel_offset;
                }
            } else {
                self.renderer.scroll_pixel_offset = -self.smooth_scroll_pixel_offset;
            }
            if !hit_boundary {
                ctx.request_repaint();
            }
        } else if self.smooth_scroll_velocity.abs() > 0.0 {
            self.smooth_scroll_velocity = 0.0;
            self.smooth_scroll_pixel_offset = 0.0;
            if let Some(index) = active_pane_renderer_idx {
                if let Some(renderer) = self.pane_renderers.get_mut(index) {
                    renderer.scroll_pixel_offset = 0.0;
                }
            } else {
                self.renderer.scroll_pixel_offset = 0.0;
            }
        }

        if viewport_changed_by_smooth_scroll {
            let renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            mouse_projection_viewport = projected_viewport_for_session(session, renderer);
        }

        // Mouse ownership, coordinate mapping and link hit-testing below all
        // consume this one post-scroll snapshot. Nothing below may mutate the
        // raw or projected viewport until the pointer batch has been routed.
        let pointer_app_mouse_eligible = pointer_over_active_terminal
            && pointer_pos.is_some_and(|pos| {
                let renderer = active_pane_renderer_idx
                    .and_then(|index| self.pane_renderers.get(index))
                    .unwrap_or(&self.renderer);
                let terminal = session.terminal.lock();
                renderer.pointer_app_mouse_eligible_projected(
                    &terminal,
                    &mouse_projection_viewport,
                    pos,
                )
            });

        // Resolve Ctrl-only link ownership before creating a PTY mouse
        // capture. Host link activation owns this press on any real grid cell
        // except a finished command header, so a mouse-reporting foreground
        // application must never receive the same press as the host opener.
        let link_press = ctx.input(|input| {
            input
                .pointer
                .button_pressed(egui::PointerButton::Primary)
                .then_some((
                    input.modifiers.ctrl
                        && !input.modifiers.shift
                        && !input.modifiers.alt
                        && !input.modifiers.command,
                    input
                        .pointer
                        .button_double_clicked(egui::PointerButton::Primary)
                        || input
                            .pointer
                            .button_triple_clicked(egui::PointerButton::Primary),
                    input.pointer.interact_pos().or(input.pointer.hover_pos()),
                ))
        });
        let mut link_press_override = false;
        if let Some((ctrl_only, multiple_click, press_pos)) = link_press {
            self.pending_link_activation = None;
            if !terminal_pointer_input_blocked && ctrl_only && !multiple_click {
                if let Some(origin) = press_pos {
                    let renderer = active_pane_renderer_idx
                        .and_then(|index| self.pane_renderers.get(index))
                        .unwrap_or(&self.renderer);
                    let terminal = session.terminal.lock();
                    if renderer.pointer_link_eligible_projected(
                        &terminal,
                        &mouse_projection_viewport,
                        origin,
                    ) {
                        let links = self
                            .link_detector
                            .detect_links_in_visible_cells_with_wrapping_and_hyperlinks(
                                mouse_projection_viewport.cells(),
                                mouse_projection_viewport.row_wrapped(),
                                |id| terminal.hyperlink_uri(id).map(str::to_owned),
                            );
                        let content_rect = renderer
                            .last_content_rect
                            .unwrap_or_else(|| ctx.viewport_rect());
                        if let Some(link) = link_at_pointer(
                            &links,
                            origin,
                            content_rect,
                            renderer.char_width,
                            renderer.line_height,
                            &mouse_projection_viewport,
                        ) {
                            self.pending_link_activation =
                                Some(crate::app::state::PendingLinkActivation {
                                    session_id: active_session_id.clone(),
                                    link,
                                    origin,
                                    cancelled: false,
                                    released_at: None,
                                });
                            link_press_override = true;
                        }
                    }
                }
            }
        }

        // Host-owned wheel input is admitted only after the post-scroll
        // snapshot has decided surface ownership. Its velocity starts next
        // frame; applying it now could invalidate this same pointer batch.
        if !terminal_pointer_input_blocked
            && pointer_over_active_terminal
            && scroll_delta != 0.0
            && !ctrl_scroll_this_frame
            && (!mouse_enabled || shift_mouse_bypass || !pointer_app_mouse_eligible)
        {
            const SCROLL_VELOCITY_DAMPING: f32 = 0.35;
            self.smooth_scroll_velocity +=
                scroll_delta * self.config.scroll_speed as f32 * SCROLL_VELOCITY_DAMPING;
            ctx.request_repaint();
        }

        let middle_paste_requested = session.purpose
            != crate::session::SessionPurpose::RetainedCommand
            && !terminal_pointer_input_blocked
            && (!mouse_enabled || shift_mouse_bypass || !pointer_app_mouse_eligible)
            && pointer_over_active_terminal
            && ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Middle));

        // Step 11: 鼠标处理（包括滚轮）
        let terminal_button_pressed = ctx.input(|input| {
            if input.pointer.button_pressed(egui::PointerButton::Primary) {
                Some(0)
            } else if input.pointer.button_pressed(egui::PointerButton::Secondary) {
                Some(2)
            } else if input.pointer.button_pressed(egui::PointerButton::Middle) {
                Some(1)
            } else {
                None
            }
        });
        let terminal_buttons_released = ctx.input(|input| {
            let mut buttons: SmallVec<[u8; 3]> = SmallVec::new();
            if input.pointer.button_released(egui::PointerButton::Primary) {
                buttons.push(0);
            }
            if input
                .pointer
                .button_released(egui::PointerButton::Secondary)
            {
                buttons.push(2);
            }
            if input.pointer.button_released(egui::PointerButton::Middle) {
                buttons.push(1);
            }
            buttons
        });
        if let Some(button) = terminal_button_pressed.filter(|button| {
            mouse_capture_accepts_new_press(self.terminal_mouse_capture.is_some())
                && (mouse_enabled || *button == 0)
                && !terminal_pointer_input_blocked
                && pointer_over_active_terminal
        }) {
            let pointer_renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            let content_rect = pointer_renderer
                .last_content_rect
                .unwrap_or_else(|| ctx.viewport_rect());
            let (mouse_cols, mouse_rows) = session.terminal.lock().get_dimensions();
            let display_cell = mouse_cell_for_current_dimensions(
                pointer_pos.or(Some(content_rect.center())),
                None,
                content_rect,
                pointer_renderer.char_width,
                pointer_renderer.line_height,
                mouse_cols,
                mouse_rows,
            );
            let application_cell = pointer_pos.and_then(|pointer| {
                application_cell_at_pointer(
                    pointer,
                    content_rect,
                    pointer_renderer.char_width,
                    pointer_renderer.line_height,
                    &mouse_projection_viewport,
                )
            });
            let reported_to_app = app_mouse_press_reports_from_snapshot(
                mouse_press_reports_to_app(
                    mouse_enabled,
                    shift_mouse_bypass,
                    pointer_app_mouse_eligible,
                    link_press_override,
                ),
                application_cell,
            );
            let (last_row, last_col) = if reported_to_app {
                application_cell.unwrap_or(display_cell)
            } else {
                display_cell
            };
            self.terminal_mouse_capture = Some(crate::app::state::TerminalMouseCapture {
                session_id: active_session_id.clone(),
                reported_to_app,
                button,
                terminal: Arc::clone(&session.terminal),
                write_tx: session.shell.write_sender(),
                protocol_responses: active_protocol_responses.clone(),
                content_rect,
                char_width: pointer_renderer.char_width,
                line_height: pointer_renderer.line_height,
                last_col,
                last_row,
                pending_controls: std::collections::VecDeque::new(),
                press_accepted: false,
                release_observed: false,
                // A host Ctrl-link owns the complete primary-button gesture.
                // Suppress the local-selection release path as well as PTY
                // reporting so an older terminal selection cannot overwrite
                // PRIMARY when the link opens.
                local_selection_cancelled: link_press_override,
            });
        }

        let pointer_any_down = ctx.input(|input| input.pointer.any_down());
        let capture_button_explicitly_released = self
            .terminal_mouse_capture
            .as_ref()
            .is_some_and(|capture| terminal_buttons_released.contains(&capture.button));
        let capture_finished = self.terminal_mouse_capture.is_some()
            && (capture_button_explicitly_released || !pointer_any_down);
        if capture_finished {
            if let Some(capture) = self.terminal_mouse_capture.as_mut() {
                capture.release_observed = true;
            }
        }
        let primary_copy_route = primary_copy_route(
            self.terminal_mouse_capture.as_ref().map(|capture| {
                (
                    capture.reported_to_app,
                    capture.local_selection_cancelled,
                    capture.button,
                )
            }),
            capture_finished,
            terminal_buttons_released.contains(&0),
        );
        let local_primary_selection_terminal =
            if primary_copy_route == PrimaryCopyRoute::CapturedLocal {
                self.terminal_mouse_capture
                    .as_ref()
                    .map(|capture| Arc::clone(&capture.terminal))
            } else {
                None
            };
        let capture_for_route = self.terminal_mouse_capture.as_ref();
        let capture_route_state =
            capture_for_route.map(|capture| (capture.reported_to_app, capture.button));
        let pointer_routes_to_terminal =
            pointer_over_active_terminal || capture_for_route.is_some();
        let sequence_reports_to_app = capture_route_state
            .map(|(reported_to_app, _)| reported_to_app)
            .unwrap_or(!shift_mouse_bypass && pointer_app_mouse_eligible);
        let reported_capture_release = captured_release_button(
            capture_route_state,
            &terminal_buttons_released,
            pointer_any_down,
        );
        let only_release = terminal_pointer_input_blocked;

        let (
            mouse_terminal,
            mouse_write_tx,
            mouse_protocol_responses,
            mouse_session_id,
            content_rect,
            char_width,
            line_height,
            fallback_cell,
        ) = if let Some(capture) = capture_for_route {
            (
                Arc::clone(&capture.terminal),
                capture.write_tx.clone(),
                capture.protocol_responses.clone(),
                capture.session_id.clone(),
                capture.content_rect,
                capture.char_width,
                capture.line_height,
                Some((capture.last_row, capture.last_col)),
            )
        } else {
            let pointer_renderer = active_pane_renderer_idx
                .and_then(|index| self.pane_renderers.get(index))
                .unwrap_or(&self.renderer);
            (
                Arc::clone(&session.terminal),
                session.shell.write_sender(),
                active_protocol_responses.clone(),
                active_session_id.clone(),
                pointer_renderer
                    .last_content_rect
                    .unwrap_or_else(|| ctx.viewport_rect()),
                pointer_renderer.char_width,
                pointer_renderer.line_height,
                None,
            )
        };
        let mouse_uses_active_projection = mouse_session_id == active_session_id
            && Arc::ptr_eq(&mouse_terminal, &session.terminal);
        let mut mouse_route_closed = false;
        let lossy_mouse_reports: Vec<Vec<u8>> = if (!sequence_reports_to_app
            || !pointer_routes_to_terminal
            || (terminal_pointer_input_blocked && reported_capture_release.is_none()))
            || (pointer_pos.is_none() && fallback_cell.is_none())
        {
            self.mouse_scroll_accumulator = 0.0;
            Vec::new()
        } else {
            let terminal = mouse_terminal.lock();
            if !terminal.is_mouse_enabled() {
                self.mouse_scroll_accumulator = 0.0;
                // The application disabled mouse reporting while a sequence was
                // active. An unaccepted press can be retired, but a release
                // already encoded behind backpressure must still follow its
                // accepted press on the original writer route.
                let queued_release = self.terminal_mouse_capture.as_ref().is_some_and(|capture| {
                    capture.pending_controls.iter().any(|pending| {
                        pending.kind == crate::app::state::PendingMouseControlKind::Release
                    })
                });
                mouse_route_closed = self.terminal_mouse_capture.is_some() && !queued_release;
                drop(terminal);
                Vec::new()
            } else {
                let mut reports = Vec::new();
                let (_, mouse_rows) = terminal.get_dimensions();
                let projected_pointer_cell = mouse_uses_active_projection.then(|| {
                    pointer_pos.and_then(|pointer| {
                        application_cell_at_pointer(
                            pointer,
                            content_rect,
                            char_width,
                            line_height,
                            &mouse_projection_viewport,
                        )
                    })
                });
                let frame_route = app_mouse_frame_route(
                    mouse_uses_active_projection,
                    projected_pointer_cell,
                    fallback_cell,
                );
                if let Some((row, col)) = frame_route.lossy_cell {
                    if let Some(capture) = self.terminal_mouse_capture.as_mut() {
                        if capture.session_id == mouse_session_id {
                            capture.last_col = col;
                            capture.last_row = row;
                        }
                    }
                }

                if let Some((row, col)) =
                    (!only_release).then_some(frame_route.lossy_cell).flatten()
                {
                    // 处理鼠标滚轮（当启用鼠标报告时）
                    let line_h = line_height.max(1.0);
                    let mut discrete_scroll_steps: isize = 0;
                    let mut point_scroll_delta: f32 = 0.0;

                    ctx.input(|i| {
                        for event in &i.events {
                            if let egui::Event::MouseWheel {
                                unit,
                                delta,
                                modifiers,
                                ..
                            } = event
                            {
                                if modifiers.ctrl {
                                    continue;
                                }
                                match unit {
                                    egui::MouseWheelUnit::Line => {
                                        discrete_scroll_steps = bounded_wheel_step_accumulate(
                                            discrete_scroll_steps,
                                            delta.y,
                                            1,
                                        );
                                    }
                                    egui::MouseWheelUnit::Page => {
                                        discrete_scroll_steps = bounded_wheel_step_accumulate(
                                            discrete_scroll_steps,
                                            delta.y,
                                            mouse_rows.max(1),
                                        );
                                    }
                                    egui::MouseWheelUnit::Point => {
                                        if delta.y.is_finite() {
                                            let limit =
                                                line_h * MAX_MOUSE_WHEEL_REPORTS_PER_FRAME as f32;
                                            point_scroll_delta =
                                                (point_scroll_delta + delta.y).clamp(-limit, limit);
                                        }
                                    }
                                }
                            }
                        }
                    });

                    if point_scroll_delta != 0.0 {
                        let limit = line_h * MAX_MOUSE_WHEEL_REPORTS_PER_FRAME as f32;
                        self.mouse_scroll_accumulator = (self.mouse_scroll_accumulator
                            + point_scroll_delta)
                            .clamp(-limit, limit);
                    }

                    let point_scroll_steps = ((self.mouse_scroll_accumulator / line_h) as isize)
                        .clamp(
                            -MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
                            MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
                        );
                    if point_scroll_steps != 0 {
                        self.mouse_scroll_accumulator -= point_scroll_steps as f32 * line_h;
                    }

                    let total_scroll_steps = discrete_scroll_steps
                        .saturating_add(point_scroll_steps)
                        .clamp(
                            -MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
                            MAX_MOUSE_WHEEL_REPORTS_PER_FRAME,
                        );
                    if total_scroll_steps != 0 {
                        let button = if total_scroll_steps > 0 { 64 } else { 65 };

                        for _ in 0..total_scroll_steps.unsigned_abs() {
                            if let Some(report) = terminal.get_mouse_report(button, col, row) {
                                reports.push(report);
                            }
                        }
                    }

                    if let Some(capture) = self.terminal_mouse_capture.as_ref() {
                        if capture.reported_to_app
                            && terminal_button_pressed == Some(capture.button)
                        {
                            if let Some(report) =
                                terminal.get_mouse_report(capture.button, col, row)
                            {
                                if let Some(capture) = self.terminal_mouse_capture.as_mut() {
                                    queue_mouse_control(
                                        &mut capture.pending_controls,
                                        crate::app::state::PendingMouseControlKind::Press,
                                        report,
                                    );
                                }
                            }
                        }
                    }

                    let pointer_moved =
                        ctx.input(|input| input.pointer.delta() != egui::Vec2::ZERO);
                    let motion_button = reported_capture_button(capture_route_state);
                    if pointer_moved
                        && terminal.should_report_mouse_motion(motion_button.is_some())
                        && self.last_terminal_mouse_motion.as_ref().is_none_or(
                            |(session_id, last_col, last_row)| {
                                session_id != &mouse_session_id
                                    || *last_col != col
                                    || *last_row != row
                            },
                        )
                    {
                        let base_button = motion_button.unwrap_or(3);
                        if let Some(report) =
                            terminal.get_mouse_report(base_button.saturating_add(32), col, row)
                        {
                            reports.push(report);
                            self.last_terminal_mouse_motion =
                                Some((mouse_session_id.clone(), col, row));
                        }
                    }
                } else {
                    // Synthetic summary/padding rows never become an app
                    // surface. A capture whose terminal is no longer active is
                    // release-only as well: never reinterpret the new pane's
                    // display coordinates as the old PTY's raw grid. Do not
                    // retain wheel fractions that could fire on re-entry.
                    self.mouse_scroll_accumulator = 0.0;
                }

                // A release is emitted exactly once and only for a press
                // captured by this terminal. Mode 1002 therefore cannot
                // see an orphan release after a drag began elsewhere.
                if let (Some(button), Some((row, col))) =
                    (reported_capture_release, frame_route.release_cell)
                {
                    if let Some(report) = terminal.get_mouse_release_report(button, col, row) {
                        if let Some(capture) = self.terminal_mouse_capture.as_mut() {
                            queue_mouse_control(
                                &mut capture.pending_controls,
                                crate::app::state::PendingMouseControlKind::Release,
                                report,
                            );
                        }
                    }
                }

                drop(terminal);
                reports
            }
        };

        let has_mouse_input = !lossy_mouse_reports.is_empty()
            || self
                .terminal_mouse_capture
                .as_ref()
                .is_some_and(|capture| !capture.pending_controls.is_empty());
        let mouse_protocol_input_blocked = mouse_protocol_input_is_blocked(
            &mouse_session_id,
            &self.osc_paste_input_barriers,
            &mouse_protocol_responses,
        );
        let mut mouse_write_error = None;
        if !mouse_route_closed {
            if let Some(capture) = self.terminal_mouse_capture.as_mut() {
                if capture.reported_to_app {
                    if let Err(error) =
                        flush_mouse_controls(capture, &self.osc_paste_input_barriers)
                    {
                        if !error.is_backpressure() {
                            mouse_route_closed = true;
                        }
                        mouse_write_error = Some(error);
                    }
                }
            }
        }

        if !mouse_route_closed
            && mouse_write_error.is_none()
            && mouse_lossy_reports_allowed(
                mouse_protocol_input_blocked,
                mouse_capture_allows_lossy(self.terminal_mouse_capture.as_ref()),
            )
        {
            for report in lossy_mouse_reports {
                if let Err(error) = mouse_write_tx.try_send(report) {
                    if !error.is_backpressure() && self.terminal_mouse_capture.is_some() {
                        mouse_route_closed = true;
                    }
                    mouse_write_error = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = mouse_write_error {
            // Motion/wheel are deliberately lossy. Press/release remain queued
            // on the capture and are retried in order on the next frame.
            self.status_message = format!("鼠标报告发送失败：{error}");
            self.status_expires_at = Some(std::time::Instant::now() + Duration::from_secs(3));
            if error.is_backpressure() && self.terminal_mouse_capture.is_some() {
                ctx.request_repaint_after(Duration::from_millis(10));
            }
        }
        let capture_complete = self
            .terminal_mouse_capture
            .as_ref()
            .is_some_and(mouse_capture_is_complete);
        if mouse_route_closed || capture_complete {
            self.terminal_mouse_capture = None;
            self.last_terminal_mouse_motion = None;
        }

        // Step 12: 链接检测和交互
        if terminal_pointer_input_blocked {
            self.hovered_link = None;
            self.pending_link_activation = None;
        } else {
            let terminal_ptr = Arc::as_ptr(&session.terminal) as usize;
            let terminal = session.terminal.lock();
            let pointer = ctx.input(|input| input.pointer.hover_pos());

            self.hovered_link = if let Some(renderer_idx) = active_pane_renderer_idx {
                let projection_key = mouse_projection_viewport.key();
                let needs_refresh = self
                    .pane_renderers
                    .get(renderer_idx)
                    .is_some_and(|renderer| {
                        renderer.cached_links_projection_key != Some(projection_key)
                            || terminal_ptr != renderer.cached_links_terminal_ptr
                    });
                if needs_refresh {
                    let links = self
                        .link_detector
                        .detect_links_in_visible_cells_with_wrapping_and_hyperlinks(
                            mouse_projection_viewport.cells(),
                            mouse_projection_viewport.row_wrapped(),
                            |id| terminal.hyperlink_uri(id).map(str::to_owned),
                        );
                    if let Some(renderer) = self.pane_renderers.get_mut(renderer_idx) {
                        renderer.cached_links = Arc::new(links);
                        renderer.cached_links_projection_key = Some(projection_key);
                        renderer.cached_links_terminal_ptr = terminal_ptr;
                    }
                }
                self.pane_renderers
                    .get(renderer_idx)
                    .and_then(|renderer| Some((renderer, pointer?, renderer.last_content_rect?)))
                    .and_then(|(renderer, pointer, rect)| {
                        link_at_pointer(
                            &renderer.cached_links,
                            pointer,
                            rect,
                            renderer.char_width,
                            renderer.line_height,
                            &mouse_projection_viewport,
                        )
                    })
            } else {
                let projection_key = mouse_projection_viewport.key();
                if self.cached_links_projection_key != Some(projection_key)
                    || terminal_ptr != self.cached_links_terminal_ptr
                {
                    self.cached_links = self
                        .link_detector
                        .detect_links_in_visible_cells_with_wrapping_and_hyperlinks(
                            mouse_projection_viewport.cells(),
                            mouse_projection_viewport.row_wrapped(),
                            |id| terminal.hyperlink_uri(id).map(str::to_owned),
                        );
                    self.cached_links_projection_key = Some(projection_key);
                    self.cached_links_terminal_ptr = terminal_ptr;
                }
                pointer
                    .zip(self.renderer.last_content_rect)
                    .and_then(|(pointer, rect)| {
                        link_at_pointer(
                            &self.cached_links,
                            pointer,
                            rect,
                            self.renderer.char_width,
                            self.renderer.line_height,
                            &mouse_projection_viewport,
                        )
                    })
            };
            let hovered_link_eligible = pointer.is_some_and(|pos| {
                let renderer = active_pane_renderer_idx
                    .and_then(|index| self.pane_renderers.get(index))
                    .unwrap_or(&self.renderer);
                renderer.pointer_link_eligible_projected(&terminal, &mouse_projection_viewport, pos)
            });
            drop(terminal);
            if self.hovered_link.is_some() && hovered_link_eligible {
                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
            }

            // 链接悬停提示:在指针右下方画一个浮层显示完整 URL 和 Ctrl+Click 操作提示。
            // OSC8 等链接显示的"文本"可能与真实目标不同(例如 "click here"),
            // 鼠标悬停透出真实跳转目标,避免用户被诱导点击未知链接。
            if let Some(link) = self.hovered_link.as_ref().filter(|_| hovered_link_eligible) {
                if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                    egui::Area::new(egui::Id::new("link_hover_tooltip"))
                        .order(egui::Order::Tooltip)
                        .fixed_pos(pos + egui::vec2(14.0, 18.0))
                        .interactable(false)
                        .show(ctx, |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                ui.set_max_width(520.0);
                                ui.label(egui::RichText::new(&link.text).monospace());
                                ui.label(egui::RichText::new("Ctrl+Click 打开").small().weak());
                            });
                        });
                }
            }

            // The stable Ctrl-link target was captured before PTY mouse-route
            // ownership above. Modifier release does not change that target;
            // only a drag/session change/multiclick cancels it here.
            if let Some(pending) = self.pending_link_activation.as_mut() {
                if pending.session_id != active_session_id
                    || ctx.input(|input| {
                        input.pointer.any_down()
                            && input
                                .pointer
                                .hover_pos()
                                .is_some_and(|pos| link_activation_dragged(pending.origin, pos))
                    })
                {
                    pending.cancelled = true;
                }
            }
            let link_release = ctx.input(|input| {
                let awaiting_release = self
                    .pending_link_activation
                    .as_ref()
                    .is_some_and(|pending| pending.released_at.is_none());
                let released = input.pointer.button_released(egui::PointerButton::Primary)
                    || (!input.pointer.any_down() && awaiting_release);
                (
                    released,
                    input
                        .pointer
                        .button_double_clicked(egui::PointerButton::Primary)
                        || input
                            .pointer
                            .button_triple_clicked(egui::PointerButton::Primary),
                )
            });
            if link_release.0 {
                let releasable = self
                    .pending_link_activation
                    .as_ref()
                    .is_some_and(|pending| {
                        link_activation_release_allowed(
                            &pending.session_id,
                            &active_session_id,
                            pending.cancelled,
                            link_release.1,
                        )
                    });
                if releasable {
                    let released_at = ctx.input(|input| input.time);
                    if let Some(pending) = self.pending_link_activation.as_mut() {
                        pending.released_at = Some(released_at);
                    }
                    let raw_delay =
                        ctx.options(|options| options.input_options.max_double_click_delay);
                    let delay = if raw_delay.is_finite() {
                        raw_delay.max(0.0)
                    } else {
                        0.3
                    };
                    ctx.request_repaint_after(Duration::from_secs_f64(delay));
                } else {
                    self.pending_link_activation = None;
                }
            }
            let raw_delay = ctx.options(|options| options.input_options.max_double_click_delay);
            let double_click_delay = if raw_delay.is_finite() {
                raw_delay.max(0.0)
            } else {
                0.3
            };
            let now = ctx.input(|input| input.time);
            let link = self
                .pending_link_activation
                .as_ref()
                .filter(|pending| {
                    link_activation_ready(pending.released_at, now, double_click_delay)
                })
                .is_some()
                .then(|| self.pending_link_activation.take())
                .flatten()
                .map(|pending| pending.link);
            if let Some(link) = link {
                let (msg, dur) = match link::open_link(&link) {
                    Ok(_) => (
                        format!("Opened: {}", link.text),
                        Duration::from_millis(2500),
                    ),
                    Err(e) => (
                        format!("Failed to open link: {}", e),
                        Duration::from_secs(4),
                    ),
                };
                self.status_message = msg;
                self.status_expires_at = Some(std::time::Instant::now() + dur);
            }
        }

        // A host-owned middle click may scroll both raw and projected views to
        // the bottom. Run it only after every consumer of the immutable mouse
        // snapshot above has finished, so it cannot retarget this pointer batch.
        if middle_paste_requested {
            if let Some(clipboard) = &self.clipboard {
                let primary_text = clipboard.paste_primary().unwrap_or_default();
                let text = if primary_text.is_empty() {
                    clipboard.paste().unwrap_or_default()
                } else {
                    primary_text
                };
                let paste_result = paste_text_into_session(
                    session,
                    text,
                    self.config.paste_confirm,
                    PasteOrigin::Clipboard,
                    false,
                    crate::session_manager::user_input_flush_block(
                        &session.metadata.session_id,
                        mouse_input_barrier_session_id.as_deref(),
                        &self.osc_paste_input_barriers,
                        &active_protocol_responses,
                    )
                    .is_some(),
                    &mut self.pending_paste_confirm,
                );
                match paste_result {
                    Ok(true) if self.pending_paste_confirm.is_none() => {
                        app::commands::clear_block_selection_state_for_session(
                            &mut self.block_selection,
                            &mut self.command_sidebar.selected,
                            &active_session_id,
                        );
                    }
                    Ok(_) => {}
                    Err(error) => {
                        self.status_message = format!("粘贴失败：{error}");
                        self.status_expires_at =
                            Some(std::time::Instant::now() + Duration::from_secs(4));
                    }
                }
            }
        }

        // 检查光标闪烁是否需要进行（在render_ui之前完成）
        let app_wants_cursor_visible = {
            let terminal = session.terminal.lock();
            terminal.is_cursor_visible()
        };

        // 结束 session 的可变借用，render_ui 需要 &mut self
        #[allow(dropping_references)]
        drop(session);

        if shell_exit_observed {
            self.task_manager
                .handle_terminal_session_exit(&active_session_id, shell_exit_code);
            if retain_exited_task_terminal {
                self.session_manager
                    .retain_exited_command(&active_session_id);
            }
        }

        // Bookmark truth follows retained semantic records, not scrollback or
        // captured-output availability. The version-gated pass is O(1) on
        // static frames and scans only bookmarked sessions after a real deque
        // insertion/rotation.
        self.prune_block_bookmarks_to_retained_records();

        // 渲染 UI
        // A host-owned Ctrl-link press must not start renderer-local text
        // selection in the same frame. The mouse capture already suppresses
        // PTY reporting and PRIMARY-copy release for this gesture.
        self.render_ui(
            root_ui,
            terminal_pointer_input_blocked || link_press_override,
        );

        if !terminal_pointer_input_blocked && !self.terminal_input_blocked(ctx) {
            let selection_for_primary = match primary_copy_route {
                PrimaryCopyRoute::CapturedLocal => local_primary_selection_terminal
                    .and_then(|terminal| terminal.lock().copy_selection()),
                PrimaryCopyRoute::Generic => {
                    let session = self.session_manager.get_active_session_mut();
                    let terminal = session.terminal.lock();
                    if terminal.is_mouse_enabled() {
                        None
                    } else {
                        terminal.copy_selection()
                    }
                }
                PrimaryCopyRoute::None | PrimaryCopyRoute::SuppressCaptured => None,
            };
            if let (Some(clipboard), Some(text)) = (&self.clipboard, selection_for_primary) {
                if !text.is_empty() {
                    let _ = clipboard.copy_primary(&text);
                }
            }
        }

        // channel 中还有未处理的数据时，立即请求下一帧继续处理
        if has_more_data || background_has_more {
            ctx.request_repaint();
        } else {
            // 二次检查：render_ui 期间 PTY 线程可能又发送了新数据
            let has_pending_data = if !has_new_output {
                let session = self.session_manager.get_active_session_mut();
                session.purpose != crate::session::SessionPurpose::RetainedCommand
                    && !session.shell.events().is_empty()
            } else {
                false
            };
            let has_new_output = has_new_output || background_had_output || has_pending_data;

            let should_repaint = has_new_output
                || cursor_state_changed
                || has_keyboard_input
                || has_cursor_move_input
                || has_mouse_input
                || self.debug_panel.is_open;

            if should_repaint {
                ctx.request_repaint();
            } else if app_wants_cursor_visible {
                let now = std::time::Instant::now();
                let time_until_next = self.next_cursor_blink_time.saturating_duration_since(now);
                if time_until_next.as_millis() == 0 {
                    ctx.request_repaint();
                } else {
                    ctx.request_repaint_after(time_until_next);
                }
            } else {
                // 安全网：1000ms 超时防止极端竞态
                ctx.request_repaint_after(std::time::Duration::from_millis(1000));
            }
        }

        // Debounce 保存配置和会话
        self.flush_config_save();
        self.flush_session_save();
        self.check_config_hot_reload(ctx);

        // Handle shell exit: close current session
        if shell_exited {
            crate::debug_log!(
                "[SHELL EXIT] handling shell exit, session_count: {}",
                session_count_before
            );
            if retain_exited_task_terminal {
                crate::debug_log!("[SHELL EXIT] retaining task terminal for review");
            } else if session_count_before > 1 {
                // Close the current session if there are multiple sessions.
                // If it was its tab's only pane, the tab goes with it.
                self.close_session_or_owning_tab(active_session_idx);
                self.schedule_session_save();
                crate::debug_log!(
                    "[SHELL EXIT] closed session, remaining: {}",
                    self.session_manager.len()
                );
            } else if shell_startup_failed {
                // The last shell never reached a prompt. Closing here would
                // make the window vanish before the user can read why, so keep
                // it open with the failure message on screen instead.
                crate::debug_log!("[SHELL EXIT] startup failure, keeping window open");
            } else {
                // Close the window if this is the only session
                crate::debug_log!("[SHELL EXIT] closing window");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        // Drain/stage only after every frame interaction *and* scheduled shell
        // exit/active-session transition. Publishing next frame is harmless;
        // observing the closing session before close_session_synced bumps its
        // focus epoch would let a dead A commit immediately before B becomes
        // active. A retained/only exited session is never a follow authority.
        let active_session_id_after_close = self
            .session_manager
            .sessions()
            .get(self.session_manager.active_index())
            .map(|session| session.metadata.session_id.as_str());
        let active_session_is_live_for_follow = ssh_files_follow::poll_allowed_after_shell_exit(
            shell_exited,
            &active_session_id,
            active_session_id_after_close,
        );
        if active_session_is_live_for_follow {
            self.update_ssh_files_follow(ctx, frame_start_files_user_intent);
        }
    }
}

impl Drop for TerminalApp {
    fn drop(&mut self) {
        // 保存 agent 会话（取消/清空会话时会删除快照文件）
        self.agent_panel.persist();

        // 取消聊天面板在途请求并落盘聊天库（仅在面板曾成功加载、且本窗口是
        // 持锁主实例时写入；加载失败的旧文件绝不覆盖）。
        self.ai_chat_panel.cancel_all();
        self.ai_chat_panel.persist();

        // 保存配置
        if self.config_save_pending {
            if let Err(e) = self.config.save() {
                eprintln!("[Config] Failed to save on exit: {}", e);
            }
        }

        // 保存当前会话到持久化存储（包含每个 session 的 cwd）。只有持有实例锁
        // 的主实例能更新共享快照，避免后开的临时窗口在退出时覆盖完整状态。
        if self._lock_file.is_some() && !self.session_persistence_blocked {
            if let Ok(session_history_path) = self.config.resolved_session_history_path() {
                let _ = session_persistence::ensure_session_history_dir(&session_history_path);

                let snapshot = self.current_sessions_snapshot();
                if let Err(e) = snapshot.save(&session_history_path) {
                    eprintln!("[SessionPersistence] Failed to save sessions: {}", e);
                }
            }
        }

        // A completed command's rendered output is written off the UI thread.
        // Give already accepted snapshots a bounded chance to reach the shared
        // jsh journal before process teardown terminates that worker.
        if !execution_journal::flush(Duration::from_secs(2)) {
            log::warn!("timed out flushing jsh execution output journal");
        }
        // 命令历史同样由后台写入器落盘；退出前给已接受的记录一个有界冲刷
        // 窗口（与 frost 相同的 2 秒）。
        if let Err(error) =
            jterm_core::command_history::flush_pending(std::time::Duration::from_secs(2))
        {
            log::warn!("command history did not flush before exit: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        app_mouse_frame_route, app_mouse_press_reports_from_snapshot, application_cell_at_pointer,
        bounded_wheel_step_accumulate, captured_release_button, clipboard_5522_response_for_mime,
        clipboard_5522_response_for_mime_with_limit, desktop_notification_channel,
        encode_submitted_command, ensure_direct_paste_route_available,
        flush_pending_mouse_controls, fontconfig_match_family_file, kitty_graphics_payload,
        link_activation_dragged, link_activation_ready, link_activation_release_allowed,
        link_at_pointer, maybe_notify_long_command, mouse_capture_accepts_new_press,
        mouse_cell_for_current_dimensions, mouse_lossy_reports_allowed, mouse_press_reports_to_app,
        mouse_protocol_input_is_blocked, mouse_sequence_allows_lossy, mouse_sequence_is_complete,
        normalized_paste_body, osc52_clipboard_response_with_limit, osc52_read_rate_limit_allows,
        paste_policy, paste_requires_confirmation, primary_copy_route, queue_mouse_control,
        reported_capture_button, roll_notification_rate_window, should_notify_long_command,
        show_desktop_notification, snapshot_age_label, take_tagged_cursor_move,
        workspace_drag_pointer_cancelled, ClipboardRequestGuard, DesktopNotification, PasteOrigin,
        PasteWriteError, PrimaryCopyRoute, DESKTOP_NOTIFICATION_QUEUE_CAPACITY,
        KITTY_BASE64_CHUNK_BYTES, MAX_OSC52_READS_PER_WINDOW, OSC52_READ_RATE_WINDOW,
        OSC_5522_DATA_CHUNK_BYTES,
    };
    use crate::app::events::{
        normalize_terminal_shortcut_events, restore_missing_image_paste_key_event,
        semantic_paste_modifiers, shortcut_event_to_key_event, PasteKeyState,
    };
    use base64::Engine as _;
    use eframe::egui;
    use image::ImageEncoder as _;
    use std::time::Duration;

    #[test]
    fn files_panel_f5_refresh_requires_tree_focus_and_rejects_text_or_popup_focus() {
        assert!(crate::TerminalApp::files_panel_refresh_shortcut(
            crate::sidebar::SidebarView::Files,
            true,
            true,
            false,
            false,
            true,
        ));
        for (view, hovered, tree_focused, text_focused, popup_open) in [
            (
                crate::sidebar::SidebarView::Files,
                true,
                false,
                false,
                false,
            ),
            (
                crate::sidebar::SidebarView::Files,
                false,
                true,
                false,
                false,
            ),
            (crate::sidebar::SidebarView::Files, true, true, true, false),
            (crate::sidebar::SidebarView::Files, true, true, false, true),
            (
                crate::sidebar::SidebarView::Sessions,
                true,
                true,
                false,
                false,
            ),
        ] {
            assert!(!crate::TerminalApp::files_panel_refresh_shortcut(
                view,
                hovered,
                tree_focused,
                text_focused,
                popup_open,
                true,
            ));
        }
    }

    #[test]
    fn files_tree_shortcuts_never_capture_terminal_text_or_popup_input() {
        assert!(crate::TerminalApp::files_tree_keyboard_enabled(
            crate::sidebar::SidebarView::Files,
            true,
            true,
            false,
            false,
        ));
        for (view, hovered, tree_focused, text_focused, popup_open) in [
            (
                crate::sidebar::SidebarView::Files,
                false,
                true,
                false,
                false,
            ),
            (
                crate::sidebar::SidebarView::Files,
                true,
                false,
                false,
                false,
            ),
            (crate::sidebar::SidebarView::Files, true, true, true, false),
            (crate::sidebar::SidebarView::Files, true, true, false, true),
            (
                crate::sidebar::SidebarView::Sessions,
                true,
                true,
                false,
                false,
            ),
        ] {
            assert!(!crate::TerminalApp::files_tree_keyboard_enabled(
                view,
                hovered,
                tree_focused,
                text_focused,
                popup_open,
            ));
        }
    }

    #[test]
    fn directory_snapshot_age_uses_stable_human_scale_buckets() {
        assert_eq!(snapshot_age_label(Duration::ZERO), "刚刚");
        assert_eq!(snapshot_age_label(Duration::from_secs(59)), "59 秒前");
        assert_eq!(snapshot_age_label(Duration::from_secs(60)), "1 分钟前");
        assert_eq!(snapshot_age_label(Duration::from_secs(3_600)), "1 小时前");
        assert_eq!(snapshot_age_label(Duration::from_secs(86_400)), "1 天前");
    }

    #[test]
    fn drop_target_resolves_rows_blank_space_and_outside() {
        use egui::{pos2, vec2, Rect};
        use std::path::{Path, PathBuf};
        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 400.0));
        let dir_rect = Rect::from_min_size(pos2(0.0, 20.0), vec2(200.0, 18.0));
        let file_rect = Rect::from_min_size(pos2(0.0, 38.0), vec2(200.0, 18.0));
        let rows = vec![
            (dir_rect, PathBuf::from("/base/docs"), true),
            (file_rect, PathBuf::from("/base/notes.md"), false),
        ];
        let root = Path::new("/base");

        // 目录行 → 目录本身；文件行 → 父目录；空白处 → 根；面板外 → None。
        assert_eq!(
            crate::TerminalApp::resolve_drop_target(pos2(50.0, 25.0), panel, &rows, Some(root)),
            Some(PathBuf::from("/base/docs"))
        );
        assert_eq!(
            crate::TerminalApp::resolve_drop_target(pos2(50.0, 45.0), panel, &rows, Some(root)),
            Some(PathBuf::from("/base"))
        );
        assert_eq!(
            crate::TerminalApp::resolve_drop_target(pos2(50.0, 300.0), panel, &rows, Some(root)),
            Some(PathBuf::from("/base"))
        );
        assert_eq!(
            crate::TerminalApp::resolve_drop_target(pos2(500.0, 25.0), panel, &rows, Some(root)),
            None
        );
        // 没有根（远程起始目录未就绪）时，空白处不落点。
        assert_eq!(
            crate::TerminalApp::resolve_drop_target(pos2(50.0, 300.0), panel, &rows, None),
            None
        );
    }

    #[test]
    fn copy_path_payload_is_the_plain_full_path() {
        // 本地与远程行都是完整路径文本；远程行没有 ssh:/docker: 前缀。
        assert_eq!(
            crate::TerminalApp::fs_copy_path_payload(std::path::Path::new("/home/yj/notes.md")),
            "/home/yj/notes.md"
        );
        assert_eq!(
            crate::TerminalApp::fs_copy_path_payload(std::path::Path::new("/var/log/syslog")),
            "/var/log/syslog"
        );
        assert_eq!(
            crate::TerminalApp::fs_copy_path_payload(std::path::Path::new("/")),
            "/"
        );
    }

    #[test]
    fn font_fallback_loop_skips_a_non_regular_candidate() {
        let root = std::env::temp_dir().join(format!("ember-font-fallback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        // A directory behind a font-looking name fails the descriptor's
        // regular-file check and must fall through to the next candidate.
        let bad = root.join("bad.ttf");
        std::fs::create_dir(&bad).unwrap();
        let good = root.join("good.ttf");
        std::fs::write(&good, b"fallback-font-bytes").unwrap();

        let mut fonts = egui::FontDefinitions::default();
        let mut loaded_paths = std::collections::HashMap::new();
        assert!(super::load_first_matching_font(
            &mut fonts,
            &mut loaded_paths,
            &[],
            &[bad.to_str().unwrap(), good.to_str().unwrap()],
            "fallback_test",
            &[egui::FontFamily::Monospace],
            false,
        ));
        assert_eq!(
            fonts.font_data["fallback_test"].font.as_ref(),
            b"fallback-font-bytes".as_slice()
        );
        assert!(fonts.families[&egui::FontFamily::Monospace]
            .iter()
            .any(|name| name == "fallback_test"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn transformed_pointer_mapping_rejects_summary_and_reports_raw_grid_coordinates() {
        let mut terminal = crate::terminal::TerminalState::new(12, 6);
        terminal.process_input(
            b"\x1b]133;A\x07$ \x1b]133;C;id=fold\x07OUT\r\nMORE\x1b]133;D;0;id=fold\x07",
        );
        let zone_id = terminal.command_records().back().unwrap().sequence;
        let mut policy = crate::terminal::ProjectionPolicy::new();
        assert!(policy.collapse(zone_id));
        let mut view_state = crate::terminal::ProjectionViewState::new();
        let bottom = terminal.projected_viewport_with_state(
            crate::terminal::HistoryProjection::identity(),
            true,
            &policy,
            &mut view_state,
        );
        view_state.set_offset(bottom.max_scroll_offset(), &bottom);
        let viewport = terminal.projected_viewport_with_state(
            crate::terminal::HistoryProjection::identity(),
            true,
            &policy,
            &mut view_state,
        );
        let summary_row = viewport
            .row_kinds()
            .iter()
            .position(|kind| {
                matches!(
                    kind,
                    crate::terminal::ProjectedRowKind::CollapsedSummary { .. }
                )
            })
            .expect("collapsed output should have a visible summary");
        let (live_row, raw_cell) = (0..viewport.rows())
            .find_map(|row| {
                viewport
                    .application_cell(crate::terminal::DisplayPoint::new(row, 0))
                    .map(|raw| (row, raw))
            })
            .expect("the transformed viewport should retain a live grid row");
        let char_width = 10.0;
        let line_height = 10.0;
        let rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                viewport.columns() as f32 * char_width,
                viewport.rows() as f32 * line_height,
            ),
        );
        let pointer = |row| egui::pos2(5.0, row as f32 * line_height + 5.0);

        assert_eq!(
            application_cell_at_pointer(
                pointer(summary_row),
                rect,
                char_width,
                line_height,
                &viewport,
            ),
            None
        );
        assert_eq!(
            application_cell_at_pointer(
                pointer(live_row),
                rect,
                char_width,
                line_height,
                &viewport,
            ),
            Some(raw_cell)
        );

        let link = crate::link::Link {
            line: summary_row,
            col_start: 0,
            col_end: 1,
            link_type: crate::link::LinkType::Url,
            text: "https://example.test".to_owned(),
        };
        assert_eq!(
            link_at_pointer(
                &[link],
                pointer(summary_row),
                rect,
                char_width,
                line_height,
                &viewport,
            ),
            None
        );

        let live_link = crate::link::Link {
            line: live_row,
            col_start: 0,
            col_end: 1,
            link_type: crate::link::LinkType::Url,
            text: "https://visible.example.test".to_owned(),
        };
        assert_eq!(
            link_at_pointer(
                std::slice::from_ref(&live_link),
                pointer(live_row),
                rect,
                char_width,
                line_height,
                &viewport,
            ),
            Some(live_link)
        );

        assert!(!app_mouse_press_reports_from_snapshot(
            true,
            application_cell_at_pointer(
                pointer(summary_row),
                rect,
                char_width,
                line_height,
                &viewport,
            )
        ));
        assert!(app_mouse_press_reports_from_snapshot(
            true,
            application_cell_at_pointer(
                pointer(live_row),
                rect,
                char_width,
                line_height,
                &viewport,
            )
        ));

        let summary_route = app_mouse_frame_route(true, Some(None), Some(raw_cell));
        assert_eq!(summary_route.lossy_cell, None);
        assert_eq!(summary_route.release_cell, Some(raw_cell));
    }

    #[test]
    fn inactive_mouse_capture_uses_only_its_last_raw_cell_for_release() {
        let last_raw_cell = (2, 7);
        let route = app_mouse_frame_route(false, Some(Some((99, 101))), Some(last_raw_cell));

        assert_eq!(route.lossy_cell, None);
        assert_eq!(route.release_cell, Some(last_raw_cell));
    }

    #[test]
    fn pointer_gone_cancels_a_held_workspace_drag_but_release_gets_one_frame_to_commit() {
        assert!(workspace_drag_pointer_cancelled(true, false, false));
        assert!(workspace_drag_pointer_cancelled(false, false, true));
        assert!(!workspace_drag_pointer_cancelled(true, false, true));
        assert!(!workspace_drag_pointer_cancelled(false, true, false));
    }

    #[test]
    fn long_command_notification_gates_mirror_anvil() {
        let config = crate::config::Config {
            notify_long_blocks: true,
            notify_long_block_threshold_ms: 10_000,
            ..crate::config::Config::default()
        };

        // Long enough, unwatched, real command: notify.
        assert!(should_notify_long_command(
            &config,
            Some("cargo build"),
            Some(10_000),
            false,
        ));
        // Below the threshold, or with no measured duration: stay silent.
        assert!(!should_notify_long_command(
            &config,
            Some("cargo build"),
            Some(9_999),
            false,
        ));
        assert!(!should_notify_long_command(
            &config,
            Some("cargo build"),
            None,
            false,
        ));
        // anvil's background blocks carry an empty command line and never
        // notify; the same holds for a missing or whitespace-only command.
        assert!(!should_notify_long_command(
            &config,
            None,
            Some(60_000),
            false
        ));
        assert!(!should_notify_long_command(
            &config,
            Some("   "),
            Some(60_000),
            false,
        ));
        // A completion the user watched on a focused visible pane is silent.
        assert!(!should_notify_long_command(
            &config,
            Some("cargo build"),
            Some(60_000),
            true,
        ));

        let degraded = crate::terminal::CompletedCommandEvent {
            start_mark_seen: false,
            completion_provenance: crate::block_mode::CompletionProvenance::ShellReported,
            completed: crate::terminal::CompletedCommandOutput {
                id: "bare-d".into(),
                session_id: None,
                seq: None,
                started_at_ms: None,
                command: Some("cargo build".into()),
                cwd: None,
                exit_code: Some(0),
                duration_ms: Some(60_000),
                output: String::new(),
                output_available: true,
                truncated: false,
                total_bytes: 0,
                agent_generation: None,
            },
        };
        let mut window_started = std::time::Instant::now();
        let mut notifications_in_window = 0;
        maybe_notify_long_command(
            &config,
            &mut window_started,
            &mut notifications_in_window,
            &degraded,
            false,
        );
        assert_eq!(notifications_in_window, 0);

        let disabled = crate::config::Config {
            notify_long_blocks: false,
            ..config
        };
        assert!(!should_notify_long_command(
            &disabled,
            Some("cargo build"),
            Some(60_000),
            false,
        ));
    }

    #[test]
    fn notification_rate_window_resets_only_after_it_elapses() {
        let mut window_started = std::time::Instant::now();
        let mut in_window = super::MAX_NOTIFICATIONS_PER_WINDOW;

        // Inside the window the exhausted counter stays exhausted.
        roll_notification_rate_window(&mut window_started, &mut in_window);
        assert_eq!(in_window, super::MAX_NOTIFICATIONS_PER_WINDOW);

        // Once the window has elapsed the counter resets.
        window_started = std::time::Instant::now() - super::NOTIFICATION_RATE_WINDOW;
        roll_notification_rate_window(&mut window_started, &mut in_window);
        assert_eq!(in_window, 0);
        assert!(window_started.elapsed() < super::NOTIFICATION_RATE_WINDOW);
    }

    #[test]
    fn desktop_notification_queue_is_bounded_without_spending_failed_rate_slots() {
        let (production_tx, _production_rx) = desktop_notification_channel();
        for index in 0..DESKTOP_NOTIFICATION_QUEUE_CAPACITY {
            production_tx
                .try_send(DesktopNotification {
                    title: index.to_string(),
                    body: String::new(),
                })
                .unwrap();
        }
        assert!(matches!(
            production_tx.try_send(DesktopNotification {
                title: "overflow".to_string(),
                body: String::new(),
            }),
            Err(crossbeam_channel::TrySendError::Full(_))
        ));

        let (tx, rx) = crossbeam_channel::bounded(1);
        let mut window_started = std::time::Instant::now();
        let mut sent = 0;

        show_desktop_notification(
            Some(&tx),
            &mut window_started,
            &mut sent,
            "title".to_string(),
            "body".to_string(),
        );
        assert_eq!(sent, 1);
        let notification = rx.recv().unwrap();
        assert_eq!(notification.title, "title");
        assert_eq!(notification.body, "body");

        tx.try_send(DesktopNotification {
            title: "queue filler".to_string(),
            body: String::new(),
        })
        .unwrap();
        show_desktop_notification(
            Some(&tx),
            &mut window_started,
            &mut sent,
            "dropped".to_string(),
            "full queue".to_string(),
        );
        assert_eq!(sent, 1);

        drop(rx);
        show_desktop_notification(
            Some(&tx),
            &mut window_started,
            &mut sent,
            "dropped".to_string(),
            "closed queue".to_string(),
        );
        assert_eq!(sent, 1);
    }

    /// The confirmation trigger is now `jterm_core`'s, but it must trip at the
    /// same points ember's own predicate did — plus on an embedded paste
    /// marker, which is an injection attempt worth surfacing.
    #[test]
    fn risky_paste_detection_covers_newlines_and_large_single_lines() {
        use jterm_core::pty_input::{classify_paste, should_confirm};
        let risky = |text: &str| {
            should_confirm(
                &classify_paste(text),
                crate::app::state::PASTE_CONFIRM_THRESHOLD_BYTES,
            )
        };

        assert!(!risky("printf safe"));
        assert!(risky("first\nsecond"));
        assert!(risky(
            &"x".repeat(crate::app::state::PASTE_CONFIRM_THRESHOLD_BYTES + 1)
        ));
        assert!(!risky(
            &"x".repeat(crate::app::state::PASTE_CONFIRM_THRESHOLD_BYTES)
        ));
        assert!(risky("ok\x1b[201~rm -rf ~"));
    }

    #[test]
    fn clipboard_visual_spoofing_forces_confirmation_even_when_disabled() {
        let ordinary_unicode = jterm_core::pty_input::classify_paste("printf '雪🙂'");
        assert!(!paste_requires_confirmation(
            PasteOrigin::Clipboard,
            false,
            &ordinary_unicode,
            false,
        ));

        let hidden = jterm_core::pty_input::classify_paste("printf safe\u{202e}hidden");
        assert!(paste_requires_confirmation(
            PasteOrigin::Clipboard,
            false,
            &hidden,
            true,
        ));
    }

    #[test]
    fn paste_normalization_cannot_hide_enter_as_a_carriage_return() {
        use jterm_core::pty_input::{classify_paste, should_confirm};
        assert_eq!(
            normalized_paste_body("first\rsecond\r\nthird", false),
            "first\nsecond\nthird"
        );
        // A lone CR is an executable Enter, so it has to count as a second line
        // for the confirmation policy too.
        assert!(should_confirm(
            &classify_paste("printf risky\r"),
            crate::app::state::PASTE_CONFIRM_THRESHOLD_BYTES
        ));
    }

    #[test]
    fn bracketed_paste_cannot_embed_an_early_terminator() {
        use jterm_core::pty_input::{encode_paste, PasteModes};
        let paste = encode_paste(
            "safe\x1b[201~injected",
            PasteModes { bracketed: true },
            paste_policy(false),
        );
        assert_eq!(paste.bytes, b"\x1b[200~safeinjected\x1b[201~");
        assert_eq!(
            paste
                .bytes
                .windows(b"\x1b[201~".len())
                .filter(|window| *window == b"\x1b[201~")
                .count(),
            1
        );
        // The body kept for the confirmation modal is already defused, so the
        // preview shows exactly what the shell will receive.
        assert_eq!(
            normalized_paste_body("safe\x1b[201~injected", false),
            "safeinjected"
        );
    }

    #[test]
    fn submitted_ui_command_places_enter_after_bracketed_paste() {
        use jterm_core::pty_input::{encode_paste, PasteModes};
        assert_eq!(
            encode_paste(
                "cd '/tmp'\n",
                PasteModes { bracketed: true },
                paste_policy(true)
            )
            .bytes,
            b"\x1b[200~cd '/tmp'\x1b[201~\r"
        );
        assert_eq!(
            encode_paste(
                "cd '/tmp'\n",
                PasteModes { bracketed: false },
                paste_policy(true)
            )
            .bytes,
            b"cd '/tmp'\r"
        );
    }

    /// An approved agent suggestion is still model output: it must be framed
    /// like any other payload so an embedded terminator cannot end the frame and
    /// leave the rest as typed commands. Its Enter stays outside the frame.
    #[test]
    fn a_submitted_agent_command_is_framed_and_cannot_break_out() {
        assert_eq!(
            encode_submitted_command("ls -la", true),
            b"\x1b[200~ls -la\x1b[201~\r"
        );
        assert_eq!(encode_submitted_command("ls -la", false), b"ls -la\r");
        assert_eq!(
            encode_submitted_command("ls\x1b[201~\rrm -rf ~", true),
            b"\x1b[200~ls\nrm -rf ~\x1b[201~\r"
        );
        // Nothing to run means nothing is written — not a bare Enter.
        assert!(encode_submitted_command("", true).is_empty());
    }

    /// One de-fanging pass is not a security boundary: `jterm_core` deletes a
    /// marker and resumes after it, so `ESC [` + `ESC[201~` + `201~` collapses
    /// into a *fresh* `ESC[201~`. The paths that keep control bytes (an approved
    /// agent command, the sidebar `cd`) would then frame that terminator and hand
    /// the shell an early frame close followed by executable lines. Exactly one
    /// terminator may ever leave these encoders.
    #[test]
    fn a_nested_paste_terminator_cannot_be_spliced_back_into_the_frame() {
        let terminators = |bytes: &[u8]| {
            bytes
                .windows(b"\x1b[201~".len())
                .filter(|window| *window == b"\x1b[201~")
                .count()
        };
        // Two levels of nesting; the payload after it is what a single pass
        // would have let the shell run.
        let nested = "\x1b[\x1b[\x1b[201~201~201~\rrm -rf ~";

        let submitted = encode_submitted_command(nested, true);
        assert_eq!(terminators(&submitted), 1, "{submitted:?}");
        assert_eq!(submitted, b"\x1b[200~\nrm -rf ~\x1b[201~\r");

        // Same for the body a paste keeps: no marker survives to be framed, and
        // the preview therefore shows what the shell will really receive.
        for submit_after_paste in [false, true] {
            let body = normalized_paste_body(nested, submit_after_paste);
            assert!(!body.contains("\x1b[201~"), "{body:?}");
            assert_eq!(
                normalized_paste_body(&body, submit_after_paste),
                body,
                "the stored body must be a fixed point of the encoder"
            );
        }
    }

    /// The shared prompt boundary strips terminal controls before either
    /// framing branch, while the local validation still rejects visual spoofing.
    #[test]
    fn control_preserving_prompt_policy_is_post_validation_only() {
        assert_eq!(normalized_paste_body("a\x1b[31mb", false), "a[31mb");
        assert_eq!(normalized_paste_body("a\x1b[31mb", true), "a[31mb");

        let sanitized = crate::review_text::sanitize_prompt_payload(
            "a\x1b[31mb",
            crate::review_text::MAX_PROMPT_INSERT_BYTES,
            crate::review_text::VisualSpoofDisposition::Reject,
        )
        .unwrap();
        assert_eq!(normalized_paste_body(&sanitized.text, true), "a[31mb");
        assert!(matches!(
            crate::review_text::sanitize_prompt_payload(
                "safe\u{2066}hidden",
                crate::review_text::MAX_PROMPT_INSERT_BYTES,
                crate::review_text::VisualSpoofDisposition::Reject,
            ),
            Err(crate::review_text::ReviewTextError::VisualSpoof)
        ));
        // Multiline payloads are never truncated: the modal, not silent
        // mangling, is what protects the user here.
        assert_eq!(normalized_paste_body("one\ntwo", false), "one\ntwo");
    }

    /// The stored body is what the accept path re-encodes, in whichever
    /// bracketed-paste mode is live *then*. Both framings must therefore be
    /// reachable from one pending paste, and re-encoding must be a fixed point
    /// so a backpressure retry sends the same bytes rather than a doubly
    /// processed payload.
    #[test]
    fn barrier_retry_reencodes_pending_paste_for_the_delivery_time_mode() {
        use jterm_core::pty_input::{encode_paste, PasteModes};
        let body = normalized_paste_body("echo one\r\necho \x1b[201~two", false);
        assert_eq!(body, "echo one\necho two");

        assert!(matches!(
            ensure_direct_paste_route_available(true, false),
            Err(PasteWriteError::Busy)
        ));
        assert!(matches!(
            ensure_direct_paste_route_available(false, true),
            Err(PasteWriteError::Busy)
        ));
        ensure_direct_paste_route_available(false, false).unwrap();

        // No framed bytes were staged while the route was blocked. A retry
        // after DECSET 2004 changes therefore uses the mode live *now*.
        let framed = encode_paste(&body, PasteModes { bracketed: true }, paste_policy(false));
        assert_eq!(framed.bytes, b"\x1b[200~echo one\necho two\x1b[201~");
        let unframed = encode_paste(&body, PasteModes { bracketed: false }, paste_policy(false));
        assert_eq!(unframed.bytes, body.as_bytes());

        assert_eq!(normalized_paste_body(&body, false), body);
        assert_eq!(
            encode_paste(
                &framed.echo_text,
                PasteModes { bracketed: true },
                paste_policy(false)
            )
            .bytes,
            framed.bytes
        );
    }

    #[test]
    fn synthetic_wheel_deltas_are_saturating_and_bounded() {
        assert_eq!(
            bounded_wheel_step_accumulate(0, f32::INFINITY, usize::MAX),
            64
        );
        assert_eq!(
            bounded_wheel_step_accumulate(0, f32::NEG_INFINITY, usize::MAX),
            -64
        );
        assert_eq!(bounded_wheel_step_accumulate(63, 1000.0, 512), 64);
        assert_eq!(bounded_wheel_step_accumulate(-63, -1000.0, 512), -64);
        assert_eq!(bounded_wheel_step_accumulate(7, f32::NAN, 512), 7);
    }
    fn encoded_test_png(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        let mut value = 0x1234_5678_u32;
        for _ in 0..width as usize * height as usize {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            pixels.extend_from_slice(&value.to_le_bytes());
        }

        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&pixels, width, height, image::ExtendedColorType::Rgba8)
            .unwrap();
        png
    }

    fn kitty_packet_bodies(packet: &[u8]) -> Vec<&str> {
        std::str::from_utf8(packet)
            .unwrap()
            .split("\x1b\\")
            .filter(|body| !body.is_empty())
            .collect()
    }

    #[test]
    fn clipboard_request_guard_releases_the_single_flight_slot() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        drop(ClipboardRequestGuard(std::sync::Arc::clone(&in_flight)));
        assert!(!in_flight.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn osc52_response_is_bounded_before_base64_allocation() {
        let normal = osc52_clipboard_response_with_limit("hello", 64);
        assert_eq!(normal, b"\x1b]52;c;aGVsbG8=\x1b\\");

        let capped = osc52_clipboard_response_with_limit(&"x".repeat(128), 16);
        assert_eq!(capped, b"\x1b]52;c;\x1b\\");
        assert!(capped.len() <= 16);
        assert!(osc52_clipboard_response_with_limit("x", 4).is_empty());
    }

    #[test]
    fn osc52_read_rate_limit_resets_after_its_window() {
        let base = std::time::Instant::now();
        let mut window = base;
        let mut count = 0;
        for _ in 0..MAX_OSC52_READS_PER_WINDOW {
            assert!(osc52_read_rate_limit_allows(base, &mut window, &mut count));
        }
        assert!(!osc52_read_rate_limit_allows(base, &mut window, &mut count));
        assert!(osc52_read_rate_limit_allows(
            base + OSC52_READ_RATE_WINDOW,
            &mut window,
            &mut count,
        ));
        assert_eq!(count, 1);
    }

    #[test]
    fn mouse_drag_reports_require_the_press_time_capture() {
        assert_eq!(reported_capture_button(None), None);
        assert_eq!(reported_capture_button(Some((false, 0))), None);
        assert_eq!(reported_capture_button(Some((true, 2))), Some(2));

        assert_eq!(captured_release_button(None, &[0], false), None);
        assert_eq!(captured_release_button(Some((false, 0)), &[0], false), None);
        assert_eq!(captured_release_button(Some((true, 0)), &[2], true), None);
        assert_eq!(
            captured_release_button(Some((true, 0)), &[0], false),
            Some(0)
        );
        // Some backends only report the button-up state after the pointer
        // leaves the window. The captured last cell still receives release.
        assert_eq!(
            captured_release_button(Some((true, 2)), &[], false),
            Some(2)
        );
    }

    #[test]
    fn captured_mouse_route_uses_current_dimensions_after_output_and_resize() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 80.0));
        let pointer = egui::pos2(99.0, 79.0);
        let mut terminal = crate::terminal::TerminalState::new(10, 8);
        terminal.process_input(b"\x1b[?1000h\x1b[?1006h");

        let (cols, rows) = terminal.get_dimensions();
        let press_cell =
            mouse_cell_for_current_dimensions(Some(pointer), None, rect, 10.0, 10.0, cols, rows);
        assert_eq!(press_cell, (7, 9));
        assert_eq!(
            terminal.get_mouse_report(0, press_cell.1, press_cell.0),
            Some(b"\x1b[<0;10;8M".to_vec())
        );

        // Ordinary PTY output does not retire the captured sequence. After a
        // resize, an outside-window release uses the captured last cell but
        // clamps it to the origin terminal's current dimensions.
        terminal.process_batch(b"output");
        terminal.on_resize(4, 3);
        let (cols, rows) = terminal.get_dimensions();
        let release_cell =
            mouse_cell_for_current_dimensions(None, Some(press_cell), rect, 10.0, 10.0, cols, rows);
        assert_eq!(release_cell, (2, 3));
        assert_eq!(
            terminal.get_mouse_release_report(0, release_cell.1, release_cell.0),
            Some(b"\x1b[<0;4;3m".to_vec())
        );
    }

    #[test]
    fn captured_mouse_route_survives_alt_screen_swap_with_current_dimensions() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(80.0, 40.0));
        let pointer = egui::pos2(79.0, 39.0);
        let mut terminal = crate::terminal::TerminalState::new(8, 4);
        terminal.process_input(b"\x1b[?1000h\x1b[?1006h");
        let press_cell =
            mouse_cell_for_current_dimensions(Some(pointer), None, rect, 10.0, 10.0, 8, 4);
        let press = terminal
            .get_mouse_report(0, press_cell.1, press_cell.0)
            .unwrap();

        terminal.process_input(b"\x1b[?47h");
        terminal.on_resize(3, 2);
        let (cols, rows) = terminal.get_dimensions();
        let release_cell = mouse_cell_for_current_dimensions(
            Some(pointer),
            Some(press_cell),
            rect,
            10.0,
            10.0,
            cols,
            rows,
        );
        let release = terminal
            .get_mouse_release_report(0, release_cell.1, release_cell.0)
            .unwrap();

        assert_eq!(press, b"\x1b[<0;8;4M");
        assert_eq!(release_cell, (1, 2));
        assert_eq!(release, b"\x1b[<0;3;2m");
    }

    #[test]
    fn mouse_control_edges_remain_ordered_and_gate_lossy_reports() {
        use crate::app::state::PendingMouseControlKind::{Press, Release};

        let mut controls = std::collections::VecDeque::new();
        queue_mouse_control(&mut controls, Press, b"press".to_vec());
        queue_mouse_control(&mut controls, Release, b"release".to_vec());
        queue_mouse_control(&mut controls, Press, b"duplicate".to_vec());
        assert_eq!(controls.len(), 2);
        assert_eq!(controls[0].kind, Press);
        assert_eq!(controls[1].kind, Release);

        assert!(!mouse_sequence_allows_lossy(true, false, false, false));
        assert!(mouse_sequence_allows_lossy(true, true, false, true));
        assert!(!mouse_sequence_allows_lossy(true, true, true, true));
        assert!(!mouse_sequence_is_complete(true, true, true, false));
        assert!(mouse_sequence_is_complete(true, true, true, true));
        assert!(mouse_sequence_is_complete(false, false, true, true));
    }

    #[test]
    fn backpressured_mouse_edges_stay_at_the_front_until_both_are_accepted() {
        use crate::app::state::PendingMouseControlKind::{Press, Release};

        let mut controls = std::collections::VecDeque::new();
        queue_mouse_control(&mut controls, Press, b"press".to_vec());
        queue_mouse_control(&mut controls, Release, b"release".to_vec());
        let mut press_accepted = false;

        let blocked: Result<(), ()> =
            flush_pending_mouse_controls(&mut controls, &mut press_accepted, |_| Err(()));
        assert_eq!(blocked, Err(()));
        assert!(!press_accepted);
        assert_eq!(controls.len(), 2);

        let mut admitted = Vec::new();
        flush_pending_mouse_controls(&mut controls, &mut press_accepted, |bytes| {
            admitted.extend_from_slice(bytes);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(admitted, b"pressrelease");
        assert!(press_accepted);
        assert!(controls.is_empty());
    }

    #[test]
    fn slow_osc_paste_holds_click_edges_and_drops_wheel_until_its_reply_is_ahead() {
        use crate::app::state::PendingMouseControlKind::{Press, Release};

        let barriers = crate::session_manager::SessionInputBarriers::default();
        let responses =
            crate::session_manager::ProtocolResponseSender::new(egui::Context::default());
        let guard = barriers.acquire("mouse-session".to_owned());
        let mut controls = std::collections::VecDeque::new();
        queue_mouse_control(&mut controls, Press, b"click".to_vec());
        queue_mouse_control(&mut controls, Release, b"release".to_vec());
        let mut press_accepted = false;
        let mut admitted = Vec::new();

        let producer_blocked =
            mouse_protocol_input_is_blocked("mouse-session", &barriers, &responses);
        if !producer_blocked {
            flush_pending_mouse_controls(&mut controls, &mut press_accepted, |bytes| {
                admitted.extend_from_slice(bytes);
                Ok::<_, ()>(())
            })
            .unwrap();
        }
        assert!(admitted.is_empty());
        assert_eq!(controls.len(), 2, "click/release are durable edges");
        assert!(!mouse_lossy_reports_allowed(producer_blocked, true));

        // Publication happens before the producer guard is released. Even
        // after release, a backpressured protocol response remains the gate.
        responses.try_enqueue(b"paste reply".to_vec()).unwrap();
        drop(guard);
        assert!(mouse_protocol_input_is_blocked(
            "mouse-session",
            &barriers,
            &responses
        ));
        assert!(!mouse_lossy_reports_allowed(true, true));

        // Once that route has no producer or pending response, the original
        // stateful sequence is admitted in order; lossy wheel becomes eligible.
        let drained_responses =
            crate::session_manager::ProtocolResponseSender::new(egui::Context::default());
        let blocked =
            mouse_protocol_input_is_blocked("mouse-session", &barriers, &drained_responses);
        assert!(!blocked);
        flush_pending_mouse_controls(&mut controls, &mut press_accepted, |bytes| {
            admitted.extend_from_slice(bytes);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(admitted, b"clickrelease");
        assert!(press_accepted);
        assert!(mouse_lossy_reports_allowed(blocked, true));
    }

    #[test]
    fn prior_render_cursor_bytes_keep_their_route_and_precede_current_input() {
        let mut target = Some(17);
        let mut cursor = b"\x1b[C".to_vec();
        let (routed_to, mut ordered) =
            take_tagged_cursor_move(&mut target, &mut cursor).expect("tagged cursor input");
        assert_eq!(routed_to, 17);
        ordered.extend_from_slice("中".as_bytes());
        ordered.push(0x03);
        assert_eq!(
            ordered,
            [b"\x1b[C".as_slice(), "中".as_bytes(), &[0x03]].concat()
        );
        assert!(target.is_none());
        assert!(cursor.is_empty());

        let mut stale_target = None;
        let mut stale = b"\x1b[D".to_vec();
        assert!(take_tagged_cursor_move(&mut stale_target, &mut stale).is_none());
        assert!(stale.is_empty());
    }

    #[test]
    fn captured_primary_release_never_falls_back_to_the_replacement_terminal() {
        assert_eq!(
            primary_copy_route(Some((false, false, 0)), true, true),
            PrimaryCopyRoute::CapturedLocal
        );
        assert_eq!(
            primary_copy_route(Some((true, false, 0)), true, true),
            PrimaryCopyRoute::SuppressCaptured
        );
        assert_eq!(
            primary_copy_route(Some((false, true, 0)), true, true),
            PrimaryCopyRoute::SuppressCaptured
        );
        assert_eq!(
            primary_copy_route(None, false, true),
            PrimaryCopyRoute::Generic
        );
        assert_eq!(
            primary_copy_route(Some((true, false, 2)), true, false),
            PrimaryCopyRoute::None
        );
    }

    #[test]
    fn osc_5522_response_chunks_data_at_4096_bytes() {
        let data = vec![0x5a; OSC_5522_DATA_CHUNK_BYTES + 1];
        let response = clipboard_5522_response_for_mime("application/octet-stream", &data);
        let response = String::from_utf8(response).unwrap();
        let packets: Vec<&str> = response
            .split("\x1b\\")
            .filter(|packet| packet.contains("status=DATA"))
            .collect();
        assert_eq!(packets.len(), 2);

        let decoded: Vec<Vec<u8>> = packets
            .iter()
            .map(|packet| {
                let payload = packet.rsplit_once(';').unwrap().1;
                base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .unwrap()
            })
            .collect();
        assert_eq!(decoded[0].len(), OSC_5522_DATA_CHUNK_BYTES);
        assert_eq!(decoded[1].len(), 1);
        assert_eq!(decoded.concat(), data);
        assert_eq!(response.matches("status=OK").count(), 1);
        assert_eq!(response.matches("status=DONE").count(), 1);
    }

    #[test]
    fn osc_5522_response_rejects_data_over_policy_limit() {
        let response = clipboard_5522_response_for_mime_with_limit("text/plain", b"12345", 4);
        assert_eq!(
            String::from_utf8(response).unwrap(),
            "\x1b]5522;type=read:status=EPERM\x1b\\"
        );
    }

    #[test]
    fn kitty_png_payload_uses_standard_bounded_chunks_and_put() {
        let png = encoded_test_png(64, 64);
        let packet = kitty_graphics_payload("image/png", &png).unwrap();
        let bodies = kitty_packet_bodies(&packet);
        assert!(
            bodies.len() >= 3,
            "expected multiple transfer chunks plus put"
        );

        let transfer_bodies = &bodies[..bodies.len() - 1];
        let mut encoded = String::new();
        let mut image_id = None;
        for (index, body) in transfer_bodies.iter().enumerate() {
            let body = body.strip_prefix("\x1b_G").unwrap();
            let (control, payload) = body.split_once(';').unwrap();
            assert!(payload.len() <= KITTY_BASE64_CHUNK_BYTES);
            let expected_more = u8::from(index + 1 < transfer_bodies.len());
            let expected_more_control = format!("m={expected_more}");
            assert!(control.split(',').any(|part| part == expected_more_control));

            if index == 0 {
                assert!(control.split(',').any(|part| part == "a=t"));
                assert!(control.split(',').any(|part| part == "f=100"));
                image_id = control
                    .split(',')
                    .find_map(|part| part.strip_prefix("i="))
                    .map(str::to_owned);
            } else {
                assert_eq!(control, expected_more_control);
            }
            encoded.push_str(payload);
        }

        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(&encoded)
                .unwrap(),
            png
        );
        let expected_put = format!("\x1b_Ga=p,i={}", image_id.unwrap());
        assert_eq!(bodies.last().copied(), Some(expected_put.as_str()));
    }

    #[test]
    fn kitty_png_payload_marks_a_single_transfer_final_and_keeps_padding() {
        let mut png = encoded_test_png(1, 1);
        while png.len().is_multiple_of(3) {
            png.push(0);
        }
        let packet = kitty_graphics_payload("image/png", &png).unwrap();
        let bodies = kitty_packet_bodies(&packet);
        assert_eq!(bodies.len(), 2);
        let (_, payload) = bodies[0].split_once(';').unwrap();
        assert!(bodies[0].contains("a=t,f=100"));
        assert!(bodies[0].contains("m=0"));
        assert!(payload.ends_with('='));
        assert!(bodies[1].contains("a=p"));
    }

    #[test]
    fn kitty_image_paste_rejects_non_png_and_invalid_png() {
        assert!(kitty_graphics_payload("image/jpeg", b"\xff\xd8\xff\xe0").is_none());
        assert!(kitty_graphics_payload("image/png", b"not a png").is_none());
    }

    #[test]
    fn copy_event_becomes_ctrl_c_key_event() {
        let modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };

        let event = shortcut_event_to_key_event(egui::Event::Copy, modifiers)
            .expect("copy event should map to a key event");

        assert_eq!(
            event,
            egui::Event::Key {
                key: egui::Key::C,
                physical_key: Some(egui::Key::C),
                pressed: true,
                repeat: false,
                modifiers,
            }
        );
    }

    #[test]
    fn paste_event_becomes_ctrl_shift_v_key_event_when_restored() {
        let modifiers = egui::Modifiers {
            ctrl: true,
            shift: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![egui::Event::Paste("ignored clipboard payload".to_owned())];

        normalize_terminal_shortcut_events(&mut events, modifiers, true, false, false);

        assert_eq!(
            events,
            vec![egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: true,
                repeat: false,
                modifiers,
            }]
        );
    }

    #[test]
    fn ctrl_link_release_uses_press_target_and_drag_or_multiclick_cancels() {
        let origin = egui::pos2(10.0, 10.0);
        assert!(!link_activation_dragged(origin, egui::pos2(12.0, 11.0)));
        assert!(link_activation_dragged(origin, egui::pos2(40.0, 10.0)));
        // Modifier state is intentionally absent: Ctrl-only eligibility was
        // decided at press, so releasing Ctrl before mouse-up cannot retarget.
        assert!(link_activation_release_allowed(
            "session-a",
            "session-a",
            false,
            false,
        ));
        assert!(!link_activation_release_allowed(
            "session-a",
            "session-b",
            false,
            false,
        ));
        assert!(!link_activation_release_allowed(
            "session-a",
            "session-a",
            true,
            false,
        ));
        assert!(!link_activation_release_allowed(
            "session-a",
            "session-a",
            false,
            true,
        ));
        assert!(!link_activation_ready(None, 1.0, 0.3));
        assert!(!link_activation_ready(Some(1.0), 1.29, 0.3));
        assert!(
            link_activation_ready(Some(1.0), 1.3, 0.3),
            "single-click open waits until a second click can no longer promote the gesture"
        );
    }

    #[test]
    fn host_link_press_suppresses_app_mouse_and_second_button_fails_closed() {
        assert!(mouse_press_reports_to_app(true, false, true, false));
        assert!(!mouse_press_reports_to_app(true, false, true, true));
        assert_eq!(
            primary_copy_route(Some((false, true, 0)), true, true),
            PrimaryCopyRoute::SuppressCaptured,
            "a consumed host-link release must not copy a stale local selection to PRIMARY"
        );
        assert!(!mouse_press_reports_to_app(true, true, true, false));
        assert!(!mouse_press_reports_to_app(true, false, false, false));

        assert!(mouse_capture_accepts_new_press(false));
        assert!(
            !mouse_capture_accepts_new_press(true),
            "left-down capture rejects an interleaved right/middle press instead of overwriting its release route"
        );
    }

    #[test]
    fn semantic_clipboard_events_are_dropped_when_not_restored() {
        let modifiers = egui::Modifiers::default();
        let mut events = vec![
            egui::Event::Copy,
            egui::Event::Paste("ignored".to_owned()),
            egui::Event::Text("a".to_owned()),
        ];

        normalize_terminal_shortcut_events(&mut events, modifiers, false, false, false);

        assert_eq!(events, vec![egui::Event::Text("a".to_owned())]);
    }

    #[test]
    fn settings_text_edit_keeps_semantic_clipboard_events() {
        let modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![
            egui::Event::Copy,
            egui::Event::Cut,
            egui::Event::Paste("pasted API key".to_owned()),
        ];
        let expected = events.clone();

        normalize_terminal_shortcut_events(&mut events, modifiers, true, false, true);

        assert_eq!(events, expected);
    }

    #[test]
    fn semantic_paste_event_is_preserved_when_requested() {
        let modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![egui::Event::Paste("ignored".to_owned())];

        normalize_terminal_shortcut_events(&mut events, modifiers, true, true, false);

        assert_eq!(events, vec![egui::Event::Paste("ignored".to_owned())]);
    }

    #[test]
    fn ctrl_shift_v_remains_explicit_text_paste_with_osc_5522_enabled() {
        let modifiers = egui::Modifiers {
            ctrl: true,
            shift: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![egui::Event::Paste("clipboard text".to_owned())];

        normalize_terminal_shortcut_events(&mut events, modifiers, true, true, false);

        assert_eq!(
            events,
            vec![egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: true,
                repeat: false,
                modifiers,
            }]
        );
    }

    #[test]
    fn released_ctrl_shift_v_recovers_shift_for_text_paste_routing() {
        let event_modifiers = egui::Modifiers {
            ctrl: true,
            shift: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![
            egui::Event::Paste("clipboard text".to_owned()),
            egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: false,
                repeat: false,
                modifiers: event_modifiers,
            },
        ];

        let recovered = semantic_paste_modifiers(&events, egui::Modifiers::NONE);
        normalize_terminal_shortcut_events(&mut events, recovered, true, true, false);

        assert!(events.iter().any(|event| {
            matches!(event,
                egui::Event::Key {
                    key: egui::Key::V,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.ctrl && modifiers.shift
            )
        }));
        assert!(!events
            .iter()
            .any(|event| matches!(event, egui::Event::Paste(_))));
    }

    #[test]
    fn released_plain_ctrl_v_stays_an_osc_5522_paste_event() {
        let event_modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![
            egui::Event::Paste("clipboard text".to_owned()),
            egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: false,
                repeat: false,
                modifiers: event_modifiers,
            },
        ];

        let recovered = semantic_paste_modifiers(&events, egui::Modifiers::NONE);
        normalize_terminal_shortcut_events(&mut events, recovered, true, true, false);

        assert!(events
            .iter()
            .any(|event| matches!(event, egui::Event::Paste(_))));
        assert!(!recovered.shift);
    }

    #[test]
    fn image_only_clipboard_restores_ctrl_v_after_ctrl_is_released() {
        let mut state = PasteKeyState::default();
        let expected_modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![egui::Event::Key {
            key: egui::Key::V,
            physical_key: Some(egui::Key::V),
            pressed: false,
            repeat: false,
            // This is what arrives when Ctrl is released before V.
            modifiers: egui::Modifiers::NONE,
        }];

        assert!(restore_missing_image_paste_key_event(
            &mut events,
            &mut state
        ));
        assert_eq!(
            events[0],
            egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: true,
                repeat: false,
                modifiers: expected_modifiers,
            }
        );
    }

    #[test]
    fn image_paste_restoration_does_not_duplicate_existing_paste_input() {
        let mut state = PasteKeyState::default();
        let modifiers = egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        };
        let mut events = vec![
            egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: true,
                repeat: false,
                modifiers,
            },
            egui::Event::Key {
                key: egui::Key::V,
                physical_key: Some(egui::Key::V),
                pressed: false,
                repeat: false,
                modifiers,
            },
        ];

        assert!(!restore_missing_image_paste_key_event(
            &mut events,
            &mut state
        ));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn semantic_text_paste_suppresses_v_release_in_a_later_frame() {
        let mut state = PasteKeyState::default();
        let mut paste_frame = vec![egui::Event::Paste("clipboard text".to_owned())];
        assert!(!restore_missing_image_paste_key_event(
            &mut paste_frame,
            &mut state
        ));

        let mut release_frame = vec![egui::Event::Key {
            key: egui::Key::V,
            physical_key: Some(egui::Key::V),
            pressed: false,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }];
        assert!(!restore_missing_image_paste_key_event(
            &mut release_frame,
            &mut state
        ));
        assert_eq!(release_frame.len(), 1);
        assert!(matches!(
            release_frame[0],
            egui::Event::Key {
                key: egui::Key::V,
                pressed: false,
                ..
            }
        ));
    }

    #[test]
    fn ordinary_v_press_suppresses_release_restoration_across_frames() {
        let mut state = PasteKeyState::default();
        let mut press_frame = vec![egui::Event::Key {
            key: egui::Key::V,
            physical_key: Some(egui::Key::V),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }];
        assert!(!restore_missing_image_paste_key_event(
            &mut press_frame,
            &mut state
        ));

        let mut release_frame = vec![egui::Event::Key {
            key: egui::Key::V,
            physical_key: Some(egui::Key::V),
            pressed: false,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }];
        assert!(!restore_missing_image_paste_key_event(
            &mut release_frame,
            &mut state
        ));
        assert_eq!(release_frame.len(), 1);
    }

    /// A fallback slot must stay empty when its font is missing.
    ///
    /// `fc-match` answers every query with *something*, so the unverified
    /// lookup hands back the default monospace file for a font nobody
    /// installed. That substitute would then occupy the icon or CJK slot while
    /// carrying none of the glyphs the slot exists to supply.
    #[test]
    fn a_missing_family_resolves_to_nothing_rather_than_a_substitute() {
        assert_eq!(
            fontconfig_match_family_file("Definitely Not An Installed Family 4f2b"),
            None
        );
    }
}
