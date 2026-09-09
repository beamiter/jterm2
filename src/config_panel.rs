use crate::config::{AppRendererType, Config, TabBarPosition};
use crate::theme::Theme;
use crate::theme::ThemeExt as _;
use egui::{Color32, RichText};
use jterm_core::jsh_remote::RemoteHostConfig;

fn accent_color(theme: &Theme) -> Color32 {
    Theme::rgb_to_color32(theme.tabbar.active_border)
}

fn muted_text_color(theme: &Theme) -> Color32 {
    Theme::rgb_to_color32(theme.ui.text_disabled)
}

fn warning_color(theme: &Theme) -> Color32 {
    Theme::rgb_to_color32(theme.terminal.ansi_colors[11])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigTab {
    Font,
    Appearance,
    Advanced,
    Ai,
    Remote,
}

/// One `[[remote_hosts]]` entry under edit. Wraps the full family type so
/// fields the tab does not surface (ssh_args, session, remote_shell,
/// deploy_artifact) round-trip untouched; `user` gets its own buffer because
/// an `Option<String>` is edited as text, with empty meaning None.
#[derive(Clone, Debug)]
struct RemoteHostDraft {
    host: RemoteHostConfig,
    user: String,
}

impl RemoteHostDraft {
    fn from_config(host: &RemoteHostConfig) -> Self {
        Self {
            user: host.user.clone().unwrap_or_default(),
            host: host.clone(),
        }
    }

    /// The entry as apply_to_config would write it: trimmed, empty user →
    /// None. Validation runs on this form so the label matches what Save
    /// actually persists.
    fn applied(&self) -> RemoteHostConfig {
        let user = self.user.trim();
        // Do not clone `self.host.user`: the Settings buffer is authoritative
        // for visible rows, and the embedded copy may still contain a hostile
        // oversized value that the user has just repaired. Every field cloned
        // below was bounded by `precheck_remote_host_draft` first.
        RemoteHostConfig {
            name: self.host.name.trim().to_string(),
            host: self.host.host.trim().to_string(),
            user: (!user.is_empty()).then(|| user.to_string()),
            docker: self.host.docker,
            remote_shell: self.host.remote_shell.clone(),
            session: self.host.session.clone(),
            ssh_args: self.host.ssh_args.clone(),
            deploy: self.host.deploy.clone(),
            deploy_artifact: self.host.deploy_artifact.clone(),
        }
    }

    fn validate_for_apply(&self) -> Result<(), String> {
        crate::config::precheck_remote_host_draft(&self.host, &self.user)?;
        crate::config::validate_remote_host(&self.applied())
    }

    fn template() -> Self {
        Self {
            host: RemoteHostConfig {
                name: String::new(),
                host: String::new(),
                user: None,
                docker: false,
                remote_shell: "jsh".to_string(),
                session: None,
                ssh_args: Vec::new(),
                deploy: "persist".to_string(),
                deploy_artifact: None,
            },
            user: String::new(),
        }
    }
}

pub enum ConfigAction {
    CustomThemeApplied(Box<Theme>),
    DebugPanelToggled(bool),
    SaveRequested,
    ResetToDefaults,
}

pub struct ConfigPanel {
    pub is_open: bool,
    active_tab: ConfigTab,
    // 编辑中的临时值
    edit_font_size: f32,
    edit_font_weight: f32,
    edit_line_spacing: f32,
    edit_padding: f32,
    edit_scrollback_lines: usize,
    edit_scroll_speed: u32,
    edit_font_family: String,
    edit_theme: String,
    edit_restore_session: bool,
    edit_opacity: f32,
    pub edit_debug_overlay: bool,
    edit_gpu_rendering: bool,
    edit_app_renderer: AppRendererType,
    edit_ui_scale: f32,
    edit_tab_bar_position: TabBarPosition,
    edit_paste_confirm: bool,
    edit_osc52_clipboard_write: bool,
    edit_osc52_clipboard_read: bool,
    edit_notify_long_blocks: bool,
    edit_notify_long_block_threshold_ms: u64,
    edit_show_repo_strip: bool,
    edit_bottom_bar: bool,
    edit_block_mode: bool,
    edit_block_compact: bool,
    edit_ai_enabled: bool,
    edit_ai_provider: String,
    edit_ai_model: String,
    edit_ai_base_url: String,
    edit_ai_max_tokens: u32,
    /// 温度草稿：空串 = 用 provider 默认。
    edit_ai_temperature: String,
    edit_ai_redact_secrets: bool,
    edit_ai_stream: bool,
    edit_ai_share_command_context: bool,
    edit_command_correction_enabled: bool,
    /// Configured credential path retained for hand-edited configs. The UI
    /// stores newly entered keys at ember's private default instead of asking
    /// the user to enter a path.
    edit_ai_api_key_file: String,
    /// 待存储的 API key 明文；点击 Save key 写成 0600 文件后立即清空，
    /// 从不进入 config.toml。
    edit_ai_key_draft: String,
    /// 最近一次存 key 的结果提示（成功路径或错误信息）。
    ai_key_store_status: Option<Result<String, String>>,
    edit_agent_max_turns: u32,
    edit_experimental_task_sidebar: bool,
    edit_remote_hosts: Vec<RemoteHostDraft>,
    // 系统字体缓存
    monospace_fonts: Vec<String>,
    all_fonts: Vec<String>,
    available_themes: Vec<String>,
    fonts_loaded: bool,
    // Font filter
    font_filter: String,
    show_all_fonts: bool,
    // Custom theme editor
    custom_themes: Vec<Theme>,
    editing_theme: Option<Theme>,
    custom_theme_name: String,
    custom_theme_error: Option<String>,
    base_theme_for_new: String,
    // 保存编辑状态
    has_changes: bool,
}

impl ConfigPanel {
    pub fn new() -> Self {
        Self {
            is_open: false,
            active_tab: ConfigTab::Font,
            edit_opacity: 1.0,
            edit_font_size: 14.0,
            edit_font_weight: 0.85,
            edit_line_spacing: 1.3,
            edit_padding: 2.0,
            edit_scrollback_lines: 10000,
            edit_scroll_speed: 3,
            edit_font_family: String::new(),
            edit_theme: "dark".to_string(),
            edit_restore_session: false,
            edit_debug_overlay: false,
            edit_gpu_rendering: true,
            edit_app_renderer: AppRendererType::Glow,
            edit_ui_scale: 0.0,
            edit_tab_bar_position: TabBarPosition::default(),
            edit_paste_confirm: true,
            edit_osc52_clipboard_write: true,
            edit_osc52_clipboard_read: false,
            edit_notify_long_blocks: true,
            edit_notify_long_block_threshold_ms: 10_000,
            edit_show_repo_strip: true,
            edit_bottom_bar: jterm_core::bottom_bar::ENABLED_BY_DEFAULT,
            edit_block_mode: true,
            edit_block_compact: false,
            edit_ai_enabled: false,
            edit_ai_provider: "anthropic".to_string(),
            edit_ai_model: String::new(),
            edit_ai_base_url: String::new(),
            edit_ai_max_tokens: 1_024,
            edit_ai_temperature: String::new(),
            edit_ai_redact_secrets: true,
            edit_ai_stream: true,
            edit_ai_share_command_context: false,
            edit_command_correction_enabled: false,
            edit_ai_api_key_file: String::new(),
            edit_ai_key_draft: String::new(),
            ai_key_store_status: None,
            edit_agent_max_turns: 20,
            edit_experimental_task_sidebar: false,
            edit_remote_hosts: Vec::new(),
            monospace_fonts: Vec::new(),
            all_fonts: Vec::new(),
            available_themes: Vec::new(),
            fonts_loaded: false,
            font_filter: String::new(),
            show_all_fonts: false,
            custom_themes: Vec::new(),
            editing_theme: None,
            custom_theme_name: String::new(),
            custom_theme_error: None,
            base_theme_for_new: "dark".to_string(),
            has_changes: false,
        }
    }

    pub fn refresh_theme_list(&mut self) {
        self.custom_themes = Theme::load_custom_themes();
        let mut themes: Vec<String> = Theme::available_themes()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for ct in &self.custom_themes {
            if !themes.contains(&ct.name) {
                themes.push(ct.name.clone());
            }
        }
        self.available_themes = themes;
    }

    pub fn open(&mut self, config: &Config) {
        self.is_open = true;
        self.sync_from_config(config);

        if !self.fonts_loaded {
            // 字体列表是延迟加载且缓存的，首次访问时触发 fc-list
            self.monospace_fonts = Config::get_monospace_fonts().clone();
            self.all_fonts = Config::get_all_fonts().clone();
            if self.monospace_fonts.is_empty() {
                self.monospace_fonts = vec![
                    "SauceCodePro Nerd Font".to_string(),
                    "JetBrains Mono".to_string(),
                    "Fira Code".to_string(),
                    "Monospace".to_string(),
                ];
            }
            self.fonts_loaded = true;
        }
        self.font_filter.clear();
        self.refresh_theme_list();
    }

    pub fn sync_from_config(&mut self, config: &Config) {
        self.has_changes = false;
        self.edit_font_size = config.font_size;
        self.edit_font_weight = config.font_weight;
        self.edit_line_spacing = config.line_spacing;
        self.edit_padding = config.padding;
        self.edit_scrollback_lines = config.scrollback_lines;
        self.edit_scroll_speed = config.scroll_speed;
        self.edit_font_family = config.font_family.clone();
        self.edit_theme = config.theme.clone();
        self.edit_restore_session = config.restore_session;
        self.edit_opacity = config.opacity;
        self.edit_gpu_rendering = config.gpu_rendering;
        self.edit_app_renderer = config.app_renderer;
        self.edit_ui_scale = config.ui_scale.unwrap_or(0.0);
        self.edit_tab_bar_position = config.tab_bar_position;
        self.edit_paste_confirm = config.paste_confirm;
        self.edit_osc52_clipboard_write = config.osc52_clipboard_write;
        self.edit_osc52_clipboard_read = config.osc52_clipboard_read;
        self.edit_notify_long_blocks = config.notify_long_blocks;
        self.edit_notify_long_block_threshold_ms = config.notify_long_block_threshold_ms;
        self.edit_show_repo_strip = config.show_repo_strip;
        self.edit_bottom_bar = config.bottom_bar;
        self.edit_block_mode = config.block_mode;
        self.edit_block_compact = config.block_compact;
        self.edit_ai_enabled = config.ai_enabled;
        self.edit_ai_provider = config.ai_provider.clone();
        self.edit_ai_model = config.ai_model.clone();
        self.edit_ai_base_url = config.ai_base_url.clone();
        self.edit_ai_max_tokens = config.ai_max_tokens;
        self.edit_ai_temperature = config
            .ai_temperature
            .map(|t| format!("{t}"))
            .unwrap_or_default();
        self.edit_ai_redact_secrets = config.ai_redact_secrets;
        self.edit_ai_stream = config.ai_stream;
        self.edit_ai_share_command_context = config.ai_share_command_context;
        self.edit_command_correction_enabled = config.command_correction_enabled;
        self.edit_ai_api_key_file = config.ai_api_key_file.clone().unwrap_or_default();
        // Never carry a secret draft across reopening/resetting the panel.
        self.edit_ai_key_draft.clear();
        self.ai_key_store_status = None;
        self.edit_agent_max_turns = config.agent_max_turns;
        self.edit_experimental_task_sidebar = config.experimental_task_sidebar;
        self.edit_remote_hosts = config
            .remote_hosts
            .iter()
            .map(RemoteHostDraft::from_config)
            .collect();
    }

    /// Apply all buffered edit values to the given Config.
    pub fn apply_to_config(&self, config: &mut Config) {
        config.font_size = self.edit_font_size;
        config.font_weight = self.edit_font_weight;
        config.line_spacing = self.edit_line_spacing;
        config.padding = self.edit_padding;
        config.scrollback_lines = self.edit_scrollback_lines;
        config.scroll_speed = self.edit_scroll_speed;
        config.font_family = self.edit_font_family.clone();
        config.theme = self.edit_theme.clone();
        config.restore_session = self.edit_restore_session;
        config.opacity = self.edit_opacity;
        config.gpu_rendering = self.edit_gpu_rendering;
        config.app_renderer = self.edit_app_renderer;
        config.ui_scale = if self.edit_ui_scale > 0.0 {
            Some(self.edit_ui_scale)
        } else {
            None
        };
        config.tab_bar_position = self.edit_tab_bar_position;
        config.paste_confirm = self.edit_paste_confirm;
        config.osc52_clipboard_write = self.edit_osc52_clipboard_write;
        config.osc52_clipboard_read = self.edit_osc52_clipboard_read;
        config.notify_long_blocks = self.edit_notify_long_blocks;
        config.notify_long_block_threshold_ms = self.edit_notify_long_block_threshold_ms;
        config.show_repo_strip = self.edit_show_repo_strip;
        config.bottom_bar = self.edit_bottom_bar;
        config.block_mode = self.edit_block_mode;
        config.block_compact = self.edit_block_compact;
        config.ai_enabled = self.edit_ai_enabled;
        config.ai_provider = self.edit_ai_provider.clone();
        config.ai_model = self.edit_ai_model.trim().to_string();
        config.ai_base_url = self.edit_ai_base_url.trim().to_string();
        config.ai_max_tokens = self.edit_ai_max_tokens.clamp(64, 32_768);
        config.ai_temperature = self
            .edit_ai_temperature
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|t| t.is_finite() && (0.0..=2.0).contains(t));
        config.ai_redact_secrets = self.edit_ai_redact_secrets;
        config.ai_stream = self.edit_ai_stream;
        config.ai_share_command_context = self.edit_ai_share_command_context;
        config.command_correction_enabled = self.edit_command_correction_enabled;
        config.ai_api_key_file =
            Some(self.edit_ai_api_key_file.trim().to_string()).filter(|path| !path.is_empty());
        config.agent_max_turns = self.edit_agent_max_turns.clamp(1, 100);
        config.experimental_task_sidebar = self.edit_experimental_task_sidebar;
        config.remote_hosts = self
            .edit_remote_hosts
            .iter()
            .enumerate()
            .map(|(index, draft)| {
                if index < crate::config::MAX_REMOTE_HOST_UI_ROWS {
                    draft.applied()
                } else {
                    // Off-view drafts have no editable widgets.  Preserve their
                    // exact owning strings instead of silently trimming them
                    // when an unrelated visible setting is saved.
                    draft.host.clone()
                }
            })
            .collect();
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.editing_theme = None;
    }

    pub fn toggle(&mut self, config: &Config) {
        if self.is_open {
            self.close();
        } else {
            self.open(config);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, theme: &Theme) -> Vec<ConfigAction> {
        let mut actions = Vec::new();

        if !self.is_open {
            return actions;
        }

        let screen_rect = ctx.viewport_rect();
        let panel_width = 580.0;
        let panel_height = 560.0;
        let panel_pos = egui::pos2(
            (screen_rect.width() - panel_width) / 2.0,
            (screen_rect.height() - panel_height) / 3.0,
        );

        egui::Window::new("Settings")
            .title_bar(false)
            .resizable(false)
            .movable(true)
            .default_pos(panel_pos)
            .fixed_size([panel_width, panel_height])
            .frame(egui::Frame {
                fill: Theme::rgb_to_color32(theme.ui.panel_bg),
                stroke: egui::Stroke::new(1.0, Theme::rgb_to_color32(theme.ui.border)),
                inner_margin: egui::Margin::same(12),
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            })
            .show(ctx, |ui| {
                // 标题栏
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("Settings").size(18.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("x").clicked() {
                            self.is_open = false;
                            self.editing_theme = None;
                        }
                    });
                });
                ui.separator();

                // Tab 栏
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_tab, ConfigTab::Font, "Font");
                    ui.selectable_value(&mut self.active_tab, ConfigTab::Appearance, "Appearance");
                    ui.selectable_value(&mut self.active_tab, ConfigTab::Advanced, "Advanced");
                    ui.selectable_value(&mut self.active_tab, ConfigTab::Ai, "AI");
                    ui.selectable_value(&mut self.active_tab, ConfigTab::Remote, "Remote");
                });
                ui.separator();

                // 内容区域
                egui::ScrollArea::vertical()
                    .max_height(panel_height - 140.0)
                    .auto_shrink([false; 2])
                    .show(ui, |ui| match self.active_tab {
                        ConfigTab::Font => {
                            self.render_font_tab(ui, &mut actions, theme);
                        }
                        ConfigTab::Appearance => {
                            self.render_appearance_tab(ui, &mut actions, theme);
                        }
                        ConfigTab::Advanced => {
                            self.render_advanced_tab(ui, &mut actions);
                        }
                        ConfigTab::Ai => {
                            self.render_ai_tab(ui);
                        }
                        ConfigTab::Remote => {
                            self.render_remote_tab(ui, theme);
                        }
                    });

                // 底部按钮栏
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Reset to Defaults").clicked() {
                        actions.push(ConfigAction::ResetToDefaults);
                        self.has_changes = true;
                    }
                    if self.has_changes {
                        ui.label(RichText::new("*").color(accent_color(theme)).size(12.0));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Save").clicked() {
                            actions.push(ConfigAction::SaveRequested);
                            self.has_changes = false;
                        }
                    });
                });
            });

        actions
    }

    fn render_font_tab(
        &mut self,
        ui: &mut egui::Ui,
        _actions: &mut Vec<ConfigAction>,
        theme: &Theme,
    ) {
        ui.label(RichText::new("Font Settings").strong().size(14.0));
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Size:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_font_size, 8.0..=72.0)
                        .step_by(1.0)
                        .show_value(true),
                )
                .changed()
            {
                // Just mark as changed, don't apply yet
                self.has_changes = true;
            }
            ui.label("px");
        });

        ui.horizontal(|ui| {
            ui.label("Weight:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_font_weight, 0.5..=2.0)
                        .step_by(0.05)
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Line Spacing:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_line_spacing, 0.8..=3.0)
                        .step_by(0.05)
                        .show_value(true),
                )
                .changed()
            {
                // Just mark as changed, don't apply yet
                self.has_changes = true;
            }
        });

        ui.separator();

        let current_display = if self.edit_font_family.is_empty() {
            "Default".to_string()
        } else {
            self.edit_font_family.clone()
        };
        ui.horizontal(|ui| {
            ui.label("Current:");
            ui.label(
                RichText::new(&current_display)
                    .strong()
                    .color(accent_color(theme)),
            );
        });

        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.add(
                egui::TextEdit::singleline(&mut self.font_filter)
                    .desired_width(200.0)
                    .hint_text("Filter fonts..."),
            );
            if !self.font_filter.is_empty() && ui.small_button("x").clicked() {
                self.font_filter.clear();
            }
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_all_fonts, "Show all fonts");
            let count = if self.show_all_fonts {
                self.all_fonts.len()
            } else {
                self.monospace_fonts.len()
            };
            ui.label(
                RichText::new(format!("({} fonts)", count))
                    .size(11.0)
                    .color(muted_text_color(theme)),
            );
        });

        ui.add_space(4.0);

        let filter_lower = self.font_filter.to_lowercase();
        let fonts: &Vec<String> = if self.show_all_fonts {
            &self.all_fonts
        } else {
            &self.monospace_fonts
        };

        let show_default = self.font_filter.is_empty() || "default".contains(&filter_lower);
        let matched_fonts: Vec<&String> = fonts
            .iter()
            .filter(|f| filter_lower.is_empty() || f.to_lowercase().contains(&filter_lower))
            .collect();

        let total = matched_fonts.len() + if show_default { 1 } else { 0 };

        if total == 0 {
            ui.label(
                RichText::new("No matching fonts")
                    .italics()
                    .color(muted_text_color(theme)),
            );
        } else {
            let row_height = 22.0;
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .auto_shrink([false; 2])
                .show_rows(ui, row_height, total, |ui, row_range| {
                    for row_idx in row_range {
                        if row_idx == 0 && show_default {
                            let is_selected = self.edit_font_family.is_empty();
                            let resp = ui.selectable_label(is_selected, "Default");
                            if resp.clicked() && !is_selected {
                                self.edit_font_family.clear();
                                self.has_changes = true;
                            }
                            continue;
                        }
                        let font_idx = if show_default { row_idx - 1 } else { row_idx };
                        if let Some(font_name) = matched_fonts.get(font_idx) {
                            let is_selected = self.edit_font_family == **font_name;
                            let label = RichText::new(font_name.as_str());
                            let resp = ui.selectable_label(is_selected, label);
                            if resp.clicked() && !is_selected {
                                self.edit_font_family = (*font_name).clone();
                                self.has_changes = true;
                            }
                        }
                    }
                });
        }

        if !self.edit_font_family.is_empty()
            && !self
                .monospace_fonts
                .iter()
                .any(|f| f == &self.edit_font_family)
            && !self.all_fonts.iter().any(|f| f == &self.edit_font_family)
        {
            ui.colored_label(warning_color(theme), "Font not found in system");
        }

        ui.add_space(2.0);
        ui.label(
            RichText::new("Note: font change requires restart")
                .size(10.0)
                .color(muted_text_color(theme)),
        );
    }

    fn render_appearance_tab(
        &mut self,
        ui: &mut egui::Ui,
        actions: &mut Vec<ConfigAction>,
        theme: &Theme,
    ) {
        ui.label(RichText::new("Appearance Settings").strong().size(14.0));
        ui.separator();

        // Theme selector
        ui.horizontal(|ui| {
            ui.label("Theme:");
            if egui::ComboBox::from_id_salt("theme_selector")
                .selected_text(self.edit_theme.clone())
                .show_ui(ui, |ui| {
                    let mut changed = false;
                    for theme in &self.available_themes {
                        let is_custom = !Theme::is_builtin(theme);
                        let label = if is_custom {
                            format!("{} (custom)", theme)
                        } else {
                            theme.clone()
                        };
                        if ui
                            .selectable_value(&mut self.edit_theme, theme.clone(), label)
                            .changed()
                        {
                            changed = true;
                        }
                    }
                    changed
                })
                .inner
                .unwrap_or(false)
            {
                self.editing_theme = None;
                self.custom_theme_error = None;
                self.has_changes = true;
            }
        });

        // Custom theme management buttons
        ui.horizontal(|ui| {
            // Edit current custom theme
            if !Theme::is_builtin(&self.edit_theme) {
                if ui.button("Edit").clicked() {
                    if let Some(ct) = self
                        .custom_themes
                        .iter()
                        .find(|t| t.name == self.edit_theme)
                    {
                        self.editing_theme = Some(ct.clone());
                        self.custom_theme_name = ct.name.clone();
                        self.custom_theme_error = None;
                    }
                }
                if ui.button("Delete").clicked() {
                    let name = self.edit_theme.clone();
                    match Theme::delete_custom_theme(&name) {
                        Ok(()) => {
                            self.edit_theme = "dark".to_string();
                            self.editing_theme = None;
                            self.custom_theme_error = None;
                            self.refresh_theme_list();
                            self.has_changes = true;
                        }
                        Err(error) => {
                            self.custom_theme_error =
                                Some(format!("Failed to delete theme: {error}"));
                        }
                    }
                }
            }
        });

        if self.editing_theme.is_none() {
            if let Some(error) = &self.custom_theme_error {
                ui.colored_label(warning_color(theme), error);
            }
        }

        ui.add_space(4.0);

        // New custom theme
        if self.editing_theme.is_none() {
            ui.horizontal(|ui| {
                ui.label("Base:");
                egui::ComboBox::from_id_salt("base_theme_for_new")
                    .selected_text(self.base_theme_for_new.clone())
                    .width(120.0)
                    .show_ui(ui, |ui| {
                        for name in Theme::available_themes() {
                            ui.selectable_value(
                                &mut self.base_theme_for_new,
                                name.to_string(),
                                name,
                            );
                        }
                    });
                if ui.button("+ New Custom Theme").clicked() {
                    if let Some(base) = Theme::get_builtin(&self.base_theme_for_new) {
                        let custom_name =
                            self.generate_custom_name(&self.base_theme_for_new.clone());
                        let mut new_theme = base;
                        new_theme.name = custom_name.clone();
                        self.custom_theme_name = custom_name;
                        self.editing_theme = Some(new_theme);
                        self.custom_theme_error = None;
                    }
                }
            });
        }

        ui.separator();

        // Tab bar position
        ui.horizontal(|ui| {
            ui.label("Tab Bar:");
            if ui
                .selectable_value(&mut self.edit_tab_bar_position, TabBarPosition::Top, "Top")
                .changed()
            {
                self.has_changes = true;
            }
            if ui
                .selectable_value(
                    &mut self.edit_tab_bar_position,
                    TabBarPosition::Sidebar,
                    "Sidebar",
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.separator();

        // Padding slider
        ui.horizontal(|ui| {
            ui.label("Padding:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_padding, 0.0..=20.0)
                        .step_by(0.5)
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
            ui.label("px");
        });

        // Opacity slider
        ui.horizontal(|ui| {
            ui.label("Opacity:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_opacity, 0.05..=1.0)
                        .step_by(0.025)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        // UI Scale slider
        ui.horizontal(|ui| {
            ui.label("UI Scale:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_ui_scale, 0.0..=3.0)
                        .step_by(0.1)
                        .custom_formatter(|v, _| {
                            if v <= 0.0 {
                                "Auto (native DPI)".to_string()
                            } else {
                                format!("{:.1}x", v)
                            }
                        })
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        // Theme editor (inline)
        if self.editing_theme.is_some() {
            ui.separator();
            render_theme_editor(
                ui,
                actions,
                &mut self.editing_theme,
                &mut self.custom_theme_name,
                &mut self.edit_theme,
                &mut self.available_themes,
                &mut self.custom_themes,
                &mut self.has_changes,
                &mut self.custom_theme_error,
            );
        }
    }

    fn generate_custom_name(&self, base: &str) -> String {
        let existing: Vec<&str> = self.custom_themes.iter().map(|t| t.name.as_str()).collect();
        for i in 1..100 {
            let name = format!("custom-{}-{}", base, i);
            if !existing.contains(&name.as_str()) {
                return name;
            }
        }
        format!("custom-{}", base)
    }
}

#[allow(clippy::too_many_arguments)]
fn render_theme_editor(
    ui: &mut egui::Ui,
    actions: &mut Vec<ConfigAction>,
    editing_theme: &mut Option<Theme>,
    custom_theme_name: &mut String,
    edit_theme: &mut String,
    available_themes: &mut Vec<String>,
    custom_themes: &mut Vec<Theme>,
    has_changes: &mut bool,
    custom_theme_error: &mut Option<String>,
) {
    let mut changed = false;
    let mut should_cancel = false;
    let mut should_save = false;

    // Name field + action buttons
    ui.horizontal(|ui| {
        ui.label(RichText::new("Theme Editor").strong().size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Cancel").clicked() {
                should_cancel = true;
            }
            if ui.button("Save Theme").clicked() {
                should_save = true;
            }
        });
    });

    if should_cancel {
        *editing_theme = None;
        *custom_theme_error = None;
        return;
    }

    if should_save {
        if let Some(theme) = editing_theme.as_mut() {
            theme.name.clone_from(custom_theme_name);
            if let Err(e) = theme.save_custom_theme() {
                eprintln!("[Theme] Failed to save: {}", e);
                *custom_theme_error = Some(format!("Failed to save theme: {e}"));
            } else {
                edit_theme.clone_from(&theme.name);
                // Refresh theme list
                *custom_themes = Theme::load_custom_themes();
                let mut themes: Vec<String> = Theme::available_themes()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                for ct in custom_themes.iter() {
                    if !themes.contains(&ct.name) {
                        themes.push(ct.name.clone());
                    }
                }
                *available_themes = themes;
                *has_changes = true;
                *custom_theme_error = None;
                *editing_theme = None;
                return;
            }
        }
    }

    ui.horizontal(|ui| {
        ui.label("Name:");
        if ui.text_edit_singleline(custom_theme_name).changed() {
            *custom_theme_error = None;
        }
    });
    if let Some(error) = custom_theme_error.as_deref() {
        ui.colored_label(ui.visuals().error_fg_color, error);
    }

    ui.add_space(4.0);

    let Some(theme) = editing_theme.as_mut() else {
        return;
    };

    // Terminal Colors
    egui::CollapsingHeader::new(RichText::new("Terminal").strong())
        .default_open(true)
        .show(ui, |ui| {
            changed |= color_row_rgb(ui, "Foreground", &mut theme.terminal.foreground);
            changed |= color_row_rgb(ui, "Background", &mut theme.terminal.background);
            changed |= color_row_rgb(ui, "Cursor", &mut theme.terminal.cursor);
            changed |= color_row_rgba(ui, "Selection", &mut theme.terminal.selection);
        });

    // ANSI Colors
    egui::CollapsingHeader::new(RichText::new("ANSI Colors").strong())
        .default_open(false)
        .show(ui, |ui| {
            let names = [
                "Black",
                "Red",
                "Green",
                "Yellow",
                "Blue",
                "Magenta",
                "Cyan",
                "White",
                "Bright Black",
                "Bright Red",
                "Bright Green",
                "Bright Yellow",
                "Bright Blue",
                "Bright Magenta",
                "Bright Cyan",
                "Bright White",
            ];
            ui.label(
                RichText::new("Normal")
                    .size(11.0)
                    .color(muted_text_color(theme)),
            );
            egui::Grid::new("ansi_normal")
                .num_columns(8)
                .spacing([4.0, 2.0])
                .show(ui, |ui| {
                    for (name, color) in names[..8].iter().zip(&mut theme.terminal.ansi_colors[..8])
                    {
                        changed |= color_btn_rgb(ui, name, color);
                    }
                    ui.end_row();
                });
            ui.add_space(4.0);
            ui.label(
                RichText::new("Bright")
                    .size(11.0)
                    .color(muted_text_color(theme)),
            );
            egui::Grid::new("ansi_bright")
                .num_columns(8)
                .spacing([4.0, 2.0])
                .show(ui, |ui| {
                    for (name, color) in names[8..].iter().zip(&mut theme.terminal.ansi_colors[8..])
                    {
                        changed |= color_btn_rgb(ui, name, color);
                    }
                    ui.end_row();
                });
        });

    // UI Colors
    egui::CollapsingHeader::new(RichText::new("UI").strong())
        .default_open(false)
        .show(ui, |ui| {
            changed |= color_row_rgb(ui, "Window BG", &mut theme.ui.window_bg);
            changed |= color_row_rgb(ui, "Panel BG", &mut theme.ui.panel_bg);
            changed |= color_row_rgb(ui, "Border", &mut theme.ui.border);
            changed |= color_row_rgb(ui, "Text", &mut theme.ui.text);
            changed |= color_row_rgb(ui, "Text Disabled", &mut theme.ui.text_disabled);
        });

    // Tab Bar
    egui::CollapsingHeader::new(RichText::new("Tab Bar").strong())
        .default_open(false)
        .show(ui, |ui| {
            changed |= color_row_rgb(ui, "Background", &mut theme.tabbar.bg);
            changed |= color_row_rgb(ui, "Border", &mut theme.tabbar.border);
            changed |= color_row_rgb(ui, "Inactive Text", &mut theme.tabbar.inactive_text);
            changed |= color_row_rgb(ui, "Active Text", &mut theme.tabbar.active_text);
            changed |= color_row_rgb(ui, "Active Border", &mut theme.tabbar.active_border);
            changed |= color_row_rgb(ui, "Close Btn", &mut theme.tabbar.close_btn_bg);
            changed |= color_row_rgb(ui, "Close Hover", &mut theme.tabbar.close_btn_hover);
        });

    // Scrollbar
    egui::CollapsingHeader::new(RichText::new("Scrollbar").strong())
        .default_open(false)
        .show(ui, |ui| {
            changed |= color_row_rgba(ui, "Track Normal", &mut theme.scrollbar.track_normal);
            changed |= color_row_rgba(ui, "Track Hover", &mut theme.scrollbar.track_hover);
            changed |= color_row_rgba(ui, "Track Drag", &mut theme.scrollbar.track_drag);
            changed |= color_row_rgba(ui, "Thumb Normal", &mut theme.scrollbar.thumb_normal);
            changed |= color_row_rgba(ui, "Thumb Hover", &mut theme.scrollbar.thumb_hover);
            changed |= color_row_rgba(ui, "Thumb Drag", &mut theme.scrollbar.thumb_drag);
        });

    // Search
    egui::CollapsingHeader::new(RichText::new("Search").strong())
        .default_open(false)
        .show(ui, |ui| {
            changed |= color_row_rgb(ui, "Background", &mut theme.search.bg);
            changed |= color_row_rgb(ui, "Border", &mut theme.search.border);
            changed |= color_row_rgb(ui, "Text", &mut theme.search.text);
        });

    // Command Palette
    egui::CollapsingHeader::new(RichText::new("Command Palette").strong())
        .default_open(false)
        .show(ui, |ui| {
            changed |= color_row_rgb(ui, "Session", &mut theme.palette.session_color);
            changed |= color_row_rgb(ui, "Edit", &mut theme.palette.edit_color);
            changed |= color_row_rgb(ui, "Search", &mut theme.palette.search_color);
            changed |= color_row_rgb(ui, "Terminal", &mut theme.palette.terminal_color);
            changed |= color_row_rgb(ui, "Window", &mut theme.palette.window_color);
        });

    // Live preview
    if changed {
        actions.push(ConfigAction::CustomThemeApplied(Box::new(theme.clone())));
    }
}

impl ConfigPanel {
    fn render_ai_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("AI & Agent").strong().size(14.0));
        ui.separator();

        if ui
            .checkbox(&mut self.edit_ai_enabled, "Enable AI features")
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "Off by default. Nothing leaves this machine until you enable AI \
                 and explicitly submit a request. Native Codex runs in a pinned \
                 workspace sandbox; accepting command and file-change approvals \
                 is disabled throughout each live native session.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );
        ui.separator();

        ui.label(
            RichText::new("Inline AI provider (does not configure native Codex)")
                .small()
                .strong(),
        );
        ui.label(
            RichText::new(
                "Native tasks use the installed, ChatGPT-authenticated Codex CLI in an isolated empty config; inline provider/model settings and user Codex extensions do not carry into that live runtime.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );

        ui.horizontal(|ui| {
            ui.label("Provider:");
            for (value, label) in [
                ("anthropic", "Anthropic"),
                ("openai-compatible", "OpenAI-compatible"),
                ("ollama", "Ollama"),
            ] {
                if ui
                    .selectable_label(self.edit_ai_provider == value, label)
                    .clicked()
                {
                    self.edit_ai_provider = value.to_string();
                    self.has_changes = true;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Model:");
            if ui
                .add(egui::TextEdit::singleline(&mut self.edit_ai_model).desired_width(260.0))
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Base URL:");
            if ui
                .add(egui::TextEdit::singleline(&mut self.edit_ai_base_url).desired_width(260.0))
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Max output tokens:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_ai_max_tokens, 64..=32_768)
                        .logarithmic(true)
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Temperature:");
            if ui
                .add(
                    egui::TextEdit::singleline(&mut self.edit_ai_temperature)
                        .hint_text("provider default")
                        .desired_width(80.0),
                )
                .changed()
            {
                self.has_changes = true;
            }
            ui.label("0.0–2.0, empty = provider default");
        });

        let default_key_path = jterm_core::ai::default_api_key_path();
        ui.horizontal(|ui| {
            ui.label("API key:");
            let key_hint = if self.edit_ai_api_key_file.is_empty() {
                "Paste API key"
            } else {
                "******** (stored; enter to replace)"
            };
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.edit_ai_key_draft)
                    .password(true)
                    .hint_text(key_hint)
                    .desired_width(260.0),
            );
            if response.changed() {
                self.ai_key_store_status = None;
            }
            if ui
                .add_enabled(
                    !self.edit_ai_key_draft.trim().is_empty(),
                    egui::Button::new("Save key"),
                )
                .clicked()
            {
                // The environment override remains read-only. Settings-entered
                // keys always use ember's private per-user default.
                let path = default_key_path.clone();
                self.ai_key_store_status = Some(match crate::persistence_file::write_api_key_file(
                    &path,
                    &self.edit_ai_key_draft,
                ) {
                    Ok(()) => {
                        self.edit_ai_key_draft.clear();
                        self.edit_ai_api_key_file = path.clone();
                        self.has_changes = true;
                        Ok(format!("Key stored in {path}"))
                    }
                    Err(error) => Err(error.to_string()),
                });
            }
        });
        ui.label(
            RichText::new(format!(
                "Keys entered here are saved to {default_key_path} with owner-only permissions. The value is masked and is never written to config.toml. Environment API key variables still take precedence."
            ))
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );
        if let Some(status) = &self.ai_key_store_status {
            let (text, color) = match status {
                Ok(message) => (message.as_str(), ui.visuals().weak_text_color()),
                Err(error) => (error.as_str(), ui.visuals().error_fg_color),
            };
            ui.label(RichText::new(text).size(11.0).color(color));
        }
        ui.separator();

        if ui
            .checkbox(
                &mut self.edit_ai_redact_secrets,
                "Redact secrets in attached command context",
            )
            .changed()
        {
            self.has_changes = true;
        }

        if ui
            .checkbox(
                &mut self.edit_ai_stream,
                "Stream AI replies as they generate",
            )
            .on_hover_text(
                "Turn this off behind a proxy that buffers server-sent events, or \
                 for an endpoint that mishandles stream: true — replies then \
                 arrive in one piece instead of never arriving.",
            )
            .changed()
        {
            self.has_changes = true;
        }

        if ui
            .checkbox(
                &mut self.edit_ai_share_command_context,
                "Allow cloud AI to receive command context",
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "A direct loopback Ollama endpoint can attach semantic context without this \
                 opt-in; inherited HTTP proxy settings disable that exemption. Enable it before \
                 Anthropic, OpenAI-compatible, or remote Ollama requests send the command, \
                 working directory, and captured output. Native Codex redaction applies to this \
                 initial attachment; Codex may separately send worktree files and tool output.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );

        // anvil/forge parity: the correction row greys out with the AI master
        // switch, because a disabled monitor must not look configurable.
        if ui
            .add_enabled(
                self.edit_ai_enabled,
                egui::Checkbox::new(
                    &mut self.edit_command_correction_enabled,
                    "Offer corrections for failed commands",
                ),
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "When a command fails with a likely typo (unknown command, subcommand, option, or \
                 package name), show an editable correction card. Nothing is inserted or run \
                 without an explicit click; only a candidate verified against this host's PATH or \
                 APT index can be run directly. Local evidence never leaves the machine; the AI \
                 fallback requires the command-context sharing consent above.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );

        ui.horizontal(|ui| {
            ui.label("Inline Agent turn budget:");
            if ui
                .add(egui::Slider::new(&mut self.edit_agent_max_turns, 1..=100).show_value(true))
                .changed()
            {
                self.has_changes = true;
            }
        });
        ui.separator();
        if ui
            .checkbox(
                &mut self.edit_experimental_task_sidebar,
                "Show experimental Tasks dashboard",
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "Enables isolated task worktrees and the native Codex app-server dashboard. A live session can run sequential review-feedback turns but cannot resume after it stops. This does not grant cloud command-context sharing.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );
        ui.label(
            RichText::new(
                "Native Codex writes are confined to the task worktree, but its current workspace-write sandbox may read other host files. Start it only when that provider access is acceptable.",
            )
            .size(11.0)
            .color(ui.visuals().warn_fg_color),
        );
    }

    fn render_advanced_tab(&mut self, ui: &mut egui::Ui, actions: &mut Vec<ConfigAction>) {
        ui.label(RichText::new("Advanced Settings").strong().size(14.0));
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Scrollback Lines:");
            if ui
                .add(
                    egui::Slider::new(&mut self.edit_scrollback_lines, 100..=100_000)
                        .logarithmic(true)
                        .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Scroll Speed:");
            if ui
                .add(egui::Slider::new(&mut self.edit_scroll_speed, 1..=10).show_value(true))
                .changed()
            {
                self.has_changes = true;
            }
        });

        ui.separator();

        if ui
            .checkbox(
                &mut self.edit_restore_session,
                "Restore sessions on startup",
            )
            .changed()
        {
            self.has_changes = true;
        }

        ui.separator();

        ui.label(RichText::new("Notifications").strong());
        if ui
            .checkbox(
                &mut self.edit_notify_long_blocks,
                "Notify when a long command finishes unwatched",
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.horizontal(|ui| {
            ui.label("Long-command threshold (ms):");
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.edit_notify_long_block_threshold_ms,
                        1_000..=600_000,
                    )
                    .logarithmic(true)
                    .show_value(true),
                )
                .changed()
            {
                self.has_changes = true;
            }
        });

        if ui
            .checkbox(
                &mut self.edit_show_repo_strip,
                "Show git branch/dirty state in pane headers",
            )
            .changed()
        {
            self.has_changes = true;
        }

        if ui
            .checkbox(
                &mut self.edit_bottom_bar,
                "Show the bottom status bar (cwd, git, last command)",
            )
            .changed()
        {
            self.has_changes = true;
        }

        if ui
            .checkbox(
                &mut self.edit_block_mode,
                "Show command cards (status stripe, border, outcome badge)",
            )
            .changed()
        {
            self.has_changes = true;
        }
        if ui
            .add_enabled(
                self.edit_block_mode,
                egui::Checkbox::new(&mut self.edit_block_compact, "Compact Block Spacing"),
            )
            .on_disabled_hover_text("Enable command cards first")
            .changed()
        {
            self.has_changes = true;
        }

        ui.separator();

        ui.label(RichText::new("Security").strong());
        if ui
            .checkbox(
                &mut self.edit_paste_confirm,
                "Confirm multiline or large pastes",
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "Recommended. Shows a preview before potentially dangerous text is sent.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );

        if ui
            .checkbox(
                &mut self.edit_osc52_clipboard_write,
                "Allow terminal apps to write the system clipboard (OSC 52)",
            )
            .changed()
        {
            self.has_changes = true;
        }

        if ui
            .checkbox(
                &mut self.edit_osc52_clipboard_read,
                "Allow terminal apps to read the system clipboard (OSC 52)",
            )
            .changed()
        {
            self.has_changes = true;
        }
        ui.label(
            RichText::new(
                "Warning: clipboard reads send its contents to programs running in the terminal. \
                 Enable this only for trusted workflows.",
            )
            .size(11.0)
            .color(ui.visuals().warn_fg_color),
        );

        ui.separator();

        // Debug overlay toggles immediately (no save needed)
        if ui
            .checkbox(&mut self.edit_debug_overlay, "Show debug overlay (F12)")
            .changed()
        {
            actions.push(ConfigAction::DebugPanelToggled(self.edit_debug_overlay));
        }

        ui.separator();

        ui.label(RichText::new("App Renderer").strong());
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.edit_app_renderer, AppRendererType::Glow, "Glow")
                .changed()
            {
                self.has_changes = true;
            }
            if ui
                .radio_value(&mut self.edit_app_renderer, AppRendererType::Wgpu, "Wgpu")
                .changed()
            {
                self.has_changes = true;
            }
        });
        if self.edit_app_renderer == AppRendererType::Glow {
            let muted = ui.visuals().weak_text_color();
            ui.label(
                RichText::new(
                    "Glow uses less memory. Terminal GPU rendering is disabled in this mode.",
                )
                .size(11.0)
                .color(muted),
            );
        }

        ui.separator();

        if ui
            .add_enabled(
                self.edit_app_renderer == AppRendererType::Wgpu,
                egui::Checkbox::new(&mut self.edit_gpu_rendering, "GPU rendering"),
            )
            .changed()
        {
            self.has_changes = true;
        }

        if self.edit_app_renderer != AppRendererType::Wgpu {
            self.edit_gpu_rendering = false;
        }
    }

    fn render_remote_tab(&mut self, ui: &mut egui::Ui, theme: &Theme) {
        ui.label(RichText::new("Remote Hosts").strong().size(14.0));
        ui.separator();
        ui.label(
            RichText::new(
                "Destinations for the remote host picker (Ctrl+Shift+S): an ssh \
                 host, or a running container reached with docker exec. \
                 ssh_args, session, remote_shell and deploy_artifact are kept \
                 as configured; edit those in config.toml.",
            )
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        let mut changed = false;
        let mut delete_index = None;
        for (index, draft) in self
            .edit_remote_hosts
            .iter_mut()
            .take(crate::config::MAX_REMOTE_HOST_UI_ROWS)
            .enumerate()
        {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                // Preflight before deriving labels or constructing the
                // bounded owning value used by shared semantic validation.
                let validation = if index >= crate::config::MAX_REMOTE_HOSTS {
                    Err(format!(
                        "entry #{} exceeds the {}-host active limit; it is retained but unavailable",
                        index + 1,
                        crate::config::MAX_REMOTE_HOSTS
                    ))
                } else {
                    draft.validate_for_apply()
                };
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut draft.host.name)
                                .hint_text("display name (host when empty)")
                                .desired_width(180.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Delete").clicked() {
                            delete_index = Some(index);
                        }
                    });
                });
                ui.horizontal(|ui| {
                    ui.label("Host:");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut draft.host.host)
                                .hint_text("ssh host or container name")
                                .desired_width(180.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label("User:");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut draft.user)
                                .hint_text("optional")
                                .desired_width(100.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut draft.host.docker, "docker (exec into a container)")
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label("Deploy:");
                    // The empty spelling means Off ([`Deploy::parse`]); the
                    // combo writes it back explicitly so the saved file reads
                    // the way the picker displays it.
                    let current_label = jterm_core::review_input::safe_inline_display(
                        if draft.host.deploy.is_empty() {
                            "off"
                        } else {
                            draft.host.deploy.as_str()
                        },
                        64,
                    );
                    egui::ComboBox::from_id_salt(("remote_host_deploy", index))
                        .selected_text(current_label)
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for mode in ["off", "persist", "incognito"] {
                                let selected = if draft.host.deploy.is_empty() {
                                    mode == "off"
                                } else {
                                    draft.host.deploy == mode
                                };
                                if ui.selectable_label(selected, mode).clicked() && !selected {
                                    draft.host.deploy = mode.to_string();
                                    changed = true;
                                }
                            }
                        });
                });
                // Invalid drafts persist so they can be repaired. The picker,
                // connection path, and remote filesystem all apply this same
                // gate again immediately before use.
                if let Err(problem) = validation {
                    ui.colored_label(warning_color(theme), problem);
                }
            });
            ui.add_space(4.0);
        }
        if self.edit_remote_hosts.len() > crate::config::MAX_REMOTE_HOST_UI_ROWS {
            ui.colored_label(
                warning_color(theme),
                format!(
                    "{} additional drafts remain in config.toml and will be saved unchanged; this panel renders only the first {}.",
                    self.edit_remote_hosts.len() - crate::config::MAX_REMOTE_HOST_UI_ROWS,
                    crate::config::MAX_REMOTE_HOST_UI_ROWS
                ),
            );
        }
        if let Some(index) = delete_index {
            self.edit_remote_hosts.remove(index);
            changed = true;
        }

        let at_capacity = self.edit_remote_hosts.len() >= crate::config::MAX_REMOTE_HOSTS;
        if ui
            .add_enabled(!at_capacity, egui::Button::new("+ Add host"))
            .clicked()
        {
            self.edit_remote_hosts.push(RemoteHostDraft::template());
            changed = true;
        }
        if at_capacity {
            let message = if self.edit_remote_hosts.len() == crate::config::MAX_REMOTE_HOSTS {
                format!(
                    "The {}-host active limit is reached; remove an entry before adding another.",
                    crate::config::MAX_REMOTE_HOSTS
                )
            } else {
                format!(
                    "All {} entries are retained, but only the first {} can be used; remove entries to re-enable Add.",
                    self.edit_remote_hosts.len(),
                    crate::config::MAX_REMOTE_HOSTS
                )
            };
            ui.colored_label(warning_color(theme), message);
        }
        if changed {
            self.has_changes = true;
        }
    }
}

// Helper: RGB color row with label + color picker
fn color_row_rgb(ui: &mut egui::Ui, label: &str, color: &mut [u8; 3]) -> bool {
    let mut changed = false;
    let muted = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.color_edit_button_srgb(color).changed() {
                changed = true;
            }
            ui.label(
                RichText::new(format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]))
                    .size(10.0)
                    .monospace()
                    .color(muted),
            );
        });
    });
    changed
}

// Helper: RGBA color row with label + color picker
fn color_row_rgba(ui: &mut egui::Ui, label: &str, color: &mut [u8; 4]) -> bool {
    let mut changed = false;
    let muted = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut c32 = Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3]);
            if egui::widgets::color_picker::color_edit_button_srgba(
                ui,
                &mut c32,
                egui::color_picker::Alpha::OnlyBlend,
            )
            .changed()
            {
                *color = [c32.r(), c32.g(), c32.b(), c32.a()];
                changed = true;
            }
            ui.label(
                RichText::new(format!(
                    "#{:02X}{:02X}{:02X}{:02X}",
                    color[0], color[1], color[2], color[3]
                ))
                .size(10.0)
                .monospace()
                .color(muted),
            );
        });
    });
    changed
}

// Helper: compact color button (for ANSI grid)
fn color_btn_rgb(ui: &mut egui::Ui, tooltip: &str, color: &mut [u8; 3]) -> bool {
    ui.color_edit_button_srgb(color)
        .on_hover_text(tooltip)
        .changed()
}

#[cfg(test)]
mod tests {
    use super::{ConfigPanel, RemoteHostDraft};
    use crate::config::Config;

    #[test]
    fn clipboard_security_settings_round_trip_through_panel_buffer() {
        let source = Config {
            paste_confirm: false,
            osc52_clipboard_write: false,
            osc52_clipboard_read: true,
            ..Config::default()
        };

        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);

        assert!(!panel.edit_paste_confirm);
        assert!(!panel.edit_osc52_clipboard_write);
        assert!(panel.edit_osc52_clipboard_read);

        panel.edit_paste_confirm = true;
        panel.edit_osc52_clipboard_write = true;
        panel.edit_osc52_clipboard_read = false;

        let mut applied = source;
        panel.apply_to_config(&mut applied);

        assert!(applied.paste_confirm);
        assert!(applied.osc52_clipboard_write);
        assert!(!applied.osc52_clipboard_read);
    }

    #[test]
    fn notification_and_repo_strip_settings_round_trip_through_panel_buffer() {
        let source = Config {
            notify_long_blocks: false,
            notify_long_block_threshold_ms: 30_000,
            show_repo_strip: false,
            ..Config::default()
        };

        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);

        assert!(!panel.edit_notify_long_blocks);
        assert_eq!(panel.edit_notify_long_block_threshold_ms, 30_000);
        assert!(!panel.edit_show_repo_strip);

        panel.edit_notify_long_blocks = true;
        panel.edit_notify_long_block_threshold_ms = 5_000;
        panel.edit_show_repo_strip = true;

        let mut applied = source;
        panel.apply_to_config(&mut applied);

        assert!(applied.notify_long_blocks);
        assert_eq!(applied.notify_long_block_threshold_ms, 5_000);
        assert!(applied.show_repo_strip);
    }

    #[test]
    fn compact_block_spacing_round_trips_through_panel_buffer() {
        let source = Config {
            block_compact: true,
            ..Config::default()
        };
        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);
        assert!(panel.edit_block_compact);

        panel.edit_block_compact = false;
        let mut applied = source;
        panel.apply_to_config(&mut applied);
        assert!(!applied.block_compact);
    }

    #[test]
    fn cloud_command_context_opt_in_round_trips_through_panel_and_config() {
        let source = Config::default();
        assert!(!source.ai_share_command_context);

        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);
        assert!(!panel.edit_ai_share_command_context);

        panel.edit_ai_share_command_context = true;
        let mut applied = source;
        panel.apply_to_config(&mut applied);
        assert!(applied.ai_share_command_context);

        let serialized = toml::to_string_pretty(&applied).expect("serialize config");
        let reparsed: Config = toml::from_str(&serialized).expect("reparse config");
        assert!(reparsed.ai_share_command_context);
    }

    #[test]
    fn api_key_draft_is_ephemeral_and_configured_path_still_round_trips() {
        let source = Config {
            ai_api_key_file: Some("/run/private/ember.key".to_string()),
            ..Config::default()
        };
        let mut panel = ConfigPanel::new();
        panel.edit_ai_key_draft = "must-not-survive".to_string();
        panel.ai_key_store_status = Some(Ok("old status".to_string()));

        panel.sync_from_config(&source);

        assert!(panel.edit_ai_key_draft.is_empty());
        assert!(panel.ai_key_store_status.is_none());
        let mut applied = Config::default();
        panel.apply_to_config(&mut applied);
        assert_eq!(applied.ai_api_key_file, source.ai_api_key_file);
    }

    #[test]
    fn experimental_task_sidebar_round_trips_through_panel() {
        let source = Config::default();
        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);
        assert!(!panel.edit_experimental_task_sidebar);

        panel.edit_experimental_task_sidebar = true;
        let mut applied = source;
        panel.apply_to_config(&mut applied);
        assert!(applied.experimental_task_sidebar);
    }

    #[test]
    fn remote_host_drafts_survive_rename_delete_add_and_a_config_round_trip() {
        let source: Config = toml::from_str(
            r#"
[[remote_hosts]]
name = "build"
host = "dev.example.com"
user = "yj"
deploy = "persist"
ssh_args = ["-p", "22"]

[[remote_hosts]]
name = "sandbox"
host = "myubuntu"
docker = true
deploy = "persist"
"#,
        )
        .expect("parse");

        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);
        assert_eq!(panel.edit_remote_hosts.len(), 2);
        assert_eq!(panel.edit_remote_hosts[0].user, "yj");

        // Rename the ssh host, drop the container, add a fresh entry whose
        // user needs the empty→None and trim treatment.
        panel.edit_remote_hosts[0].host.name = "builder".to_string();
        panel.edit_remote_hosts.remove(1);
        let mut added = RemoteHostDraft::template();
        added.host.host = "staging".to_string();
        added.user = " deploy ".to_string();
        panel.edit_remote_hosts.push(added);
        let mut blank = RemoteHostDraft::template();
        blank.host.host = "docker-box".to_string();
        blank.host.docker = true;
        blank.user = "   ".to_string();
        panel.edit_remote_hosts.push(blank);

        let mut applied = source;
        panel.apply_to_config(&mut applied);

        assert_eq!(applied.remote_hosts.len(), 3);
        assert_eq!(applied.remote_hosts[0].display_name(), "builder");
        // A field the tab does not surface must ride through unmodified.
        assert_eq!(applied.remote_hosts[0].ssh_args, ["-p", "22"]);
        assert_eq!(applied.remote_hosts[1].host, "staging");
        assert_eq!(applied.remote_hosts[1].user.as_deref(), Some("deploy"));
        assert_eq!(applied.remote_hosts[1].deploy, "persist");
        assert_eq!(applied.remote_hosts[2].user, None);
        assert!(applied.remote_hosts[2].docker);
        assert!(applied
            .remote_hosts
            .iter()
            .all(|host| crate::config::validate_remote_host(host).is_ok()));

        // The same serialization Config::save() performs must bring the
        // entries back intact.
        let serialized = toml::to_string_pretty(&applied).expect("serialize");
        let reparsed: Config = toml::from_str(&serialized).expect("reparse");
        assert_eq!(reparsed.remote_hosts, applied.remote_hosts);
    }

    #[test]
    fn off_view_remote_drafts_remain_byte_exact_when_other_settings_are_applied() {
        let remote_hosts = (0..=crate::config::MAX_REMOTE_HOST_UI_ROWS)
            .map(|index| {
                let mut host = crate::config::default_remote_hosts()[0].clone();
                host.name = format!("host-{index}");
                host.host = format!("host-{index}.example.test");
                host
            })
            .collect();
        let mut source = Config {
            remote_hosts,
            ..Config::default()
        };
        let hidden = source
            .remote_hosts
            .last_mut()
            .expect("one draft beyond the UI prefix");
        hidden.name = "  hidden name  ".to_string();
        hidden.host = "  hidden.example.test  ".to_string();
        hidden.user = Some("  hidden user  ".to_string());
        hidden.deploy = "  invalid deploy draft  ".to_string();
        let expected = hidden.clone();

        let mut panel = ConfigPanel::new();
        panel.sync_from_config(&source);
        panel.edit_font_size += 1.0;
        panel.apply_to_config(&mut source);

        assert_eq!(
            source.remote_hosts[crate::config::MAX_REMOTE_HOST_UI_ROWS],
            expected
        );
    }

    #[test]
    fn hostile_remote_drafts_fail_borrowed_preflight_before_apply_clone() {
        let mut draft = RemoteHostDraft::from_config(&crate::config::default_remote_hosts()[0]);
        draft.host.deploy = format!("secret-marker-{}", "x".repeat(256));
        let error = draft.validate_for_apply().unwrap_err();
        assert_eq!(error, "deploy exceeds the 256-byte limit");
        assert!(!error.contains("secret-marker"));

        draft.host.deploy = "persist".to_string();
        draft.user = format!("secret-user-{}", "x".repeat(4 * 1024));
        let error = draft.validate_for_apply().unwrap_err();
        assert_eq!(error, "user exceeds the 4096-byte limit");
        assert!(!error.contains("secret-user"));

        // Repairing the separate user buffer must not clone the stale owning
        // value before replacing it.
        draft.host.user = Some(format!("stale-secret-{}", "x".repeat(256 * 1024)));
        draft.user = "repaired-user".to_string();
        assert!(draft.validate_for_apply().is_ok());
        assert_eq!(draft.applied().user.as_deref(), Some("repaired-user"));
    }
}
