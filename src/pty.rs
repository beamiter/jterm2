use anyhow::{anyhow, Result};
use std::ffi::CString;
use std::os::unix::io::RawFd;

/// The descriptor-pinning capability now lives in
/// [`jterm_core::agent_task::pinned_dir`], which is where the agent-task
/// runtime that needs it moved. This crate keeps the name it always used.
///
/// The only change is the error type: core returns `String` so the module
/// carries no error-library dependency, and every consumer here already
/// stringified these messages into its own domain error.
pub(crate) use jterm_core::agent_task::pinned_dir::PinnedDirectory;

/// PTY 读取结果。流关闭与子进程退出必须分开判断：Linux 的 EIO/Hangup
/// 只说明当前没有 slave fd，子进程仍可能存活；只有 waitpid 才能确认退出。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadOutcome {
    Data(usize),
    WouldBlock,
    Eof,
    Hangup,
}

/// PTY 写入结果。区分"写入了 n 字节(可能少于请求,即 partial write)"与
/// "缓冲区已满(WouldBlock,需 poll 等待可写)"。EINTR 在内部重试,不外泄。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    Written(usize),
    WouldBlock,
}

#[cfg(unix)]
mod unix_pty {
    use super::*;
    use std::ffi::OsStr;
    use std::path::Path;
    use std::time::{Duration, Instant};

    // Keep local launches effectively immediate while allowing automounts and
    // network-backed working directories a reasonable window to resolve.
    // Most importantly, this bounds the synchronous fork-to-exec handshake.
    const CHILD_STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

    /// 子进程生命周期状态机。TerminationStarted 之后,分离的升级回收线程
    /// 独占该 pid,is_alive/waitpid 等路径绝不能再对它发信号或 wait。
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ChildLifecycle {
        Running,
        TerminationStarted,
        Reaped,
    }

    /// Retry an interrupted syscall while preserving every other error for
    /// the caller to classify. In particular, waitpid errors other than
    /// ECHILD must not be mistaken for a successfully reaped child.
    pub(super) fn retry_on_eintr<T>(
        mut operation: impl FnMut() -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        loop {
            match operation() {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => return result,
            }
        }
    }

    /// 异步信号安全地向 stderr 写一条静态消息(fork 后、execve 前只能用此类调用)。
    /// SAFETY: 仅调用 write(2),它在 POSIX 异步信号安全函数列表中。
    unsafe fn write_stderr(msg: &[u8]) {
        let _ = libc::write(
            libc::STDERR_FILENO,
            msg.as_ptr() as *const libc::c_void,
            msg.len(),
        );
    }

    /// Create a close-on-exec pipe used to report setup failures from the
    /// post-fork child. EOF means `execve` succeeded and closed the write end.
    /// The read end is non-blocking: poll(2) is the authority for waiting, so
    /// the read after readiness must never stall on a partial status record.
    fn startup_status_pipe() -> std::io::Result<[RawFd; 2]> {
        let mut pipe_fds = [-1; 2];

        #[cfg(any(target_os = "linux", target_os = "android"))]
        let result = unsafe { libc::pipe2(pipe_fds.as_mut_ptr(), libc::O_CLOEXEC) };

        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let result = unsafe {
            let result = libc::pipe(pipe_fds.as_mut_ptr());
            if result == 0 {
                for fd in pipe_fds {
                    let flags = libc::fcntl(fd, libc::F_GETFD, 0);
                    if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) < 0 {
                        libc::close(pipe_fds[0]);
                        libc::close(pipe_fds[1]);
                        return Err(std::io::Error::last_os_error());
                    }
                }
            }
            result
        };

        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let flags = unsafe { libc::fcntl(pipe_fds[0], libc::F_GETFL, 0) };
        if flags < 0
            || unsafe { libc::fcntl(pipe_fds[0], libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            let error = std::io::Error::last_os_error();
            unsafe {
                libc::close(pipe_fds[0]);
                libc::close(pipe_fds[1]);
            }
            return Err(error);
        }
        Ok(pipe_fds)
    }

    /// 读取调用线程的 errno,仅供 fork 后的子进程分支在失败的 libc 调用之后
    /// 立即使用。它只读线程局部变量、不分配内存,满足 fork→execve 之间
    /// 只允许异步信号安全调用的约束。
    fn current_errno() -> libc::c_int {
        std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(libc::EIO)
    }

    /// Only async-signal-safe syscalls are allowed between fork and exec.
    /// The fixed (stage, errno) record is far below PIPE_BUF; retry EINTR
    /// until the parent receives every status byte — a short write followed
    /// by EOF would otherwise be mistaken for a successful exec.
    unsafe fn report_startup_failure(fd: RawFd, code: u8, errno: libc::c_int) {
        let record = [code as libc::c_int, errno];
        let mut offset = 0usize;
        let len = std::mem::size_of_val(&record);
        let ptr = record.as_ptr().cast::<u8>();
        while offset < len {
            let written = libc::write(fd, ptr.add(offset).cast::<libc::c_void>(), len - offset);
            if written > 0 {
                offset += written as usize;
            } else if written < 0 && current_errno() == libc::EINTR {
                continue;
            } else {
                return;
            }
        }
    }

    /// Blocking waitpid used only on startup-failure paths, where the child
    /// is already SIGKILLed (or self-exited) and cannot outlive the wait.
    unsafe fn reap_child_blocking(child_pid: libc::pid_t) {
        let mut status = 0;
        loop {
            let result = libc::waitpid(child_pid, &mut status, 0);
            if result >= 0 || current_errno() != libc::EINTR {
                break;
            }
        }
    }

    unsafe fn kill_and_reap_child(child_pid: libc::pid_t) {
        // The child may or may not have completed setsid(), so target both its
        // prospective process group and the process itself.
        let _ = libc::kill(-child_pid, libc::SIGKILL);
        let _ = libc::kill(child_pid, libc::SIGKILL);
        reap_child_blocking(child_pid);
    }

    fn startup_timeout_ms(remaining: Duration) -> libc::c_int {
        // Round up: a sub-millisecond remainder must still yield a nonzero
        // poll timeout rather than an instant (busy-looping) zero.
        let rounded_ms =
            remaining.as_millis() + u128::from(!remaining.subsec_nanos().is_multiple_of(1_000_000));
        rounded_ms.clamp(1, libc::c_int::MAX as u128) as libc::c_int
    }

    unsafe fn abort_child_startup(
        startup_read: RawFd,
        child_pid: libc::pid_t,
        error: anyhow::Error,
    ) -> Result<()> {
        // Close first so a child still attempting to report an error cannot
        // keep the handshake alive while it is being torn down.
        libc::close(startup_read);
        kill_and_reap_child(child_pid);
        Err(error)
    }

    /// Wait for either a child setup failure record or CLOEXEC EOF from
    /// execve, bounded by `timeout` — a child stuck between fork and execve
    /// (for example an fchdir into a hung network mount) must not hang pane
    /// spawn forever.
    ///
    /// `startup_read` is owned by this function. Every return path closes it;
    /// error paths also kill (process group AND pid) and reap the child
    /// before returning, so the caller only has to dispose of `master`.
    unsafe fn wait_for_child_startup(
        startup_read: RawFd,
        child_pid: libc::pid_t,
        timeout: Duration,
    ) -> Result<()> {
        let mut record = [0 as libc::c_int; 2];
        let record_len = std::mem::size_of_val(&record);
        let mut received = 0usize;
        let started = Instant::now();

        loop {
            let elapsed = started.elapsed();
            if elapsed >= timeout {
                return abort_child_startup(
                    startup_read,
                    child_pid,
                    anyhow!(
                        "Timed out after {} ms during shell fork-to-exec startup \
                         (received {received}/{record_len} status bytes)",
                        timeout.as_millis()
                    ),
                );
            }

            let mut poll_fd = libc::pollfd {
                fd: startup_read,
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = libc::poll(
                &mut poll_fd,
                1,
                startup_timeout_ms(timeout.saturating_sub(elapsed)),
            );
            if ready < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return abort_child_startup(
                    startup_read,
                    child_pid,
                    anyhow!("Failed to poll shell startup status: {error}"),
                );
            }
            if ready == 0 {
                return abort_child_startup(
                    startup_read,
                    child_pid,
                    anyhow!(
                        "Timed out after {} ms during shell fork-to-exec startup \
                         (received {received}/{record_len} status bytes)",
                        timeout.as_millis()
                    ),
                );
            }

            let revents = poll_fd.revents;
            if revents & libc::POLLNVAL != 0 {
                return abort_child_startup(
                    startup_read,
                    child_pid,
                    anyhow!("Invalid shell startup status fd"),
                );
            }
            if revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) == 0 {
                return abort_child_startup(
                    startup_read,
                    child_pid,
                    anyhow!(
                        "Unexpected poll events 0x{revents:x} during shell fork-to-exec startup"
                    ),
                );
            }

            // POLLHUP may arrive together with POLLIN. Always read first so a
            // complete or partial failure record already in the pipe is not
            // mistaken for successful exec EOF.
            let n = libc::read(
                startup_read,
                record
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(received)
                    .cast::<libc::c_void>(),
                record_len - received,
            );
            if n > 0 {
                received += n as usize;
                if received < record_len {
                    continue;
                }

                // A failed child self-exits after reporting, so reaping — not
                // killing — is the cleanup here. Decode the (stage, errno)
                // record into the message the pane can show.
                libc::close(startup_read);
                reap_child_blocking(child_pid);
                let code = record[0] as u8;
                let errno = record[1];
                let error = std::io::Error::from_raw_os_error(errno);
                return Err(match code {
                    b'C' => {
                        anyhow!("Failed to enter saved working directory: {error} (errno {errno})")
                    }
                    b'E' => anyhow!("Failed to execute shell: {error} (errno {errno})"),
                    _ => anyhow!("Shell failed during startup: {error} (errno {errno})"),
                });
            }
            if n == 0 {
                if received != 0 {
                    return abort_child_startup(
                        startup_read,
                        child_pid,
                        anyhow!(
                            "Incomplete shell startup status during fork-to-exec startup \
                             ({received}/{record_len} bytes)"
                        ),
                    );
                }
                if revents & libc::POLLERR != 0 {
                    return abort_child_startup(
                        startup_read,
                        child_pid,
                        anyhow!("Shell startup status pipe failed"),
                    );
                }

                // EOF without an error record is the success signal: execve
                // closed the child's CLOEXEC write end.
                libc::close(startup_read);
                return Ok(());
            }

            let error = std::io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EAGAIN) if revents & libc::POLLERR == 0 => continue,
                _ => {
                    return abort_child_startup(
                        startup_read,
                        child_pid,
                        anyhow!("Failed to read shell startup status: {error}"),
                    );
                }
            }
        }
    }

    /// Shell candidates are looked up through `jterm_core::host`, which now
    /// imposes what ember's local copy used to: the execute bit, a regular
    /// file, an absolute result, and an injectable `PATH` (launchers like wofi
    /// strip it, so the process environment cannot be the only source).
    fn find_executable_in_path_with(exe_name: &str, path_var: Option<&OsStr>) -> Option<String> {
        jterm_core::host::find_executable_in(exe_name, path_var)
            .map(|path| path.to_string_lossy().into_owned())
    }

    /// A `jsh` on PATH need not be the interactive shell ember prefers: the
    /// name can be taken by an unrelated binary, or be a symlink pointing
    /// somewhere else entirely. Launching such a program makes the child reject
    /// `--session` and exit immediately (ssh, for one, exits 255), which used to
    /// close the only tab and take the whole window down with it. Accept the
    /// candidate only when it really resolves to a `jsh` program.
    ///
    /// The shell was called `rsh` until 0.3; that name was an alternatives
    /// symlink to the BSD remote shell on Debian-family systems, which is how
    /// this guard came about. The rename removed that particular collision, but
    /// resolving the candidate is still the only way to know what we exec.
    pub(super) fn is_interactive_jsh(path: &Path) -> bool {
        let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let Some(name) = resolved.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        // Version-suffixed builds (jsh-0.3.0) stay eligible; anything the name
        // resolves to under another basename does not.
        name == "jsh" || name.starts_with("jsh-") || name.starts_with("jsh.")
    }

    pub(super) fn choose_shell_with_path(
        configured_shell: Option<&str>,
        path_var: Option<&OsStr>,
        sh_fallback: &Path,
    ) -> Result<String> {
        // Priority 1: explicit config / env var (needed when PATH is stripped by launchers like wofi)
        if let Some(path) = configured_shell {
            // A bare name is a PATH lookup, never an implicit `./name`.
            // Otherwise opening a project directory containing a malicious
            // executable named "bash" could hijack `shell = "bash"`. That rule
            // now lives in `host::resolve_configured_program`, which keeps it
            // verbatim (`file_name() == token` selects the PATH branch) and adds
            // the absolutization the exec path needs.
            if let Some(resolved) = jterm_core::host::resolve_configured_program(path, path_var) {
                return Ok(resolved.to_string_lossy().into_owned());
            }
            eprintln!(
                "[PTY] Configured shell '{}' is not executable, falling back",
                path
            );
        }

        // Priority 2: jsh (preferred shell with advanced features)
        if let Some(jsh_path) = find_executable_in_path_with("jsh", path_var) {
            if is_interactive_jsh(Path::new(&jsh_path)) {
                return Ok(jsh_path);
            }
            eprintln!(
                "[PTY] Ignoring '{}': it resolves to another program, not the jsh shell",
                jsh_path
            );
        }

        // Priority 3: bash (fallback)
        if let Some(bash_path) = find_executable_in_path_with("bash", path_var) {
            return Ok(bash_path);
        }

        // Priority 4: sh (last resort). execve does not search PATH, so never
        // return a bare "sh" token here.
        if let Some(sh_path) = find_executable_in_path_with("sh", path_var) {
            return Ok(sh_path);
        }
        // The fallback is a path, not a name: `resolve_configured_program` takes
        // its directory branch and only checks that it is executable.
        if let Some(sh_path) = sh_fallback
            .to_str()
            .and_then(|token| jterm_core::host::resolve_configured_program(token, None))
        {
            return Ok(sh_path.to_string_lossy().into_owned());
        }

        Err(anyhow!(
            "No executable shell found (tried configured shell, jsh, bash, PATH sh, and {})",
            sh_fallback.display()
        ))
    }

    pub(super) fn choose_shell(configured_shell: Option<&str>) -> Result<String> {
        let path_var = std::env::var_os("PATH");
        choose_shell_with_path(configured_shell, path_var.as_deref(), Path::new("/bin/sh"))
    }

    /// Resolve the shell captured by the source Ember pane and return an
    /// explicit argv for a single user-approved validation command.
    ///
    /// Validation runs in a fresh process inside the task worktree.  Passing
    /// the command as one argv element (rather than interpolating it into a
    /// wrapper script) preserves its exact shell syntax and avoids a second
    /// quoting language. Command mode deliberately is not login mode: a login
    /// profile may change directory after the PTY has entered the validated
    /// worktree, causing the command to run against unrelated files. Supported
    /// shells also receive their no-rc flag; unknown shell families fail
    /// closed because their non-interactive startup contract is not known.
    pub(super) fn validation_command_argv(
        source_shell: Option<&str>,
        command: &str,
    ) -> Result<Vec<String>> {
        let source_shell = source_shell
            .filter(|shell| !shell.is_empty())
            .ok_or_else(|| anyhow!("Validation source shell identity is missing"))?;
        let shell = jterm_core::host::resolve_configured_program(source_shell, None)
            .ok_or_else(|| {
                anyhow!("Validation source shell is no longer executable: {source_shell}")
            })?
            .to_string_lossy()
            .into_owned();
        if is_interactive_jsh(Path::new(&shell)) {
            return Ok(vec![
                shell,
                "--norc".to_string(),
                "-c".to_string(),
                command.to_string(),
            ]);
        }
        let resolved = std::fs::canonicalize(&shell).unwrap_or_else(|_| Path::new(&shell).into());
        let family = resolved
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        let mut argv = vec![shell];
        match family.as_str() {
            "bash" => argv.extend(["--noprofile".to_string(), "--norc".to_string()]),
            "zsh" => argv.push("-f".to_string()),
            "fish" => argv.push("--no-config".to_string()),
            "sh" | "dash" | "ksh" | "ksh93" | "mksh" => {}
            _ => {
                return Err(anyhow!(
                    "Unsupported source shell for isolated validation: {}",
                    resolved.display()
                ));
            }
        }
        argv.extend(["-c".to_string(), command.to_string()]);
        Ok(argv)
    }

    pub(super) fn validation_command_environment() -> [(&'static str, &'static str); 3] {
        [
            ("BASH_ENV", "/dev/null"),
            ("ENV", "/dev/null"),
            ("ZDOTDIR", "/dev/null"),
        ]
    }

    /// ember's child-environment policy.
    ///
    /// `less_default = "FR"` is deliberate and predates the shared module: no
    /// `-X`, so `git`/`man` still get the alternate screen, and `F` quits on a
    /// page that fits. `color_defaults` stays off because ember has never
    /// shipped an `LS_COLORS`, and the locale repair stays on: the GPU renderer
    /// draws UTF-8 either way, but a `C`-locale child emits mojibake into it.
    pub(super) fn child_environment() -> jterm_core::child_env::ChildEnv<'static> {
        jterm_core::child_env::ChildEnv {
            // Spelled out rather than taken from `identity`, which reports the
            // *core* crate's version until `main` initializes it: the version a
            // child reads has to be this binary's whatever the startup order is.
            app_version: env!("CARGO_PKG_VERSION"),
            less_default: Some("FR"),
            color_defaults: false,
            normalize_locale: true,
            ..jterm_core::child_env::ChildEnv::from_identity()
        }
    }

    pub struct Pty {
        master: RawFd,
        child_pid: i32,
        exit_code_cached: Option<i32>,
        lifecycle: ChildLifecycle,
    }

    impl Pty {
        #[allow(dead_code)]
        pub fn new_with_cwd(
            cols: usize,
            rows: usize,
            cwd: Option<&str>,
            session_id: Option<&str>,
            configured_shell: Option<&str>,
            command_argv: Option<&[String]>,
        ) -> Result<Self> {
            Self::new_with_pinned_cwd(
                cols,
                rows,
                cwd,
                session_id,
                configured_shell,
                command_argv,
                None,
            )
        }

        pub(crate) fn new_with_pinned_cwd(
            cols: usize,
            rows: usize,
            cwd: Option<&str>,
            session_id: Option<&str>,
            configured_shell: Option<&str>,
            command_argv: Option<&[String]>,
            pinned_cwd: Option<PinnedDirectory>,
        ) -> Result<Self> {
            // SAFETY: 这个 unsafe 块包含多个 libc 系统调用用于 PTY 创建和进程 fork。
            // 所有的 libc 调用都检查了返回值并正确处理错误。
            // 文件描述符的生命周期被正确管理（成功时存储在 PtySession 中，失败时关闭）。
            // fork 后的子进程分支永不返回（通过 execve 或 exit），避免了未定义行为。
            unsafe {
                let win_size = libc::winsize {
                    ws_row: rows as u16,
                    ws_col: cols as u16,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };

                // Before opening file descriptors, complete every fallible
                // allocation, PATH lookup and CString conversion. This both
                // keeps the post-fork child async-signal-safe and avoids
                // leaking a PTY pair if preparation returns an error.
                // 原因:在多线程进程中 fork 后,子进程直到 execve 之间只能调用
                // 异步信号安全的函数。malloc/CString/format!/Vec/setenv/std::env/std::fs
                // 都不安全 —— 若 fork 时另一线程恰好持有 malloc 锁,子进程会永久死锁。
                // 因此这里预先构建 argv、envp、cwd 的 C 字符串,子进程分支只做 syscall。

                // 一次性辅助进程(例如 jsh 安装脚本)按原样 exec 给定 argv:
                // 不做登录 shell 包装,也不注入 --session。
                let command_cstrings: Option<(CString, Vec<CString>)> = match command_argv {
                    Some(argv) => {
                        let program = argv
                            .first()
                            .ok_or_else(|| anyhow!("Command argv must not be empty"))?;
                        // Resolve with `execvp` semantics *before* the fork: the
                        // child can only call `execve`, and a helper that is not
                        // installed has to fail here, where the caller can show
                        // it, instead of as a pane that exits 127 silently.
                        let program_path = jterm_core::host::resolve_executable(
                            program,
                            std::env::var_os("PATH").as_deref(),
                            cwd,
                        )
                        .map_err(|error| {
                            anyhow!("Command program '{program}' is not executable: {error}")
                        })?
                        .to_string_lossy()
                        .into_owned();
                        let mut args = Vec::with_capacity(argv.len());
                        for arg in argv {
                            args.push(
                                CString::new(arg.as_str())
                                    .map_err(|_| anyhow!("Invalid command argument"))?,
                            );
                        }
                        Some((
                            CString::new(program_path)
                                .map_err(|_| anyhow!("Invalid command path"))?,
                            args,
                        ))
                    }
                    None => None,
                };

                let (exec_cstr, argv_cstrings): (CString, Vec<CString>) = if let Some(command) =
                    command_cstrings
                {
                    command
                } else {
                    // Shell discovery is irrelevant for explicit command
                    // sessions. Keeping it entirely inside this branch
                    // means a broken interactive `shell =` setting cannot
                    // prevent a valid Agent/helper executable from starting.
                    let shell_path = choose_shell(configured_shell)?;
                    let shell_cstr = CString::new(shell_path.clone())
                        .map_err(|_| anyhow!("Invalid shell path: {}", shell_path))?;
                    let shell_name = Path::new(&shell_path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("sh")
                        .to_string();
                    let bash_path = if shell_name == "jsh" {
                        // `find_executable_in_path` is exec-bit-checked and
                        // returns an absolute path.
                        jterm_core::host::find_executable_in_path("bash")
                            .map(|path| path.to_string_lossy().into_owned())
                    } else {
                        None
                    };

                    if let Some(bash_path) = bash_path {
                        let exec_cmd =
                            jterm_core::process::build_jsh_exec_command(&shell_path, session_id);
                        (
                            CString::new(bash_path).map_err(|_| anyhow!("Invalid bash path"))?,
                            vec![
                                CString::new("bash").unwrap(),
                                CString::new("-ic").unwrap(),
                                CString::new(exec_cmd)
                                    .map_err(|_| anyhow!("Invalid wrapped shell command"))?,
                            ],
                        )
                    } else {
                        // argv[0] prefixed with '-' requests login-shell
                        // behavior for ordinary interactive sessions.
                        let mut argv = vec![CString::new(format!("-{shell_name}"))
                            .map_err(|_| anyhow!("Invalid shell name"))?];
                        if shell_name == "bash" {
                            argv.push(CString::new("-l").unwrap());
                        }
                        if shell_name == "jsh" {
                            if let Some(sid) = session_id.and_then(|value| CString::new(value).ok())
                            {
                                argv.push(CString::new("--session").unwrap());
                                argv.push(sid);
                            }
                        }
                        (shell_cstr, argv)
                    }
                };

                let mut argv_ptrs: Vec<*const libc::c_char> =
                    argv_cstrings.iter().map(|arg| arg.as_ptr()).collect();
                argv_ptrs.push(std::ptr::null());

                // 工作目录的 C 字符串(若指定)
                let cwd_cstr = match cwd {
                    Some(dir) => {
                        Some(CString::new(dir).map_err(|_| anyhow!("Invalid working directory"))?)
                    }
                    None => None,
                };
                let cwd_directory = match (cwd, pinned_cwd) {
                    (Some(_), directory @ Some(_)) => directory,
                    (None, Some(_)) => {
                        return Err(anyhow!(
                            "pinned working directory requires an explicit display path"
                        ));
                    }
                    (Some(_), None) | (None, None) => None,
                };

                // 构建子进程环境:继承父进程环境,覆盖终端相关变量(避免在子进程调用
                // 非异步信号安全的 setenv)。直接把构建好的 envp 传给 execve。
                // 策略与变量集合由 jterm_core::child_env 持有,四个终端共用;这里
                // 只声明 ember 的选择 —— LESS=FR、UTF-8 locale 修正、不接管 ls 颜色。
                let command_environment =
                    cwd_directory.is_some().then(validation_command_environment);
                let env_cstrings = jterm_core::child_env::envp(
                    &child_environment(),
                    command_environment.as_ref().map_or(&[], |extra| extra),
                )
                .map_err(|error| anyhow!("Invalid child environment: {error}"))?;
                let mut envp: Vec<*const libc::c_char> =
                    env_cstrings.iter().map(|c| c.as_ptr()).collect();
                envp.push(std::ptr::null());

                // Create the PTY only after all `?` exits above are behind us.
                let mut master = 0;
                let mut slave = 0;
                if libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &win_size,
                ) != 0
                {
                    return Err(anyhow!("Failed to open PTY"));
                }

                // 设置 master 非阻塞模式
                let flags = libc::fcntl(master, libc::F_GETFL, 0);
                if flags >= 0 {
                    let _ = libc::fcntl(master, libc::F_SETFL, flags | libc::O_NONBLOCK);
                }

                // 设置 FD_CLOEXEC，防止子进程继承
                let fd_flags = libc::fcntl(master, libc::F_GETFD, 0);
                if fd_flags >= 0 {
                    let _ = libc::fcntl(master, libc::F_SETFD, fd_flags | libc::FD_CLOEXEC);
                }

                // The child reports chdir/exec failures through this pipe.
                // Successful exec closes its CLOEXEC write end, so the parent
                // does not return Ok until shell startup has actually crossed
                // the exec boundary.
                let startup_pipe = match startup_status_pipe() {
                    Ok(pipe) => pipe,
                    Err(error) => {
                        libc::close(master);
                        libc::close(slave);
                        return Err(anyhow!("Failed to create shell startup pipe: {error}"));
                    }
                };
                // Capture process-global state before fork. The child branch
                // then needs only async-signal-safe syscalls.
                let inherited_instance_lock_fd =
                    crate::session_persistence::inherited_instance_lock_fd();
                let mut default_signal_action: libc::sigaction = std::mem::zeroed();
                default_signal_action.sa_sigaction = libc::SIG_DFL;
                if libc::sigemptyset(&mut default_signal_action.sa_mask) != 0 {
                    libc::close(master);
                    libc::close(slave);
                    libc::close(startup_pipe[0]);
                    libc::close(startup_pipe[1]);
                    return Err(anyhow!("Failed to prepare child signal state"));
                }
                #[cfg(target_os = "linux")]
                let expected_parent_pid = libc::getpid();

                // Fork 子进程
                let fork_result = libc::fork();

                if fork_result < 0 {
                    libc::close(master);
                    libc::close(slave);
                    libc::close(startup_pipe[0]);
                    libc::close(startup_pipe[1]);
                    return Err(anyhow!("Failed to fork"));
                }

                if fork_result == 0 {
                    // 子进程分支:从这里到 execve 只调用异步信号安全的 libc 函数。
                    // A flock is tied to the inherited open-file description,
                    // not merely this process. Close it before any operation
                    // that could stall, so a PTY child can never outlive the
                    // primary process while retaining the instance lock.
                    if inherited_instance_lock_fd >= 0 {
                        libc::close(inherited_instance_lock_fd);
                    }
                    libc::close(master);
                    libc::close(startup_pipe[0]);

                    // Do not inherit the GUI's graceful-shutdown handler. A
                    // pre-exec child never polls SHUTDOWN_REQUESTED, so that
                    // handler would neutralize the parent-death SIGTERM.
                    if libc::sigaction(libc::SIGTERM, &default_signal_action, std::ptr::null_mut())
                        != 0
                    {
                        let errno = current_errno();
                        report_startup_failure(startup_pipe[1], b'S', errno);
                        libc::_exit(127);
                    }
                    libc::sigaction(libc::SIGINT, &default_signal_action, std::ptr::null_mut());

                    // 【关键】设置父进程死亡信号：当父进程(ember)死亡时，此进程会收到SIGTERM
                    // 这是最后一道防线，确保即使ember被SIGKILL强制杀死或panic崩溃，
                    // jsh进程也会收到退出信号，不会变成孤儿进程继续运行。
                    #[cfg(target_os = "linux")]
                    {
                        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                            let errno = current_errno();
                            report_startup_failure(startup_pipe[1], b'S', errno);
                            libc::_exit(127);
                        }
                        // The parent can die between fork and prctl. Detect
                        // that race explicitly instead of waiting forever for
                        // a signal the kernel could not retroactively deliver.
                        if libc::getppid() != expected_parent_pid {
                            libc::_exit(128 + libc::SIGTERM);
                        }
                    }

                    // 创建新的会话和进程组（将此进程设为会话leader）
                    libc::setsid();

                    // 切换工作目录(使用 fork 前构建好的指针)
                    if let Some(ref directory) = cwd_directory {
                        if libc::fchdir(directory.as_raw_fd()) != 0 {
                            let errno = current_errno();
                            report_startup_failure(startup_pipe[1], b'C', errno);
                            libc::_exit(127);
                        }
                        libc::close(directory.as_raw_fd());
                    } else if let Some(ref dir_cstr) = cwd_cstr {
                        if libc::chdir(dir_cstr.as_ptr()) != 0 {
                            let errno = current_errno();
                            report_startup_failure(startup_pipe[1], b'C', errno);
                            libc::_exit(127);
                        }
                    }

                    // 设置 slave 为控制终端
                    if libc::ioctl(slave, libc::TIOCSCTTY, 0) != 0 {
                        write_stderr(b"ember: ioctl TIOCSCTTY failed\n");
                    }

                    // 重定向 stdin/stdout/stderr 到 PTY slave
                    libc::dup2(slave, libc::STDIN_FILENO);
                    libc::dup2(slave, libc::STDOUT_FILENO);
                    libc::dup2(slave, libc::STDERR_FILENO);
                    if slave > libc::STDERR_FILENO {
                        libc::close(slave);
                    }

                    // 执行 shell，使用 fork 前构建好的 argv/envp
                    libc::execve(exec_cstr.as_ptr(), argv_ptrs.as_ptr(), envp.as_ptr());

                    // 如果 execve 返回，说明出错
                    let errno = current_errno();
                    report_startup_failure(startup_pipe[1], b'E', errno);
                    libc::_exit(127);
                } else {
                    // 父进程分支
                    // 关闭 slave
                    libc::close(slave);
                    libc::close(startup_pipe[1]);

                    // 有界等待 fork→execve 握手:子进程卡在 fork 与 execve 之间
                    // (例如 fchdir 进入挂起的网络挂载)时,超时路径会杀死(进程组
                    // 与进程本身)并回收子进程,pane 创建不再被永久挂起。
                    if let Err(error) =
                        wait_for_child_startup(startup_pipe[0], fork_result, CHILD_STARTUP_TIMEOUT)
                    {
                        libc::close(master);
                        return Err(error);
                    }

                    Ok(Pty {
                        master,
                        child_pid: fork_result as i32,
                        exit_code_cached: None,
                        lifecycle: ChildLifecycle::Running,
                    })
                }
            }
        }

        pub fn get_child_pid(&self) -> i32 {
            self.child_pid
        }

        pub fn master_fd(&self) -> RawFd {
            self.master
        }

        pub fn wait_fd_readable(fd: RawFd, timeout_ms: i32) -> Result<bool> {
            let mut poll_fd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };

            // SAFETY: poll_fd 是有效的栈上变量，libc::poll 接受可变指针和长度，
            // 超时参数是合法的毫秒值。poll 调用是原子的，不会导致数据竞争。
            loop {
                let ready = unsafe { libc::poll(&mut poll_fd, 1, timeout_ms) };
                if ready < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::Interrupted {
                        continue; // EINTR:被信号打断,重试
                    }
                    return Err(anyhow!("Failed to poll PTY: {}", err));
                } else if ready == 0 {
                    return Ok(false);
                } else {
                    return Ok(
                        (poll_fd.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR)) != 0,
                    );
                }
            }
        }

        /// 单次非阻塞写入。返回写入的字节数(可能 partial),或 WouldBlock 表示缓冲区满。
        pub fn write(&mut self, data: &[u8]) -> Result<WriteOutcome> {
            // SAFETY: self.master 是有效的文件描述符，data.as_ptr() 指向有效的内存，
            // data.len() 是正确的长度。write 系统调用不会超出缓冲区边界。
            loop {
                let n = unsafe { libc::write(self.master, data.as_ptr() as *const _, data.len()) };
                if n >= 0 {
                    return Ok(WriteOutcome::Written(n as usize));
                }
                let err = std::io::Error::last_os_error();
                match err.kind() {
                    std::io::ErrorKind::Interrupted => continue, // EINTR:重试
                    std::io::ErrorKind::WouldBlock => return Ok(WriteOutcome::WouldBlock),
                    _ => return Err(anyhow!("Failed to write to PTY: {}", err)),
                }
            }
        }

        pub fn read(&mut self, buf: &mut [u8]) -> Result<ReadOutcome> {
            // SAFETY: self.master 是有效的文件描述符，buf.as_mut_ptr() 指向有效的可变内存，
            // buf.len() 是正确的缓冲区大小。read 不会超出边界。
            loop {
                let n = unsafe { libc::read(self.master, buf.as_mut_ptr() as *mut _, buf.len()) };
                if n > 0 {
                    return Ok(ReadOutcome::Data(n as usize));
                } else if n == 0 {
                    // read 返回 0 表示对端(slave)已关闭 —— EOF。
                    return Ok(ReadOutcome::Eof);
                } else {
                    let err = std::io::Error::last_os_error();
                    // Linux PTY masters report EIO (rather than read(2) == 0)
                    // once the final slave fd closes. Keep it distinct from
                    // process exit: the reader must stop polling this fd and
                    // waitpid at a bounded cadence, not fabricate an exit code.
                    if err.raw_os_error() == Some(libc::EIO) {
                        return Ok(ReadOutcome::Hangup);
                    }
                    match err.kind() {
                        std::io::ErrorKind::Interrupted => continue, // EINTR:重试
                        std::io::ErrorKind::WouldBlock => return Ok(ReadOutcome::WouldBlock),
                        _ => return Err(anyhow!("Failed to read from PTY: {}", err)),
                    }
                }
            }
        }

        pub fn resize(&mut self, cols: usize, rows: usize) -> Result<()> {
            // SAFETY: win_size 是有效的栈上变量，符合 libc::winsize 的内存布局。
            // ioctl TIOCSWINSZ 调用是标准的 PTY 窗口大小设置操作。
            unsafe {
                let win_size = libc::winsize {
                    ws_row: rows as u16,
                    ws_col: cols as u16,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };

                if libc::ioctl(
                    self.master,
                    libc::TIOCSWINSZ,
                    (&win_size) as *const _ as *mut libc::c_void,
                ) < 0
                {
                    return Err(anyhow!("Failed to resize PTY"));
                }
            }
            Ok(())
        }

        /// 把 waitpid 返回的 status 解码为退出码并缓存。
        /// 关键不变量:任何 reap(回收僵尸)的路径都必须缓存退出码,
        /// 这样 `exit_code_cached.is_some()` 等价于"PID 已被回收、可能被复用";
        /// kill 路径据此判断是否还能安全发信号,避免误杀复用了该 PID 的无辜进程。
        fn cache_status(&mut self, status: i32) {
            let code = if libc::WIFEXITED(status) {
                libc::WEXITSTATUS(status) as i32
            } else if libc::WIFSIGNALED(status) {
                -(libc::WTERMSIG(status) as i32)
            } else {
                -1
            };
            self.exit_code_cached = Some(code);
            self.lifecycle = ChildLifecycle::Reaped;
        }

        /// Observe the direct child without consuming its wait status. Keeping
        /// the exited leader as a zombie preserves ownership of its numeric
        /// PID/process-group identity until `finish_observed_exit` has killed
        /// every same-group descendant. Reaping first would make a later
        /// `kill(-pid, ...)` vulnerable to PID/PGID reuse.
        fn child_exit_observed(&self, nonblocking: bool) -> std::io::Result<bool> {
            if self.exit_code_cached.is_some() {
                return Ok(true);
            }
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let options =
                libc::WEXITED | libc::WNOWAIT | if nonblocking { libc::WNOHANG } else { 0 };
            retry_on_eintr(|| {
                // `waitid` may leave siginfo unspecified after EINTR. Reset it
                // on every attempt so WNOHANG's no-event result remains zero.
                info = unsafe { std::mem::zeroed() };
                // SAFETY: `info` is writable, P_PID scopes the observation to
                // our forked child, and WNOWAIT explicitly preserves its wait
                // status for the cleanup/reap step below.
                let result = unsafe {
                    libc::waitid(
                        libc::P_PID,
                        self.child_pid as libc::id_t,
                        &mut info,
                        options,
                    )
                };
                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            })?;
            // SAFETY: a successful waitid initializes siginfo. With WNOHANG,
            // si_pid == 0 means the selected child has not changed state.
            Ok(unsafe { info.si_pid() } == self.child_pid)
        }

        /// The direct child is known to be exited but deliberately unreaped.
        /// Kill its still-owned private process group first, then consume the
        /// original leader status exactly once. SIGKILL cannot change a status
        /// that is already waitable, so a successful CLI exit remains exit 0.
        fn finish_observed_exit(&mut self) -> std::io::Result<i32> {
            if let Some(code) = self.exit_code_cached {
                return Ok(code);
            }
            // SAFETY: WNOWAIT kept child_pid unreaped, so its process-group ID
            // cannot be recycled between this signal and the waitpid below.
            let killed = unsafe { libc::kill(-self.child_pid, libc::SIGKILL) };
            if killed < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    // Preserve the zombie ownership anchor and retry rather
                    // than publishing completion while a group we failed to
                    // terminate may still be active.
                    return Err(error);
                }
            }

            let mut status = 0;
            let child_pid = self.child_pid;
            let result = retry_on_eintr(|| {
                // SAFETY: the observed child is waitable and still owned by
                // this Pty. No other Ember path can wait while its mutex is held.
                let result = unsafe { libc::waitpid(child_pid, &mut status, 0) };
                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result)
                }
            });
            match result {
                Ok(result) if result == child_pid => {
                    self.cache_status(status);
                    Ok(self.exit_code_cached.unwrap_or(-1))
                }
                Ok(_) => Err(std::io::Error::other(
                    "waitpid returned no status for an observed PTY exit",
                )),
                Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                    self.exit_code_cached = Some(-1);
                    self.lifecycle = ChildLifecycle::Reaped;
                    Ok(-1)
                }
                Err(error) => Err(error),
            }
        }

        pub fn is_alive(&mut self) -> bool {
            // A detached reaper owns TerminationStarted children. Never issue
            // another waitid/waitpid (or later signal) for a pid once teardown
            // begins. A cached exit code likewise means the PID was reaped.
            if self.lifecycle != ChildLifecycle::Running || self.exit_code_cached.is_some() {
                return false;
            }

            match self.child_exit_observed(true) {
                Ok(false) => true,
                Ok(true) => match self.finish_observed_exit() {
                    Ok(_) => false,
                    Err(error) => {
                        log::warn!("could not finish observed PTY child exit: {error}");
                        true
                    }
                },
                Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                    // Another wait path reaped the PID without leaving a
                    // status. Unknown must fail closed, never masquerade as a
                    // successful child process.
                    self.exit_code_cached = Some(-1);
                    self.lifecycle = ChildLifecycle::Reaped;
                    false
                }
                Err(error) => {
                    // Conservatively keep the process live: an unrelated
                    // waitid failure is not evidence that the PID was reaped.
                    log::warn!("waitid(WNOWAIT) failed while checking PTY child: {error}");
                    true
                }
            }
        }

        /// 优雅终止:仅在子进程尚未被回收时发送 SIGHUP(进程组)+SIGTERM。
        /// 调用者必须持有 Pty 的互斥锁,以与 io_loop 的回收路径串行化。
        pub fn signal_terminate(&mut self) {
            if self.exit_code_cached.is_some() {
                return; // 已回收,PID 可能被复用,绝不再发信号
            }
            // 此时子进程尚未被 reap(僵尸或运行中),其 PID 仍为我们保留,kill 安全。
            // SAFETY: 负 PID 向进程组发信号是标准做法;child_pid 来自 fork。
            unsafe {
                let pgid = -self.child_pid;
                let _ = libc::kill(pgid, libc::SIGHUP);
                let _ = libc::kill(self.child_pid, libc::SIGTERM);
            }
        }

        /// 强制终止并回收:仅在尚未被回收时 SIGKILL 进程组,然后阻塞 waitpid 恰好一次。
        /// 调用者必须持有 Pty 的互斥锁。
        pub fn force_kill_and_reap(&mut self) {
            if self.exit_code_cached.is_some() {
                return; // 已被 io_loop 回收,跳过(避免误杀复用 PID)
            }
            // SAFETY: 同上,子进程尚未 reap,PID 仍保留,kill/waitpid 安全。
            unsafe {
                let pgid = -self.child_pid;
                let _ = libc::kill(pgid, libc::SIGKILL);
                let _ = libc::kill(self.child_pid, libc::SIGKILL);
                let mut status = 0;
                loop {
                    let r = libc::waitpid(self.child_pid, &mut status, 0);
                    if r > 0 {
                        self.cache_status(status);
                        break;
                    } else if r < 0
                        && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                    {
                        continue; // EINTR:重试
                    } else {
                        // ECHILD 等:无法回收(已被回收),记一个强杀退出码。
                        self.exit_code_cached = Some(-9);
                        self.lifecycle = ChildLifecycle::Reaped;
                        break;
                    }
                }
            }
        }

        /// 单次非阻塞退出观察。已退出时先清理仍归属该 PTY 进程组的后代，
        /// 再 waitpid 回收 leader；ECHILD → unknown，仍在运行 → None。
        /// 调用方应在锁外做有界轮询,把 sleep 排除在 Pty 临界区之外,避免 UI
        /// 线程的 resize/write 在子进程退出窗口内被阻塞数十毫秒。
        pub fn try_reap(&mut self) -> Option<i32> {
            if let Some(code) = self.exit_code_cached {
                return Some(code);
            }
            match self.child_exit_observed(true) {
                Ok(false) => None,
                Ok(true) => match self.finish_observed_exit() {
                    Ok(code) => Some(code),
                    Err(error) => {
                        log::warn!("could not finish observed PTY child exit: {error}");
                        None
                    }
                },
                Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                    self.exit_code_cached = Some(-1);
                    self.lifecycle = ChildLifecycle::Reaped;
                    Some(-1)
                }
                Err(error) => {
                    log::warn!("waitid(WNOWAIT) failed while observing PTY child: {error}");
                    None
                }
            }
        }

        pub fn wait_timeout(&mut self, _timeout_ms: u64) -> Result<i32> {
            // If we already have a cached exit code, return it directly
            if let Some(code) = self.exit_code_cached {
                return Ok(code);
            }

            match self.child_exit_observed(false) {
                Ok(true) => self
                    .finish_observed_exit()
                    .map_err(|error| anyhow!("waitpid failed: {error}")),
                Ok(false) => Err(anyhow!("waitid returned without an exited PTY child")),
                Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                    crate::debug_log!("[PTY] waitid returned ECHILD, process already reaped");
                    self.exit_code_cached = Some(-1);
                    self.lifecycle = ChildLifecycle::Reaped;
                    Ok(-1)
                }
                Err(error) => Err(anyhow!("waitid failed: {error}")),
            }
        }

        pub fn terminate(&mut self) -> Result<()> {
            // 生命周期检查必须先于任何信号:分离的回收线程启动后,再次调用
            // terminate 绝不能再向可能已被 OS 复用的 pid/pgid 发信号。
            if self.lifecycle != ChildLifecycle::Running {
                return Ok(());
            }
            if !self.is_alive() {
                return Ok(());
            }

            // 发信号前先认领拆除:随后的 terminate()(包括 Drop 触发的)都是
            // no-op。此时子进程尚未被 reap,其 PID/PGID 仍为我们保留,kill 安全。
            self.lifecycle = ChildLifecycle::TerminationStarted;
            self.exit_code_cached = Some(-libc::SIGTERM);

            // SAFETY: 负 PID 向进程组发信号;child_pid 经 setsid 成为会话/进程组
            // leader。上面的 is_alive 检查确保子进程尚未被回收。
            unsafe {
                let pgid = -self.child_pid;
                let _ = libc::kill(pgid, libc::SIGHUP);
                let _ = libc::kill(self.child_pid, libc::SIGTERM);
            }

            // 升级路径(等待→SIGKILL→回收)放到分离线程,而不是在这里 sleep:
            // terminate() 会从 Drop 调用,常常运行在 UI 线程上,绝不能为宽限期
            // 阻塞。线程只捕获 pid(独占其进程组),与 self 无别名。
            let child_pid = self.child_pid;
            std::thread::spawn(move || unsafe {
                std::thread::sleep(std::time::Duration::from_millis(50));
                let mut status = 0;
                let observed = loop {
                    let result = libc::waitpid(child_pid, &mut status, libc::WNOHANG);
                    if result >= 0
                        || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR)
                    {
                        break result;
                    }
                };
                if observed == 0 {
                    // 宽限期后仍存活:强杀进程组与进程本身,然后回收。
                    let _ = libc::kill(-child_pid, libc::SIGKILL);
                    let _ = libc::kill(child_pid, libc::SIGKILL);
                    loop {
                        if libc::waitpid(child_pid, &mut status, 0) >= 0
                            || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR)
                        {
                            break;
                        }
                    }
                }
            });
            Ok(())
        }
    }

    impl Drop for Pty {
        fn drop(&mut self) {
            if self.is_alive() {
                let _ = self.terminate();
            }
            // SAFETY: close 关闭文件描述符。master 是有效的 fd，
            // 关闭后不会再使用（因为这是 Drop 实现）。
            unsafe {
                let _ = libc::close(self.master);
            }
        }
    }

    #[cfg(test)]
    mod lifecycle_tests {
        use super::*;

        unsafe fn make_test_startup_child(
            partial_status: Option<u8>,
            keep_writer_open: bool,
        ) -> (RawFd, libc::pid_t) {
            let [read_fd, write_fd] = startup_status_pipe().expect("create test startup pipe");
            let pid = libc::fork();
            if pid < 0 {
                libc::close(read_fd);
                libc::close(write_fd);
                panic!(
                    "fork test startup child: {}",
                    std::io::Error::last_os_error()
                );
            }
            if pid == 0 {
                libc::close(read_fd);
                // The production child immediately execs, which closes every
                // unrelated CLOEXEC descriptor. This fixture can deliberately
                // pause before exit, so emulate that boundary explicitly: if
                // it retained a process-wide flock opened by another parallel
                // test, dropping the lock in the parent would not release it
                // until this child was killed. Keep the status writer at one
                // known descriptor and close everything above it.
                const STATUS_FD: RawFd = 3;
                if write_fd != STATUS_FD {
                    if libc::dup2(write_fd, STATUS_FD) < 0 {
                        libc::_exit(126);
                    }
                    libc::close(write_fd);
                }
                #[cfg(target_os = "linux")]
                let close_range_unavailable =
                    libc::close_range(STATUS_FD as libc::c_uint + 1, libc::c_uint::MAX, 0) != 0;
                #[cfg(not(target_os = "linux"))]
                let close_range_unavailable = true;
                if close_range_unavailable {
                    // This test module only exercises Unix targets. The
                    // conservative old-kernel fallback avoids allocation
                    // after fork. Test processes keep descriptors in this
                    // low range; production takes the exec/CLOEXEC path.
                    for fd in (STATUS_FD + 1)..1024 {
                        libc::close(fd);
                    }
                }
                if let Some(byte) = partial_status {
                    let _ = libc::write(STATUS_FD, (&byte as *const u8).cast::<libc::c_void>(), 1);
                }
                if keep_writer_open {
                    loop {
                        libc::pause();
                    }
                }
                libc::close(STATUS_FD);
                libc::_exit(0);
            }

            libc::close(write_fd);
            (read_fd, pid)
        }

        #[test]
        fn startup_handshake_times_out_after_partial_record_and_reaps_child() {
            let (read_fd, pid) = unsafe { make_test_startup_child(Some(1), true) };
            let started = Instant::now();
            let error = unsafe {
                wait_for_child_startup(read_fd, pid, Duration::from_millis(50))
                    .expect_err("a child holding a partial record must time out")
            };

            assert!(
                error.to_string().contains("fork-to-exec startup"),
                "unexpected error: {error:#}"
            );
            assert!(
                error.to_string().contains("received 1/"),
                "partial byte count missing from error: {error:#}"
            );
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "startup timeout exceeded its bounded cleanup window"
            );

            // The timeout path must have killed (group and pid) and reaped
            // the stuck child, so a second waitpid reports ECHILD.
            let mut status = 0;
            let wait_result = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            assert_eq!(wait_result, -1, "startup child was not already reaped");
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ECHILD)
            );
        }

        #[test]
        fn startup_handshake_rejects_eof_after_partial_record() {
            let (read_fd, pid) = unsafe { make_test_startup_child(Some(1), false) };
            let error = unsafe {
                wait_for_child_startup(read_fd, pid, Duration::from_secs(1))
                    .expect_err("EOF after a partial record must be rejected")
            };
            assert!(
                error
                    .to_string()
                    .contains("Incomplete shell startup status"),
                "unexpected error: {error:#}"
            );
        }

        #[test]
        fn terminate_is_idempotent_and_does_not_block_the_caller() {
            let mut pty = Pty::new_with_cwd(80, 24, Some("/"), None, Some("/bin/sh"), None)
                .expect("start /bin/sh in a PTY");
            assert_eq!(pty.lifecycle, ChildLifecycle::Running);

            let started = Instant::now();
            pty.terminate().expect("first terminate succeeds");
            // 旧的阻塞实现会在调用方线程 sleep 50ms 再回收。
            assert!(
                started.elapsed() < Duration::from_millis(50),
                "terminate must return before the grace-period escalation"
            );
            assert_eq!(pty.lifecycle, ChildLifecycle::TerminationStarted);

            pty.terminate().expect("repeated terminate is a no-op");
            assert_eq!(pty.lifecycle, ChildLifecycle::TerminationStarted);
        }
    }
}

#[cfg(windows)]
mod windows_pty {
    use super::*;

    pub struct Pty;

    impl Pty {
        pub fn new(_cols: usize, _rows: usize) -> Result<Self> {
            Err(anyhow!("PTY support not yet implemented on Windows"))
        }

        pub fn write(&mut self, _data: &[u8]) -> Result<WriteOutcome> {
            Err(anyhow!("PTY not available"))
        }

        pub fn read(&mut self, _buf: &mut [u8]) -> Result<ReadOutcome> {
            Err(anyhow!("PTY not available"))
        }

        pub fn resize(&mut self, _cols: usize, _rows: usize) -> Result<()> {
            Err(anyhow!("PTY not available"))
        }

        pub fn is_alive(&mut self) -> bool {
            false
        }

        pub fn try_reap(&mut self) -> Option<i32> {
            None
        }

        pub fn wait_timeout(&mut self, _timeout_ms: u64) -> Result<i32> {
            Err(anyhow!("PTY not available"))
        }

        pub fn signal_terminate(&mut self) {}

        pub fn force_kill_and_reap(&mut self) {}

        pub fn terminate(&mut self) -> Result<()> {
            Err(anyhow!("PTY not available"))
        }
    }
}

#[cfg(unix)]
pub use unix_pty::Pty;

/// Build the explicit shell argv used by task validation terminals.
#[cfg(unix)]
pub(crate) fn validation_command_argv(
    configured_shell: Option<&str>,
    command: &str,
) -> Result<Vec<String>> {
    unix_pty::validation_command_argv(configured_shell, command)
}

#[cfg(unix)]
pub(crate) fn resolved_shell(configured_shell: Option<&str>) -> Result<String> {
    unix_pty::choose_shell(configured_shell)
}

#[cfg(windows)]
pub use windows_pty::Pty;

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::sync::atomic::{AtomicU64, Ordering};

    #[cfg(unix)]
    static NEXT_SHELL_TEST: AtomicU64 = AtomicU64::new(0);

    #[cfg(unix)]
    struct ShellTestDir(std::path::PathBuf);

    #[cfg(unix)]
    impl ShellTestDir {
        fn new(label: &str) -> Self {
            let id = NEXT_SHELL_TEST.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ember-shell-test-{label}-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn executable(&self, name: &str) -> std::path::PathBuf {
            use std::os::unix::fs::PermissionsExt;

            let path = self.0.join(name);
            std::fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            path
        }

        fn search_path(&self) -> std::ffi::OsString {
            std::env::join_paths([&self.0]).unwrap()
        }
    }

    #[cfg(unix)]
    impl Drop for ShellTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn retry_on_eintr_retries_only_interrupted_errors() {
        let mut attempts = 0;
        let result = super::unix_pty::retry_on_eintr(|| {
            attempts += 1;
            if attempts < 3 {
                Err(std::io::Error::from_raw_os_error(libc::EINTR))
            } else {
                Ok(42)
            }
        });
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 3);

        let mut attempts = 0;
        let error = super::unix_pty::retry_on_eintr::<()>(|| {
            attempts += 1;
            Err(std::io::Error::from_raw_os_error(libc::ECHILD))
        })
        .unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
        assert_eq!(attempts, 1);
    }

    #[cfg(unix)]
    #[test]
    fn pinned_descendant_rejects_symlinked_ancestors_and_parent_components() {
        use std::os::unix::fs::symlink;

        let root = ShellTestDir::new("pinned-beneath");
        let outside = ShellTestDir::new("pinned-outside");
        std::fs::create_dir_all(root.0.join("safe/nested")).unwrap();
        std::fs::create_dir_all(outside.0.join("nested")).unwrap();
        symlink(&outside.0, root.0.join("safe/redirect")).unwrap();

        let pinned = super::PinnedDirectory::open(&root.0).unwrap();
        let cloned_root = pinned.open_beneath(std::path::Path::new(".")).unwrap();
        assert_eq!(
            std::fs::canonicalize(cloned_root.proc_path()).unwrap(),
            std::fs::canonicalize(&root.0).unwrap()
        );
        let nested = pinned.open_beneath(std::path::Path::new("safe/nested"));
        assert!(nested.is_ok(), "ordinary descendants remain available");
        assert!(pinned
            .open_beneath(std::path::Path::new("safe/redirect/nested"))
            .is_err());
        assert!(pinned
            .open_beneath(std::path::Path::new("../pinned-outside"))
            .is_err());
        assert!(pinned.open_beneath(&outside.0).is_err());

        let moved = root.0.with_extension("moved");
        std::fs::rename(&root.0, &moved).unwrap();
        assert_eq!(
            std::fs::canonicalize(pinned.proc_path()).unwrap(),
            std::fs::canonicalize(&moved).unwrap(),
            "the descriptor remains anchored to the original inode after rename"
        );
        std::fs::rename(&moved, &root.0).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn configured_bare_shell_name_is_resolved_through_path() {
        let root = ShellTestDir::new("configured");
        let executable = root.executable("custom-shell");
        let search_path = root.search_path();

        let selected = super::unix_pty::choose_shell_with_path(
            Some("custom-shell"),
            Some(&search_path),
            &root.0.join("missing-sh"),
        )
        .unwrap();

        assert_eq!(std::path::Path::new(&selected), executable);
        assert!(std::path::Path::new(&selected).is_absolute());
    }

    #[cfg(unix)]
    #[test]
    fn validation_argv_keeps_command_as_one_argument() {
        let root = ShellTestDir::new("validation-argv");
        let shell = root.executable("bash");
        let command = "printf '%s' \"$HOME && literal\"";

        let argv = super::unix_pty::validation_command_argv(shell.to_str(), command).unwrap();

        assert_eq!(
            argv,
            vec![
                shell.to_string_lossy(),
                "--noprofile".into(),
                "--norc".into(),
                "-c".into(),
                command.into()
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn validation_argv_does_not_load_a_login_profile_that_changes_cwd() {
        let root = ShellTestDir::new("validation-login-cwd");
        let home = root.0.join("home");
        let expected_cwd = root.0.join("worktree");
        std::fs::create_dir(&home).unwrap();
        std::fs::create_dir(&expected_cwd).unwrap();
        std::fs::write(home.join(".bash_profile"), b"cd /\n").unwrap();
        let argv = super::unix_pty::validation_command_argv(Some("/bin/bash"), "pwd").unwrap();

        let output = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&expected_cwd)
            .env("HOME", &home)
            .env_remove("BASH_ENV")
            .output()
            .unwrap();

        assert!(output.status.success());
        assert_eq!(
            std::path::Path::new(String::from_utf8(output.stdout).unwrap().trim()),
            expected_cwd
        );
    }

    #[cfg(unix)]
    #[test]
    fn explicit_command_scrubs_noninteractive_shell_startup_hooks() {
        let root = ShellTestDir::new("validation-startup-hook-cwd");
        let expected_cwd = root.0.join("worktree");
        let outside = root.0.join("outside");
        let bash_env = root.0.join("bash-env");
        std::fs::create_dir(&expected_cwd).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(&bash_env, format!("cd {}\n", outside.display())).unwrap();
        let escaped = std::process::Command::new("/bin/bash")
            .args(["-c", "pwd"])
            .current_dir(&expected_cwd)
            .env("BASH_ENV", &bash_env)
            .output()
            .unwrap();
        assert_eq!(
            std::path::Path::new(String::from_utf8(escaped.stdout).unwrap().trim()),
            outside
        );

        let validation_environment = super::unix_pty::validation_command_environment();
        assert!(validation_environment.contains(&("BASH_ENV", "/dev/null")));
        let mut safe = std::process::Command::new("/bin/bash");
        safe.args(["-c", "pwd"]).current_dir(&expected_cwd);
        for (name, value) in validation_environment {
            safe.env(name, value);
        }
        let safe = safe.output().unwrap();
        assert!(safe.status.success());
        assert_eq!(
            std::path::Path::new(String::from_utf8(safe.stdout).unwrap().trim()),
            expected_cwd
        );
    }

    #[cfg(unix)]
    #[test]
    fn pinned_validation_cwd_cannot_be_redirected_by_path_replacement() {
        let root = ShellTestDir::new("validation-pinned-cwd");
        let original_path = root.0.join("worktree");
        let moved_path = root.0.join("worktree-moved");
        std::fs::create_dir(&original_path).unwrap();
        let pinned = super::PinnedDirectory::open(&original_path).unwrap();
        std::fs::rename(&original_path, &moved_path).unwrap();
        std::fs::create_dir(&original_path).unwrap();
        let argv = vec!["/bin/pwd".to_string()];

        let mut pty = super::Pty::new_with_pinned_cwd(
            80,
            24,
            original_path.to_str(),
            None,
            None,
            Some(&argv),
            Some(pinned),
        )
        .unwrap();
        let mut seen = String::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut buf = [0u8; 1024];
        while std::time::Instant::now() < deadline
            && !seen.contains(moved_path.to_string_lossy().as_ref())
        {
            match pty.read(&mut buf) {
                Ok(super::ReadOutcome::Data(n)) => {
                    seen.push_str(&String::from_utf8_lossy(&buf[..n]));
                }
                Ok(super::ReadOutcome::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Ok(_) | Err(_) => break,
            }
        }

        let reported = seen.trim().trim_end_matches('\r');
        assert_eq!(
            std::path::Path::new(reported),
            moved_path,
            "child did not enter the pinned directory: {seen:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn validation_argv_uses_jsh_command_contract() {
        let root = ShellTestDir::new("validation-jsh");
        let shell = root.executable("jsh");

        let argv = super::unix_pty::validation_command_argv(shell.to_str(), "cargo test").unwrap();

        assert_eq!(argv[0], shell.to_string_lossy());
        assert_eq!(&argv[1..], ["--norc", "-c", "cargo test"]);
    }

    #[cfg(unix)]
    #[test]
    fn validation_argv_recognizes_a_configured_symlink_to_jsh() {
        let root = ShellTestDir::new("validation-jsh-symlink");
        let jsh = root.executable("jsh-0.4.0");
        let configured = root.0.join("team-shell");
        std::os::unix::fs::symlink(jsh, &configured).unwrap();

        let argv =
            super::unix_pty::validation_command_argv(configured.to_str(), "cargo test").unwrap();

        assert_eq!(&argv[1..], ["--norc", "-c", "cargo test"]);
    }

    #[cfg(unix)]
    #[test]
    fn validation_argv_disables_supported_shell_rc_files_and_rejects_unknown_families() {
        let root = ShellTestDir::new("validation-shell-families");
        let zsh = root.executable("zsh");
        let fish = root.executable("fish");
        let custom = root.executable("custom-shell");

        assert_eq!(
            &super::unix_pty::validation_command_argv(zsh.to_str(), "pwd").unwrap()[1..],
            ["-f", "-c", "pwd"]
        );
        assert_eq!(
            &super::unix_pty::validation_command_argv(fish.to_str(), "pwd").unwrap()[1..],
            ["--no-config", "-c", "pwd"]
        );
        assert!(super::unix_pty::validation_command_argv(custom.to_str(), "pwd").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn jsh_symlinked_to_another_program_is_skipped() {
        let root = ShellTestDir::new("jsh-alternatives");
        let ssh = root.executable("ssh");
        let bash = root.executable("bash");
        // A `jsh` on PATH that is really a symlink to ssh: right name, wrong
        // binary. Exec'ing it would exit 255 and close the only tab.
        std::os::unix::fs::symlink(&ssh, root.0.join("jsh")).unwrap();
        let search_path = root.search_path();

        let selected = super::unix_pty::choose_shell_with_path(
            None,
            Some(&search_path),
            &root.0.join("missing-sh"),
        )
        .unwrap();

        assert_eq!(std::path::Path::new(&selected), bash);
    }

    #[cfg(unix)]
    #[test]
    fn a_real_jsh_binary_is_still_preferred() {
        let root = ShellTestDir::new("jsh-real");
        let jsh = root.executable("jsh");
        root.executable("bash");
        let search_path = root.search_path();

        let selected = super::unix_pty::choose_shell_with_path(
            None,
            Some(&search_path),
            &root.0.join("missing-sh"),
        )
        .unwrap();

        assert_eq!(std::path::Path::new(&selected), jsh);
        assert!(super::unix_pty::is_interactive_jsh(&jsh));
    }

    #[cfg(unix)]
    #[test]
    fn versioned_jsh_builds_remain_eligible() {
        let root = ShellTestDir::new("jsh-versioned");
        let versioned = root.executable("jsh-0.3.0");
        std::os::unix::fs::symlink(&versioned, root.0.join("jsh")).unwrap();

        assert!(super::unix_pty::is_interactive_jsh(&root.0.join("jsh")));
    }

    #[cfg(unix)]
    #[test]
    fn sh_fallback_is_resolved_to_an_executable_path() {
        let root = ShellTestDir::new("path-sh");
        let executable = root.executable("sh");
        let search_path = root.search_path();

        let selected = super::unix_pty::choose_shell_with_path(
            None,
            Some(&search_path),
            &root.0.join("missing-sh"),
        )
        .unwrap();

        assert_eq!(std::path::Path::new(&selected), executable);
        assert_ne!(selected, "sh");
    }

    #[cfg(unix)]
    #[test]
    fn missing_shells_return_an_explicit_error() {
        let root = ShellTestDir::new("missing");
        let error = super::unix_pty::choose_shell_with_path(None, None, &root.0.join("missing-sh"))
            .unwrap_err();

        assert!(error.to_string().contains("No executable shell found"));
    }

    #[cfg(unix)]
    #[test]
    fn missing_working_directory_is_reported_before_returning_a_pty() {
        let root = ShellTestDir::new("missing-cwd");
        let missing_cwd = root.0.join("deleted");
        let error =
            super::Pty::new_with_cwd(80, 24, missing_cwd.to_str(), None, Some("/bin/sh"), None)
                .err()
                .expect("a missing cwd must fail before returning a PTY");

        assert!(
            error
                .to_string()
                .contains("Failed to enter saved working directory"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn exec_failure_is_reported_before_returning_a_pty() {
        use std::os::unix::fs::PermissionsExt;

        let root = ShellTestDir::new("exec-failure");
        let invalid_shell = root.0.join("invalid-shell");
        std::fs::write(&invalid_shell, b"not an executable format").unwrap();
        std::fs::set_permissions(&invalid_shell, std::fs::Permissions::from_mode(0o700)).unwrap();

        let error =
            super::Pty::new_with_cwd(80, 24, Some("/tmp"), None, invalid_shell.to_str(), None)
                .err()
                .expect("execve failure must be reported synchronously");

        assert!(
            error.to_string().contains("Failed to execute shell"),
            "{error}"
        );
    }
    /// ember's child-environment choices are now flags on a shared policy, so
    /// pin the ones a "colour fix" could quietly change: the pager stays `FR`,
    /// `ls` colours stay the user's business, and the locale repair stays on.
    #[cfg(unix)]
    #[test]
    fn child_environment_keeps_the_pager_and_locale_choices() {
        let options = super::unix_pty::child_environment();
        assert_eq!(options.less_default, Some("FR"));
        assert!(!options.color_defaults);
        assert!(options.normalize_locale);
        assert_eq!(options.app_version, env!("CARGO_PKG_VERSION"));

        let pairs = jterm_core::child_env::pairs(&options, &[]);
        let value = |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| key == std::ffi::OsStr::new(name))
                .map(|(_, value)| value.to_string_lossy().into_owned())
        };
        // The reason for the whole exercise: a child that used to be told only
        // TERM now learns the colour depth and which terminal build it is in.
        assert_eq!(value("COLORTERM").as_deref(), Some("truecolor"));
        assert_eq!(
            value("TERM_PROGRAM_VERSION").as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(value("TERM").as_deref(), Some("xterm-256color"));
        assert_eq!(value("VTE_VERSION").as_deref(), Some("7802"));
        assert_eq!(value("LS_COLORS"), None);
        // `pairs` reads this process's environment, so only one of the two
        // outcomes is available at a time — both are assertions about policy.
        match std::env::var_os("LESS") {
            Some(_) => assert_eq!(value("LESS"), None, "a user's LESS is never overridden"),
            None => assert_eq!(value("LESS").as_deref(), Some("FR")),
        }
    }

    /// Resolution moved ahead of the fork: a helper argv naming a program that
    /// is not installed must fail where the caller can report it, instead of
    /// opening a pane whose child exits 127 for no visible reason.
    #[cfg(unix)]
    #[test]
    fn an_unresolvable_command_program_fails_before_forking() {
        let argv = vec!["ember-definitely-not-installed-program".to_string()];
        let error = super::Pty::new_with_cwd(80, 24, Some("/tmp"), None, None, Some(&argv))
            .err()
            .expect("an unknown helper program must not spawn");

        assert!(error.to_string().contains("is not executable"), "{error}");
    }

    /// The one-shot helper path (jsh installer) must exec the given argv
    /// verbatim: no login-shell wrapping, no --session injection.
    #[cfg(unix)]
    #[test]
    fn an_explicit_argv_is_exec_ed_verbatim() {
        let argv = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "printf 'argv0=%s arg=%s' \"$0\" \"$1\"".to_string(),
            "jterm-command-label".to_string(),
            "payload".to_string(),
        ];
        let mut pty = super::Pty::new_with_cwd(80, 24, Some("/tmp"), None, None, Some(&argv))
            .expect("explicit argv must spawn");

        let mut seen = String::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut buf = [0u8; 1024];
        while std::time::Instant::now() < deadline && !seen.contains("arg=payload") {
            match pty.read(&mut buf) {
                Ok(super::ReadOutcome::Data(n)) => {
                    seen.push_str(&String::from_utf8_lossy(&buf[..n]));
                }
                Ok(super::ReadOutcome::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Ok(_) | Err(_) => break,
            }
        }

        assert!(
            seen.contains("argv0=jterm-command-label"),
            "argv[0] must reach the child unchanged: {seen:?}"
        );
        assert!(
            seen.contains("arg=payload"),
            "remaining arguments must reach the child: {seen:?}"
        );
    }

    /// A one-shot command is its own executable and must not depend on the
    /// separately configured interactive shell being valid.
    #[cfg(unix)]
    #[test]
    fn explicit_argv_bypasses_invalid_interactive_shell_config() {
        let argv = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "exit 0".to_string(),
        ];
        let pty = super::Pty::new_with_cwd(
            80,
            24,
            Some("/tmp"),
            None,
            Some("/ember/definitely/missing/interactive-shell"),
            Some(&argv),
        );

        assert!(pty.is_ok(), "explicit argv must not resolve the shell");
    }
}
