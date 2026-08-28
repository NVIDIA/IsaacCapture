#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# camera_viz.sh — one entry point for local development AND robot
# deployment of the camera_viz example.
#
# Local:
#   ./camera_viz.sh setup [--sender-only]      install deps + build codec
#   ./camera_viz.sh loopback CONFIG            run streamer + viz on 127.0.0.1
#   ./camera_viz.sh run CONFIG [--mode M]      run the viewer (honors source:)
#
# Remote (Jetson robot):
#   ./camera_viz.sh deploy --host H --user U [--password P] CONFIG
#   ./camera_viz.sh service-status   --host H --user U [--password P]
#   ./camera_viz.sh service-logs     --host H --user U [--password P]
#   ./camera_viz.sh service-restart  --host H --user U [--password P]
#
# SSH auth: --password uses sshpass (must be installed). Without it,
# falls back to plain ssh — you need keys set up.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$HERE/scripts"
SERVICE_NAME="camera-streamer"
SERVICE_TEMPLATE="$SCRIPTS_DIR/${SERVICE_NAME}.service.in"

# Relative path → resolved against the remote user's home by rsync + ssh.
# Don't use ~ or $HOME: rsync's --protect-args (3.2+ default) blocks
# remote shell expansion in the destination.
REMOTE_DIR='camera_viz'

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

_C_OK="\033[32m"; _C_INFO="\033[36m"; _C_WARN="\033[33m"; _C_ERR="\033[31m"; _C_RESET="\033[0m"
log_info()  { echo -e "${_C_INFO}[info]${_C_RESET}  $*"; }
log_ok()    { echo -e "${_C_OK}[ok]${_C_RESET}    $*"; }
log_warn()  { echo -e "${_C_WARN}[warn]${_C_RESET}  $*" >&2; }
log_error() { echo -e "${_C_ERR}[error]${_C_RESET} $*" >&2; }
log_step()  { echo -e "\n\033[1m=== $* ===${_C_RESET}"; }

# ──────────────────────────────────────────────────────────────────────
# Shared remote arg parsing (--host/--user/--password)
# ──────────────────────────────────────────────────────────────────────

# Sets REMOTE_{HOST,USER,PASSWORD}; remaining positionals → REMOTE_REST[].
# Defaults come from $REMOTE_HOST / $REMOTE_USER / $REMOTE_PASSWORD if
# those are exported in the calling shell; CLI flags override.
parse_remote_args() {
    : "${REMOTE_HOST:=}"
    : "${REMOTE_USER:=}"
    : "${REMOTE_PASSWORD:=}"
    REMOTE_REST=()
    while (( $# )); do
        case $1 in
            --host)     REMOTE_HOST=$2; shift 2;;
            --user)     REMOTE_USER=$2; shift 2;;
            --password) REMOTE_PASSWORD=$2; shift 2;;
            --) shift; REMOTE_REST+=("$@"); break;;
            *)  REMOTE_REST+=("$1"); shift;;
        esac
    done
    [[ -n "$REMOTE_HOST" ]] || { log_error "--host (or \$REMOTE_HOST) is required"; exit 1; }
    [[ -n "$REMOTE_USER" ]] || { log_error "--user (or \$REMOTE_USER) is required"; exit 1; }
    if [[ -n "$REMOTE_PASSWORD" ]] && ! command -v sshpass >/dev/null 2>&1; then
        log_error "REMOTE_PASSWORD set but sshpass not installed. apt install sshpass, or drop the password and use key auth."
        exit 1
    fi
}

# When a password is set we pass it via SSHPASS env var (sshpass -e)
# rather than command-line interpolation. -p would expose it on argv
# AND break on spaces / metacharacters in the password.
_sshpass_env() {
    if [[ -n "$REMOTE_PASSWORD" ]]; then
        export SSHPASS="$REMOTE_PASSWORD"
    fi
}

ssh_run() {
    local cmd="$1"
    _sshpass_env
    if [[ -n "$REMOTE_PASSWORD" ]]; then
        sshpass -e ssh -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$cmd"
    else
        ssh -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$cmd"
    fi
}

# TTY variant — needed for remote sudo prompts.
ssh_run_tty() {
    local cmd="$1"
    _sshpass_env
    if [[ -n "$REMOTE_PASSWORD" ]]; then
        sshpass -e ssh -t -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$cmd"
    else
        ssh -t -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$cmd"
    fi
}

rsync_to_remote() {
    _sshpass_env
    local rsync_ssh="ssh -o StrictHostKeyChecking=accept-new"
    if [[ -n "$REMOTE_PASSWORD" ]]; then
        rsync_ssh="sshpass -e $rsync_ssh"
    fi
    rsync -az --delete \
        --exclude='.venv/' \
        --exclude='codec/build/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        -e "$rsync_ssh" \
        "$HERE/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
}

# ──────────────────────────────────────────────────────────────────────
# setup (local)
# ──────────────────────────────────────────────────────────────────────

cmd_setup() {
    # Intercept --venv so $HERE/.venv becomes a symlink to the caller's
    # path. The rest of camera_viz.sh (cmd_run / cmd_loopback) hardcodes
    # $HERE/.venv; routing it through a symlink lets ``setup --venv PATH``
    # work without touching every command.
    local custom_venv=""
    local rest=()
    while (( $# )); do
        case $1 in
            --venv) custom_venv=$2; shift 2;;
            *)      rest+=("$1"); shift;;
        esac
    done

    if [[ -n "$custom_venv" ]]; then
        custom_venv=$(realpath -m "$custom_venv")
        if [[ -e "$HERE/.venv" && ! -L "$HERE/.venv" ]]; then
            log_error "$HERE/.venv exists as a real directory; rm it (or move it) before using --venv."
            exit 1
        fi
        ln -sfn "$custom_venv" "$HERE/.venv"
        log_info "linked $HERE/.venv → $custom_venv"
    fi

    log_step "Local setup"
    exec "$SCRIPTS_DIR/_install_deps.sh" "${rest[@]}"
}

# ──────────────────────────────────────────────────────────────────────
# Local-command helpers
# ──────────────────────────────────────────────────────────────────────

# Validate CONFIG arg + local venv; on success sets LOCAL_VENV.
_require_local_config() {
    local cmd_name="$1" config="$2"
    [[ -n "$config" ]] || { log_error "usage: camera_viz.sh $cmd_name CONFIG"; exit 1; }
    [[ -f "$config" ]] || { log_error "config not found: $config"; exit 1; }
    LOCAL_VENV="$HERE/.venv"
    [[ -x "$LOCAL_VENV/bin/python" ]] || {
        log_error "no venv at $LOCAL_VENV — run ./camera_viz.sh setup first"
        exit 1
    }
}

# RTP deps are opt-in at setup time (``--with-rtp``), so a venv provisioned
# for direct mode has no ``gi``. Fail here with the fix rather than letting
# the sender die on an ImportError several log lines later.
_require_rtp_deps() {
    "$LOCAL_VENV/bin/python" -c "import gi" >/dev/null 2>&1 && return 0
    log_error "the RTP path needs dependencies this venv doesn't have"
    log_info "Re-run setup with --with-rtp:"
    log_info "  ./camera_viz.sh setup --with-rtp"
    exit 1
}

# Write a copy of CONFIG with ``source: rtp`` forced on. Used by loopback
# so the receiver doesn't need the user to edit the YAML between runs.
# Caller is responsible for rm-ing the returned path.
_rewrite_source_rtp() {
    local src="$1" dst
    dst="$(mktemp -t camera_viz_recv.XXXXXX.yaml)"
    "$LOCAL_VENV/bin/python" - "$src" "$dst" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)
if not isinstance(cfg, dict):
    print(f"camera_viz.sh: {src} must be a YAML mapping (got {type(cfg).__name__})", file=sys.stderr)
    sys.exit(2)
cfg["source"] = "rtp"
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
PY
    echo "$dst"
}

# ──────────────────────────────────────────────────────────────────────
# run (the viewer; args after CONFIG forward to the viewer, e.g. --mode xr)
# ──────────────────────────────────────────────────────────────────────

cmd_run() {
    _require_local_config run "${1:-}"
    # ``source: rtp`` makes the viewer an RTP receiver, so it needs the same
    # opt-in deps as the sender. Direct-mode configs don't.
    if grep -qE '^[[:space:]]*source:[[:space:]]*["'\'']?rtp' "$1"; then
        _require_rtp_deps
    fi
    log_step "Starting camera_viz — Ctrl-C to exit"
    "$LOCAL_VENV/bin/python" -m isaaccapture_examples.camera_viz "$@"
}

# ──────────────────────────────────────────────────────────────────────
# loopback (local)
# ──────────────────────────────────────────────────────────────────────

# Globals so the EXIT trap can reach them after cmd_loopback has
# returned (function locals would be out of scope by then, and ``set
# -u`` then trips on the unbound name).
_LOOPBACK_SENDER_PID=
_LOOPBACK_RECV_CONFIG=

_loopback_cleanup() {
    local pid="${_LOOPBACK_SENDER_PID:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log_info "stopping camera_streamer (pid $pid)"
        kill -INT "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    local cfg="${_LOOPBACK_RECV_CONFIG:-}"
    [[ -n "$cfg" ]] && rm -f "$cfg"
}

cmd_loopback() {
    _require_local_config loopback "${1:-}"
    _require_rtp_deps
    _LOOPBACK_RECV_CONFIG="$(_rewrite_source_rtp "$1")"
    trap '_loopback_cleanup' EXIT

    log_step "Starting camera_streamer → 127.0.0.1 (background)"
    "$LOCAL_VENV/bin/python" -m isaaccapture_examples.camera_viz.camera_streamer "$1" --host 127.0.0.1 &
    _LOOPBACK_SENDER_PID=$!

    log_step "Starting camera_viz (foreground) — Ctrl-C to exit"
    "$LOCAL_VENV/bin/python" -m isaaccapture_examples.camera_viz "$_LOOPBACK_RECV_CONFIG"
}

# ──────────────────────────────────────────────────────────────────────
# deploy (remote)
# ──────────────────────────────────────────────────────────────────────

cmd_deploy() {
    # Pull deploy-only flags out of "$@" before parse_remote_args runs;
    # anything we don't recognize would land in REMOTE_REST (where the
    # positional CONFIG lives).
    local no_service=false
    # ``--streaming-host`` → injected as ``--host`` on the streamer's
    # CLI inside the rendered systemd unit. Lets you keep the YAML at
    # 127.0.0.1 for loopback and override only when deploying to a robot.
    local streaming_host="${STREAMING_HOST:-}"
    local filtered=()
    while (( $# )); do
        case $1 in
            --no-service)     no_service=true; shift;;
            --streaming-host) streaming_host=$2; shift 2;;
            *)                filtered+=("$1"); shift;;
        esac
    done
    set -- "${filtered[@]}"

    parse_remote_args "$@"
    [[ "${#REMOTE_REST[@]}" -eq 1 ]] || {
        log_error "usage: camera_viz.sh deploy --host H --user U [--password P] [--no-service] CONFIG"
        exit 1
    }
    local config="${REMOTE_REST[0]}"
    [[ -f "$HERE/$config" ]] || {
        log_error "config not found: $HERE/$config"
        exit 1
    }
    [[ -f "$SERVICE_TEMPLATE" ]] || {
        log_error "service template missing: $SERVICE_TEMPLATE"
        exit 1
    }
    command -v rsync >/dev/null || { log_error "rsync not installed"; exit 1; }

    log_step "Pushing source → $REMOTE_USER@$REMOTE_HOST:~/$REMOTE_DIR"
    rsync_to_remote
    log_ok "source synced"

    log_step "Installing deps on robot (sender-only, jetson)"
    # ``deploy`` targets Jetson robots, so we always pass --jetson:
    # JetPack ships partial CUDA + skips the unversioned symlinks the
    # cupy loader needs. If the Jetson is missing system packages /
    # symlinks, _install_deps.sh prompts on the TTY (ssh -t) and either
    # runs the sudo commands after the operator types ``y`` or aborts.
    ssh_run_tty "cd $REMOTE_DIR && bash scripts/_install_deps.sh --sender-only --jetson"
    log_ok "deps installed"

    if $no_service; then
        log_ok "source + deps installed (service skipped)"
        local manual_host_flag=""
        if [[ -n "$streaming_host" ]]; then
            manual_host_flag=" --host $streaming_host"
            log_info "(--streaming-host has no effect in --no-service mode — pass it to the streamer yourself.)"
        fi
        log_info "Run manually with:"
        log_info "  ssh $REMOTE_USER@$REMOTE_HOST 'cd ~/camera_viz && .venv/bin/python -m isaaccapture_examples.camera_viz.camera_streamer $config$manual_host_flag'"
        log_info "Re-run without --no-service when you're ready to install the systemd unit."
        return 0
    fi

    # Empty when --streaming-host wasn't given — ExecStart falls back to
    # streaming.host from the YAML. Leading space so the substituted
    # template doesn't end with a trailing space when empty.
    local extra_args=""
    if [[ -n "$streaming_host" ]]; then
        extra_args=" --host $streaming_host"
        log_info "streaming.host overridden: --host $streaming_host"
    fi

    log_step "Installing systemd unit"
    # Render the template on the remote side so $HOME expands there.
    local install_cmd
    install_cmd=$(cat <<REMOTE
set -euo pipefail
workdir="\$HOME/camera_viz"
venv="\$workdir/.venv"
config="\$workdir/$config"
unit_dir="\$HOME/.config/systemd/user"
mkdir -p "\$unit_dir"
sed -e "s|{{WORKDIR}}|\$workdir|g" \
    -e "s|{{VENV}}|\$venv|g" \
    -e "s|{{CONFIG}}|\$config|g" \
    -e "s|{{EXTRA_ARGS}}|${extra_args}|g" \
    "\$workdir/scripts/${SERVICE_NAME}.service.in" \
    > "\$unit_dir/${SERVICE_NAME}.service"
echo "wrote \$unit_dir/${SERVICE_NAME}.service"
systemctl --user daemon-reload
REMOTE
)
    ssh_run "$install_cmd"

    # One-time sudo to enable linger so the service survives logout.
    log_step "Enabling user-mode systemd persistence"
    local linger_cmd
    linger_cmd=$(cat <<REMOTE
if loginctl show-user "$REMOTE_USER" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
    echo "linger already enabled"
else
    echo "enabling linger (sudo required, one-time)"
    sudo loginctl enable-linger "$REMOTE_USER"
fi
REMOTE
)
    ssh_run_tty "$linger_cmd"

    log_step "Enabling + starting service"
    ssh_run "systemctl --user enable --now ${SERVICE_NAME}.service"
    log_ok "deployed."

    log_info "Tail logs with:   ./camera_viz.sh service-logs --host $REMOTE_HOST --user $REMOTE_USER"
    log_info "Check status:     ./camera_viz.sh service-status --host $REMOTE_HOST --user $REMOTE_USER"
}

# ──────────────────────────────────────────────────────────────────────
# service-{status,logs,restart}
# ──────────────────────────────────────────────────────────────────────

cmd_service_status() {
    parse_remote_args "$@"
    ssh_run "systemctl --user status ${SERVICE_NAME}.service --no-pager" || true
}

cmd_service_logs() {
    parse_remote_args "$@"
    # ssh -t so Ctrl-C reaches journalctl cleanly.
    ssh_run_tty "journalctl --user -u ${SERVICE_NAME}.service -f"
}

cmd_service_restart() {
    parse_remote_args "$@"
    ssh_run "systemctl --user restart ${SERVICE_NAME}.service"
    log_ok "restarted"
}

# ──────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────

show_help() {
    cat <<EOF
camera_viz.sh — local development + Jetson deployment for camera_viz

LOCAL
    setup [--venv PATH] [--sender-only] [--jetson]
          [--no-v4l2] [--no-oakd] [--with-rtp] [--with-zed]
          [--wheel PATH | --build-from-source]
                          Create .venv, install Python deps via uv into
                          the venv, build native codec. Python deps stay
                          inside .venv (no --system-site-packages).
                          System-side prerequisites (GStreamer plugins,
                          cairo/girepository dev headers, cuda-nvrtc on
                          Jetson, etc.) are PROBED. If anything is
                          missing the script prints the exact apt-get
                          command and prompts ``y/N`` before running it.
                          Reply N or run non-interactively to abort.
                          --venv PATH installs into an existing venv at
                          PATH instead of creating one in-place.
                          examples/camera_viz/.venv is symlinked → PATH
                          so run / loopback pick it up too.
                          --with-rtp adds the split-mode deps (GStreamer
                          system packages, PyGObject, native codec).
                          Needed for loopback, deploy, and any config
                          with ``source: rtp``. Implied by --sender-only.
                          --sender-only skips isaaccapture + vulkan deps
                          (use on Jetson sender hosts).
                          --jetson adds JetPack-only checks: unversioned
                          CUDA lib symlinks + ld.so wiring that JetPack
                          skips. Off on desktop.
                          isaaccapture is taken from the package index,
                          preferring a final release and falling back to
                          a release candidate. If neither is available,
                          setup offers to build it from this checkout;
                          --build-from-source picks that without asking,
                          --wheel PATH installs a wheel you already built.

    loopback CONFIG       Run camera_streamer + camera_viz on 127.0.0.1.

    run CONFIG [--mode xr|window]
                          Run the viewer with the YAML as-is. ``source:
                          local`` opens cameras directly; ``source: rtp``
                          listens on rtp.port (sender IP irrelevant — the
                          receiver binds 0.0.0.0). --mode overrides
                          display.mode: xr renders to the headset (the
                          default), window to a desktop window.

REMOTE (Jetson robot)
    deploy [--host H --user U [--password P]]
           [--streaming-host IP] [--no-service] CONFIG
                          rsync source, install deps, install + start
                          systemd user service running the streamer.
                          --no-service stops after deps so you can run
                          the streamer by hand first.
                          --streaming-host injects ``--host IP`` into the
                          unit's ExecStart so the sender streams there
                          regardless of streaming.host in the YAML. Same
                          knob via $STREAMING_HOST env var. Leave unset
                          to use the YAML value.

    service-status   [--host H --user U [--password P]]
    service-logs     [--host H --user U [--password P]]
    service-restart  [--host H --user U [--password P]]
                          Inspect / manage the deployed robot RTP service.

ENVIRONMENT (remote commands)
    REMOTE_HOST, REMOTE_USER, REMOTE_PASSWORD
                          Defaults for --host / --user / --password. CLI
                          flags override. Drop the flags from your shell
                          history by exporting these once per session.

    STREAMING_HOST        Default for --streaming-host on ``deploy``.

EXAMPLES
    ./camera_viz.sh setup
    ./camera_viz.sh loopback configs/v4l2.yaml
    ./camera_viz.sh deploy --host 10.29.90.127 --user nvidia configs/v4l2.yaml
    ./camera_viz.sh run configs/v4l2.yaml
    ./camera_viz.sh run configs/replay.yaml --mode window   # no headset needed

    # Env-var style (avoids passwords in shell history / argv):
    export REMOTE_HOST=10.29.90.127 REMOTE_USER=nvidia
    read -s REMOTE_PASSWORD && export REMOTE_PASSWORD
    ./camera_viz.sh deploy configs/v4l2.yaml
    ./camera_viz.sh service-logs

SSH AUTH
    Without a password set, uses your SSH keys. With one, uses sshpass
    (apt install sshpass). Passwords are forwarded to sshpass via the
    SSHPASS env var (sshpass -e), so they don't appear in process argv.
EOF
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

[[ $# -eq 0 ]] && { show_help; exit 0; }

cmd="$1"; shift
case "$cmd" in
    setup)            cmd_setup "$@" ;;
    loopback)         cmd_loopback "$@" ;;
    run)              cmd_run "$@" ;;
    deploy)           cmd_deploy "$@" ;;
    service-status)   cmd_service_status "$@" ;;
    service-logs)     cmd_service_logs "$@" ;;
    service-restart)  cmd_service_restart "$@" ;;
    -h|--help|help)   show_help ;;
    *) log_error "unknown command: $cmd"; show_help; exit 1 ;;
esac
