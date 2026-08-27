#!/usr/bin/env bash
#
# install.sh -- set up the local inference stack on this machine.
#
# What this does: the service account, directories, API key, wrapper script,
# llama-swap config and the systemd unit.
#
# What it deliberately does NOT do: build llama.cpp against Intel's SYCL
# backend, or download ~22GB of model weights. The build is hardware-specific
# and takes a long time, and the weights come from HuggingFace. This script
# checks for both and tells you exactly what is missing and how to get it,
# rather than pretending to be a one-command bootstrap it cannot be.
#
# Idempotent: safe to re-run. An existing API key and existing models are
# never touched.
#
# Usage:  sudo ./install.sh [--prefix /var/lib/agent-forge] [--user svc-agent-forge]
#                           [--listen 0.0.0.0:8090] [--no-start]
#
set -euo pipefail

PREFIX=/var/lib/agent-forge
ETC_DIR=/etc/agent-forge
SVC_USER=svc-agent-forge
LISTEN="0.0.0.0:8090"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DO_START=1
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)   PREFIX="${2:?}"; shift 2 ;;
        --user)     SVC_USER="${2:?}"; shift 2 ;;
        --listen)   LISTEN="${2:?}"; shift 2 ;;
        --no-start) DO_START=0; shift ;;
        --force)    FORCE=1; shift ;;
        -h|--help)  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n== %s\n' "$*"; }
warn()  { printf '  ! %s\n' "$*"; }
die()   { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "must run as root: sudo $0"
command -v systemctl >/dev/null 2>&1 || die "systemd is required"

# --- refuse to hijack an existing installation ------------------------------
# The unit path, the wrapper and $ETC_DIR are fixed locations regardless of
# --prefix, so installing to a different prefix would repoint the running
# service at a tree it knows nothing about. The service keeps going and then
# comes up against the wrong paths at its next restart, which is a horrible way
# to find out. Checked before anything is written, not partway through.
EXISTING_UNIT=/etc/systemd/system/llama-swap.service
if [ -f "$EXISTING_UNIT" ] && [ "$FORCE" -eq 0 ]; then
    EXISTING_PREFIX="$(sed -n 's|.*-config \(.*\)/llama-swap\.yaml.*|\1|p' "$EXISTING_UNIT" | head -1)"
    if [ -n "$EXISTING_PREFIX" ] && [ "$EXISTING_PREFIX" != "$PREFIX" ]; then
        die "llama-swap is already installed against $EXISTING_PREFIX, not $PREFIX.
       Installing would repoint the running service, and would overwrite the
       shared wrapper at /usr/local/bin/llama-server-sycl with one pointing at
       a different tree.
       Tear the existing one down first (./uninstall.sh), or pass --force."
    fi
fi

head_ "Plan"
say "prefix : $PREFIX"
say "config : $ETC_DIR"
say "user   : $SVC_USER"
say "listen : $LISTEN"

# --- service account --------------------------------------------------------
head_ "Service account"
if id "$SVC_USER" >/dev/null 2>&1; then
    say "$SVC_USER already exists"
else
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
    say "created $SVC_USER"
fi

# --- directories ------------------------------------------------------------
head_ "Directories"
# HOME is set to $PREFIX in the wrapper and the unit for a reason: Intel's
# compute runtime needs a writable cache for SYCL kernel JIT, and a homeless
# system user otherwise gets $HOME=/home/<user>, which does not exist. The
# symptom is not an error -- llama-server hangs indefinitely on model load.
install -d -o "$SVC_USER" -g "$SVC_USER" -m 750 "$PREFIX"
install -d -o "$SVC_USER" -g "$SVC_USER" -m 750 "$PREFIX/models" "$PREFIX/cache"
install -d -o root -g "$SVC_USER" -m 750 "$ETC_DIR"
say "$PREFIX/{models,cache} and $ETC_DIR"

# --- API key ----------------------------------------------------------------
head_ "API key"
KEY_FILE="$ETC_DIR/llama-swap.env"
if [ -f "$KEY_FILE" ]; then
    say "existing key kept at $KEY_FILE"
else
    umask 077
    printf 'LLAMA_SWAP_API_KEY=%s\n' "$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 40)" > "$KEY_FILE"
    chown root:"$SVC_USER" "$KEY_FILE"; chmod 640 "$KEY_FILE"
    say "generated a new key at $KEY_FILE (root:$SVC_USER, 640)"
fi

# --- wrapper and config -----------------------------------------------------
head_ "Configuration"
sed -e "s|/var/lib/agent-forge|$PREFIX|g" "$REPO_DIR/configs/llama-server-sycl" \
    > /usr/local/bin/llama-server-sycl
chmod 755 /usr/local/bin/llama-server-sycl
say "wrapper -> /usr/local/bin/llama-server-sycl"

if [ -f "$PREFIX/llama-swap.yaml" ]; then
    say "existing llama-swap.yaml kept (delete it to re-seed from the repo)"
else
    sed -e "s|/var/lib/agent-forge|$PREFIX|g" "$REPO_DIR/configs/llama-swap.yaml" \
        > "$PREFIX/llama-swap.yaml"
    chown "$SVC_USER:$SVC_USER" "$PREFIX/llama-swap.yaml"; chmod 640 "$PREFIX/llama-swap.yaml"
    say "llama-swap.yaml -> $PREFIX/"
fi

# --- prerequisites ----------------------------------------------------------
head_ "Prerequisites"
MISSING=0

if [ -x /usr/local/bin/llama-swap ]; then
    say "ok   llama-swap binary"
else
    warn "MISSING /usr/local/bin/llama-swap"
    warn "        get a release binary from github.com/mostlygeek/llama-swap"
    MISSING=1
fi

if [ -x /usr/local/bin/llama-server ]; then
    say "ok   llama-server binary"
    if [ -d /usr/local/lib/agent-forge ] && ls /usr/local/lib/agent-forge/*.so* >/dev/null 2>&1; then
        say "ok   SYCL shared libraries"
    else
        # The build output uses $ORIGIN-relative RPATH, so the executable alone
        # fails at load with "cannot open shared object file".
        warn "MISSING /usr/local/lib/agent-forge/*.so -- llama-server needs its"
        warn "        sibling libraries from the build's bin/ directory"
        MISSING=1
    fi
else
    warn "MISSING /usr/local/bin/llama-server -- build it (see docs/agent-forge.md):"
    warn "        cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx \\"
    warn "              -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DCMAKE_BUILD_TYPE=Release"
    warn "        requires intel-oneapi-compiler-dpcpp-cpp and intel-oneapi-mkl-devel"
    MISSING=1
fi

[ -f /opt/intel/oneapi/setvars.sh ] && say "ok   oneAPI at /opt/intel/oneapi" || {
    warn "MISSING /opt/intel/oneapi/setvars.sh -- the wrapper sources it"; MISSING=1; }

# Models named by the config that are not on disk.
NEED_MODELS=()
while read -r gguf; do
    [ -n "$gguf" ] && [ ! -f "$PREFIX/models/$gguf" ] && NEED_MODELS+=("$gguf")
done < <(grep -oP '\$\{models_dir\}/\K[^ ]+\.gguf' "$PREFIX/llama-swap.yaml" | sort -u)

if [ "${#NEED_MODELS[@]}" -eq 0 ]; then
    say "ok   all models named by llama-swap.yaml are present"
else
    warn "MISSING ${#NEED_MODELS[@]} model file(s) in $PREFIX/models:"
    for m in "${NEED_MODELS[@]}"; do warn "        $m"; done
    warn "        these are bartowski GGUF quants on HuggingFace, e.g.:"
    warn "        hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf \\"
    warn "           --local-dir $PREFIX/models"
    warn "        a model listed in the config but absent only fails when requested,"
    warn "        so the service still starts -- trim llama-swap.yaml if you want fewer"
    MISSING=1
fi

# --- unit -------------------------------------------------------------------
head_ "Service"
sed -e "s|/var/lib/agent-forge|$PREFIX|g" \
    -e "s|/etc/agent-forge|$ETC_DIR|g" \
    -e "s|^User=.*|User=$SVC_USER|" \
    -e "s|^Group=.*|Group=$SVC_USER|" \
    -e "s|-listen [0-9.]*:[0-9]*|-listen $LISTEN|" \
    "$REPO_DIR/systemd/llama-swap.service" > /etc/systemd/system/llama-swap.service
systemctl daemon-reload
say "installed /etc/systemd/system/llama-swap.service"

# media-digest is a reference job, not something to run continuously by default.
# Its units are installed so it stays available, but deliberately left disabled --
# enabling it is a choice.
for u in media-digest.service media-digest.timer; do
    if [ -f "$REPO_DIR/systemd/$u" ]; then
        sed -e "s|/var/lib/agent-forge|$PREFIX|g" -e "s|^User=.*|User=$SVC_USER|" \
            -e "s|^Group=.*|Group=$SVC_USER|" "$REPO_DIR/systemd/$u" > "/etc/systemd/system/$u"
    fi
done
systemctl daemon-reload
say "media-digest units installed but left disabled by default"

if [ "$MISSING" -ne 0 ]; then
    head_ "Not starting"
    say "prerequisites above are missing. Everything else is in place --"
    say "re-run this script once they are, or start manually:"
    say "  systemctl enable --now llama-swap"
    exit 0
fi

if [ "$DO_START" -eq 1 ]; then
    systemctl enable --now llama-swap.service
    sleep 3
    if systemctl is-active --quiet llama-swap.service; then
        say "llama-swap is running"
    else
        say "llama-swap did not come up -- journalctl -u llama-swap -n 40"
    fi
else
    say "not started (--no-start)"
fi

head_ "Done"
KEY="$(. "$KEY_FILE"; echo "$LLAMA_SWAP_API_KEY")"
cat <<NEXT
  Endpoint : http://${LISTEN}/v1
  Key      : $KEY_FILE

  Check it:
    curl -s http://127.0.0.1:${LISTEN##*:}/health
    curl -s -H "Authorization: Bearer \$LLAMA_SWAP_API_KEY" \\
         http://127.0.0.1:${LISTEN##*:}/v1/models

  Point anything speaking the OpenAI API at it, using this URL and key. Open WebUI
  is not managed by this script; see docs/agent-forge.md for the container it
  runs as.

  Tear it all down with: sudo ./uninstall.sh
NEXT
