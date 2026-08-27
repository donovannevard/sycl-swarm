#!/usr/bin/env bash
#
# uninstall.sh -- remove the local inference stack from this machine.
#
# By default this removes the service, units, binaries, config and the service
# account, but KEEPS two things it would be rude to delete without being asked:
#
#   * the model weights (~22GB in <prefix>/models) -- a long re-download
#   * the Open WebUI container and its volume -- that volume holds your chat
#     history and your login
#
# Use --purge for the weights and --remove-webui for the container. This script
# only touches what it installed; anything else on the machine is left alone.
#
# It prints what it is about to do and asks, unless given --yes.
#
# Usage:  sudo ./uninstall.sh [--purge] [--remove-webui] [--yes]
#                             [--prefix /var/lib/agent-forge] [--user svc-agent-forge]
#
set -euo pipefail

PREFIX=/var/lib/agent-forge
ETC_DIR=/etc/agent-forge
SVC_USER=svc-agent-forge
PURGE=0
REMOVE_WEBUI=0
ASSUME_YES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)       PREFIX="${2:?}"; shift 2 ;;
        --user)         SVC_USER="${2:?}"; shift 2 ;;
        --purge)        PURGE=1; shift ;;
        --remove-webui) REMOVE_WEBUI=1; shift ;;
        --yes|-y)       ASSUME_YES=1; shift ;;
        -h|--help)      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n== %s\n' "$*"; }
die()   { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "must run as root: sudo $0"

# --- --prefix does NOT isolate this script ----------------------------------
# Most of what gets removed lives at fixed paths -- /usr/local/bin/llama-*,
# /usr/local/lib/agent-forge, $ETC_DIR, the unit files -- so passing a
# different --prefix does not make this safe to "try out": it still deletes the
# real installation's binaries, libraries and API key, and only the data
# directory follows the flag. Refuse when the installed unit disagrees with the
# prefix given, so a mistyped or experimental --prefix cannot destroy a working
# install. (Learned the hard way: doing exactly that wiped a built llama-server
# and 215MB of SYCL libraries that then had to be recompiled.)
INSTALLED_UNIT=/etc/systemd/system/llama-swap.service
if [ -f "$INSTALLED_UNIT" ]; then
    INSTALLED_PREFIX="$(sed -n 's|.*-config \(.*\)/llama-swap\.yaml.*|\1|p' "$INSTALLED_UNIT" | head -1)"
    if [ -n "$INSTALLED_PREFIX" ] && [ "$INSTALLED_PREFIX" != "$PREFIX" ]; then
        die "the installed llama-swap uses $INSTALLED_PREFIX, but --prefix says $PREFIX.
       This script removes shared paths (/usr/local/bin/llama-*,
       /usr/local/lib/agent-forge, $ETC_DIR, the unit files) regardless of
       --prefix, so continuing would tear down the real installation.
       Re-run with --prefix $INSTALLED_PREFIX to remove it deliberately."
    fi
fi

# --- what is actually here --------------------------------------------------
head_ "Will remove"
[ -f /etc/systemd/system/llama-swap.service ] && say "systemd unit  llama-swap.service"
for u in media-digest.service media-digest.timer; do
    [ -f "/etc/systemd/system/$u" ] && say "systemd unit  $u"
done
for b in /usr/local/bin/llama-swap /usr/local/bin/llama-server /usr/local/bin/llama-server-sycl; do
    [ -e "$b" ] && say "binary        $b"
done
[ -d /usr/local/lib/agent-forge ] && say "libraries     /usr/local/lib/agent-forge ($(du -sh /usr/local/lib/agent-forge 2>/dev/null | cut -f1))"
[ -d "$ETC_DIR" ] && say "config        $ETC_DIR (including the API key)"
id "$SVC_USER" >/dev/null 2>&1 && say "account       $SVC_USER"

MODELS_SIZE="$(du -sh "$PREFIX/models" 2>/dev/null | cut -f1 || echo 0)"
if [ "$PURGE" -eq 1 ]; then
    [ -d "$PREFIX" ] && say "data          $PREFIX INCLUDING models ($MODELS_SIZE)  <- --purge"
else
    head_ "Will keep"
    [ -d "$PREFIX/models" ] && say "models        $PREFIX/models ($MODELS_SIZE) -- re-run with --purge to delete"
    [ -d "$PREFIX" ] && say "data          $PREFIX (jobs.db, caches)"
fi

# Match with a here-string, never `producer | grep -q`. grep -q exits on the
# first match and closes the pipe; the producer then takes SIGPIPE, and under
# `set -o pipefail` that turns a successful match into a failed condition. It
# is intermittent, because it depends on whether the producer had already
# finished writing. A here-string has no producer process to kill.
DOCKER_NAMES=""
command -v docker >/dev/null 2>&1 && DOCKER_NAMES="$(docker ps -a --format '{{.Names}}' 2>/dev/null || true)"
if grep -qx open-webui <<< "$DOCKER_NAMES"; then
    if [ "$REMOVE_WEBUI" -eq 1 ]; then
        printf '  %s\n' "container     open-webui (and its volume: chat history and login)  <- --remove-webui"
    else
        printf '  %s\n' "container     open-webui -- left running; --remove-webui to delete it and its volume"
    fi
fi

# --- warn about dependants --------------------------------------------------
UNIT_FILES="$(systemctl list-unit-files 2>/dev/null || true)"
head_ "Note"
say "Anything else on this machine pointed at this endpoint (an OpenAI-compatible"
say "client using its LLM_BASE_URL / API key) will start failing to connect once"
say "this is removed. Check for other services before continuing if you are not"
say "sure what points here."

# --- confirm ----------------------------------------------------------------
if [ "$ASSUME_YES" -ne 1 ]; then
    printf '\nProceed? [y/N] '
    reply=""
    # `[ -r /dev/tty ]` is true even with no controlling terminal -- the node
    # exists and the mode allows it, but opening it fails with ENXIO. The only
    # honest test is to try. Falling back to stdin keeps `echo n | ...` working.
    if : 2>/dev/null < /dev/tty; then
        read -r reply 2>/dev/null < /dev/tty || reply=""
    elif [ ! -t 0 ]; then
        read -r reply || reply=""
    else
        printf '\nNo terminal to confirm on. Re-run with --yes if you mean it.\n'
    fi
    case "$reply" in [yY]*) ;; *) echo; echo "Aborted; nothing changed."; exit 0 ;; esac
fi

# --- stop and disable -------------------------------------------------------
head_ "Stopping"
for u in media-digest.timer media-digest.service llama-swap.service; do
    if grep -q "^${u}" <<< "$UNIT_FILES"; then
        systemctl disable --now "$u" >/dev/null 2>&1 || true
        say "stopped and disabled $u"
    fi
done

# Any llama-server processes llama-swap spawned outlive it if it was killed
# rather than asked to stop, so make sure none are left holding the GPU.
if pgrep -f '/usr/local/bin/llama-server' >/dev/null 2>&1; then
    pkill -f '/usr/local/bin/llama-server' || true
    say "killed leftover llama-server processes"
fi

# --- remove -----------------------------------------------------------------
head_ "Removing"
for u in llama-swap.service media-digest.service media-digest.timer; do
    [ -f "/etc/systemd/system/$u" ] && { rm -f "/etc/systemd/system/$u"; say "unit $u"; }
done
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true

for b in /usr/local/bin/llama-swap /usr/local/bin/llama-server /usr/local/bin/llama-server-sycl; do
    [ -e "$b" ] && { rm -f "$b"; say "$b"; }
done
[ -d /usr/local/lib/agent-forge ] && { rm -rf /usr/local/lib/agent-forge; say "/usr/local/lib/agent-forge"; }
[ -d "$ETC_DIR" ] && { rm -rf "$ETC_DIR"; say "$ETC_DIR"; }

if [ "$PURGE" -eq 1 ]; then
    [ -d "$PREFIX" ] && { rm -rf "$PREFIX"; say "$PREFIX (including models)"; }
else
    # Keep the weights, drop everything that is cheap to rebuild.
    for d in "$PREFIX/cache" "$PREFIX/.cache" "$PREFIX/venv" "$PREFIX/jobs"; do
        [ -d "$d" ] && { rm -rf "$d"; say "$d"; }
    done
    rm -f "$PREFIX/llama-swap.yaml"
    say "kept $PREFIX/models ($MODELS_SIZE)"
fi

if [ "$REMOVE_WEBUI" -eq 1 ] && command -v docker >/dev/null 2>&1; then
    docker rm -f open-webui >/dev/null 2>&1 && say "container open-webui" || true
    docker volume rm open-webui >/dev/null 2>&1 && say "volume open-webui" || true
fi

# --- account ----------------------------------------------------------------
# Only once nothing it owns is left, and never if --purge was not given and its
# home still holds the models.
if id "$SVC_USER" >/dev/null 2>&1; then
    if [ "$PURGE" -eq 1 ] || [ ! -d "$PREFIX" ]; then
        userdel "$SVC_USER" 2>/dev/null && say "account $SVC_USER" || say "could not remove $SVC_USER (still owns files?)"
    else
        say "kept account $SVC_USER -- it still owns $PREFIX/models"
    fi
fi

head_ "Done"
if [ "$PURGE" -eq 1 ]; then
    say "Fully removed."
else
    say "Removed. Models kept at $PREFIX/models ($MODELS_SIZE)."
    say "Re-running ./install.sh will pick them up as they are."
    say "To finish the job: sudo $0 --purge"
fi
