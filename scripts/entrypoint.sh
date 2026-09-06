#!/usr/bin/env bash
#
# Cluster entrypoint. Runs one config at one exact git SHA.
#
#   scripts/entrypoint.sh --sha <full-or-short-sha> --config configs/<name>.yaml
#
# The image contains no application code. This script VERIFIES that the requested
# SHA is already present in the NFS checkout, checks it out into a per-SHA
# worktree, and runs from there. Two jobs at different SHAs therefore do not
# fight over one checkout, and the job runs the code you named rather than
# whatever HEAD happens to be.
#
# It does NOT fetch by default. A training workload runs as root with HOME=/root
# and has no SSH credentials (PLAN §2.7), so a fetch fails with an authentication
# error several layers down from the real problem. Code reaches NFS through the
# workspace; the job consumes what is there. If the SHA is missing, that is a
# readable message telling you to pull in the workspace -- not a credential
# failure.
#
# --fetch opts back in, for environments that do have credentials. Off by default.
#
# Every failure is loud. A job that silently ran the wrong code is worse than a
# job that did not run.

set -euo pipefail

REPO="${CLEFT_REPO:-/home/user/codex/cleft-aesthetics}"
WORKTREE_ROOT="${CLEFT_WORKTREE_ROOT:-/home/user/codex/worktrees}"
OUT="${CLEFT_OUT:-/home/user/codex/runs}"
# Site-specific, so configurable. The default is on NFS rather than /tmp because
# a container is recreated on every job and on every Run:AI resume, and /tmp does
# not survive that.
GITCONFIG="${GIT_CONFIG_GLOBAL:-/home/user/codex/.ssh-cleft/gitconfig}"
FETCH=0
SHA=""
CONFIG=""

die() { echo "entrypoint: FATAL: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sha)    SHA="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --out)    OUT="${2:-}"; shift 2 ;;
    --fetch)  FETCH=1; shift ;;
    *)        die "unknown argument: $1 (only --sha, --config, --out and --fetch exist)" ;;
  esac
done

[[ -n "$SHA" ]]    || die "--sha is required"
[[ -n "$CONFIG" ]] || die "--config is required"
[[ -d "$REPO/.git" ]] || die "no git repository at $REPO (set CLEFT_REPO)"

command -v git >/dev/null || die "git is not installed in this image"

# The NFS checkout is owned by 'nobody' and this container runs as root, so git
# refuses to touch it with "detected dubious ownership" and every git command
# exits non-zero. The provenance guard then cannot determine the code state and
# aborts -- correctly, but the job is dead before it starts.
#
# The container is recreated for every job and on every Run:AI resume, so this
# has to be set here rather than once by hand. GIT_CONFIG_GLOBAL points at a
# writable path because HOME may be read-only or reset.
#
# safe.directory relaxes an ownership check on a path we already trust and have
# just fetched ourselves. It is NOT a fix for the ownership itself: do not chown
# the NFS checkout, which is shared.
export GIT_CONFIG_GLOBAL="$GITCONFIG"
mkdir -p "$(dirname "$GIT_CONFIG_GLOBAL")" 2>/dev/null || true
touch "$GIT_CONFIG_GLOBAL" 2>/dev/null \
  || die "cannot write $GIT_CONFIG_GLOBAL. Set GIT_CONFIG_GLOBAL to a writable
  path that survives container recreation -- not /tmp, which does not."
for trusted in "$REPO" "$WORKTREE_ROOT"; do
  git config --global --get-all safe.directory | grep -qxF "$trusted" \
    || git config --global --add safe.directory "$trusted"
done

# Prove it worked before relying on it, rather than assuming the config took.
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "git still cannot read $REPO after adding safe.directory. Run
  'git -C $REPO status' by hand to see the real error; do not chown the checkout."

if [[ "$FETCH" -eq 1 ]]; then
  echo "entrypoint: fetching $REPO (--fetch given)"
  git -C "$REPO" fetch --all --tags --prune \
    || die "fetch failed. A training workload runs as root with HOME=/root and
  has no SSH credentials, so --fetch only works where credentials exist. Drop
  --fetch and pull in the workspace instead."
else
  echo "entrypoint: not fetching (pass --fetch to enable); verifying $SHA is already present"
fi

# VERIFY, do not fetch. A typo'd SHA must not silently resolve to something else,
# a SHA that was force-pushed away must stop the job, and a SHA that simply has
# not reached NFS yet must say exactly that rather than failing as a credential
# error six layers down.
git -C "$REPO" cat-file -e "${SHA}^{commit}" 2>/dev/null || die "commit $SHA is not present in $REPO.

  This job cannot fetch it: the image runs as root with HOME=/root and carries no
  SSH credentials (PLAN §2.7). Code reaches NFS through the workspace, and the
  job consumes what is there.

  Remedy -- in the WORKSPACE, not in a training job:

      cd $REPO && git pull

  Then relaunch this workload with the same --sha. Verify it landed with:

      git -C $REPO cat-file -e ${SHA}^{commit} && echo present

  If credentials do exist in this environment, pass --fetch instead."
FULL_SHA="$(git -C "$REPO" rev-parse "${SHA}^{commit}")"
SHORT_SHA="${FULL_SHA:0:8}"

WORKTREE="$WORKTREE_ROOT/$SHORT_SHA"
mkdir -p "$WORKTREE_ROOT"

if [[ -d "$WORKTREE" ]]; then
  # Run:AI can pause and resume a workload, so this script must be re-runnable.
  # Reusing the existing worktree is correct; recreating it is not.
  echo "entrypoint: reusing worktree $WORKTREE"
  ACTUAL="$(git -C "$WORKTREE" rev-parse HEAD)"
  [[ "$ACTUAL" == "$FULL_SHA" ]] \
    || die "worktree $WORKTREE is at $ACTUAL, expected $FULL_SHA"
else
  echo "entrypoint: creating worktree $WORKTREE at $FULL_SHA"
  git -C "$REPO" worktree add --detach "$WORKTREE" "$FULL_SHA"
fi

CONFIG_PATH="$WORKTREE/$CONFIG"
[[ -f "$CONFIG_PATH" ]] || die "config not found at that SHA: $CONFIG_PATH"

export CLEFT_REPO_ROOT="$WORKTREE"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"

echo "entrypoint: sha=$FULL_SHA config=$CONFIG out=$OUT"
echo "entrypoint: python=$(python -VV)"
echo "entrypoint: image_digest=${CLEFT_IMAGE_DIGEST:-<not set>}"

# Run identity. Run:AI recreates the pod when it resumes a paused workload, so
# the run id must come from the JOB, not from the clock and not from the pod.
# Without one, a resumed job would start a second run directory from epoch 0 and
# silently discard its checkpoint.
JOB_ID="${CLEFT_JOB_ID:-${RUNAI_JOB_ID:-${RUNAI_JOB_UUID:-${RUNAI_JOB_NAME:-}}}}"
if [[ -n "$JOB_ID" ]]; then
  echo "entrypoint: job_id=$JOB_ID (this run is resumable)"
else
  echo "entrypoint: WARNING no job id in the environment." >&2
  echo "entrypoint:   Checked CLEFT_JOB_ID, RUNAI_JOB_ID, RUNAI_JOB_UUID, RUNAI_JOB_NAME." >&2
  echo "entrypoint:   The run id will fall back to a timestamp, so if Run:AI pauses" >&2
  echo "entrypoint:   and resumes this workload it will start a SECOND run directory" >&2
  echo "entrypoint:   from epoch 0. Set CLEFT_JOB_ID in the workload spec." >&2
fi

cd "$WORKTREE"
exec python -m cleft.run --config "$CONFIG_PATH" --out "$OUT"
