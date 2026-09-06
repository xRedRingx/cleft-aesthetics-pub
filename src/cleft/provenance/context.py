"""The run context: owns the run directory and all three guards.

CLEFT_PIPELINE_PLAN_v1.md 2.4-2.5. Created once at run start, before any work.
Nothing in the project writes an output except through ``ctx.path()`` or
``ctx.atomic()``.

Run identity is derived from the Run:AI job ID when one is available, not from a
timestamp. Run:AI pauses a workload by deleting and recreating the pod, so the
entrypoint re-runs from the top; a timestamped run id would produce a second
directory for one logical run, guard 2 would never fire, and the job would
silently restart from epoch 0 and discard its checkpoint. Hours, with no error.
A job-derived id makes a resumed pod land in the same directory, which turns
guard 2 from an abort into a resume check.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..config import load_config
from . import gitinfo
from .hashing import hash_file, hash_path

TIERS = ("keeper", "dev")
OUTPUT_TIERS = ("SHAREABLE", "CLUSTER-ONLY")

#: Filenames that are patient-keyed by nature. The clinical cohort never leaves
#: EHU infrastructure, so these can only ever be CLUSTER-ONLY.
#: Filenames that are patient-keyed by nature. The clinical cohort never leaves
#: EHU infrastructure, so these can only ever be CLUSTER-ONLY.
#:
#: ``contact_sheet``/``overlay``/``montage``/``thumbnail`` are here because a
#: rendered contact sheet contains actual patient FACES -- the single worst thing
#: that could be mislabelled SHAREABLE. A plot of aggregate curves is fine and
#: deliberately not matched; anything that renders the images themselves is not.
PATIENT_KEYED = re.compile(
    r"(predictions|oof|patient|per_image|per-image"
    r"|contact_sheet|contact-sheet|overlay|montage|thumbnail)",
    re.IGNORECASE,
)

#: Environment variables that identify the *job*, in priority order. Every one of
#: these must be stable across pod recreation, which is the whole point.
#:
#: ``CLEFT_JOB_ID`` is first so the workload spec can set it explicitly and not
#: depend on which variables a given Run:AI version injects.
#:
#: Pod-level variables (``POD_NAME``, ``HOSTNAME``, ``RUNAI_POD_*``) are
#: deliberately NOT in this list. They change when the pod is recreated, which is
#: exactly the failure being fixed.
#: UUIDs before names, because a workload name is reusable and a UUID is not.
#: Two workloads called "test3" would otherwise share a run directory.
#:
#: Measured on the cluster 2026-07-27: RUNAI_JOB_ID and RUNAI_JOB_UUID do NOT
#: exist. What is actually present is JOB_UUID / jobUUID (a real UUID) and
#: RUNAI_JOB_NAME / JOB_NAME / jobName ("test3"). The Run:AI names were kept in
#: the list anyway in case a later version injects them; they cost one dict
#: lookup. Environment variables are case-sensitive, so the camelCase spellings
#: are separate entries rather than duplicates.
JOB_ID_ENV_VARS = (
    "CLEFT_JOB_ID",
    "JOB_UUID",
    "jobUUID",
    "RUNAI_JOB_UUID",
    "RUNAI_JOB_ID",
    "RUNAI_JOB_NAME",
    "JOB_NAME",
    "jobName",
)

#: The subset above that identifies a workload only by name. A name can be reused
#: across workloads, so a run id built from one is not guaranteed unique.
NAME_ONLY_JOB_VARS = frozenset({"RUNAI_JOB_NAME", "JOB_NAME", "jobName"})

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REQUIREMENTS = _REPO_ROOT / "docker" / "requirements.txt"


class GuardError(RuntimeError):
    """A mechanical guard refused to let the run proceed."""


def _stamp(moment: datetime) -> str:
    """Compact UTC timestamp, milliseconds. ``%f`` is microseconds, so trim 3."""
    return moment.strftime(TIMESTAMP_FORMAT)[:-4] + "Z"


def _sanitize_token(value: str) -> str:
    """Make a job id safe as one path component, without losing its identity."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned[:48] or "unnamed-job"


def job_token() -> tuple[str | None, str | None]:
    """The stable job identifier and which variable it came from."""
    for name in JOB_ID_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return _sanitize_token(value), name
    return None, None


def _read_declared_python() -> tuple[str | None, str]:
    """The python version declared in docker/requirements.txt, and its status.

    The declaration is measured by ``build-image.yml``, which runs ``python -VV``
    inside the built image and fails the build if it disagrees with this file. It
    is never taken from a cluster workspace: the workspace runs as ``jovyan``
    under conda and is not the pinned image.
    """
    try:
        text = _REQUIREMENTS.read_text(encoding="utf-8")
    except OSError:
        return None, "MISSING"
    version = re.search(r"^#\s*python:\s*(\d+\.\d+\.\d+)\s*$", text, re.MULTILINE)
    status = re.search(
        r"^#\s*python_declaration_status:\s*(MEASURED|UNVERIFIED)\s*$", text, re.MULTILINE
    )
    return (
        version.group(1) if version else None,
        status.group(1) if status else "UNDECLARED",
    )


DECLARED_PYTHON, DECLARED_PYTHON_STATUS = _read_declared_python()


def is_absolute_path(raw: str) -> bool:
    """Is this an absolute path on ANY platform?

    ``Path("/home/user/codex/bch/photos").is_absolute()`` is **False on Windows**
    -- a POSIX path has no drive letter, so Windows calls it relative. A config
    written for the cluster would then be silently re-rooted under the repo, and
    guard 3 would hash a path nobody declared and either pass or fail for reasons
    unrelated to the data.

    The cluster is Linux and the answer has to agree with it wherever this runs.
    """
    return bool(PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).drive)


def resolve_declared_path(raw: str, repo_root: Path) -> Path:
    """A declared input path: absolute as given, relative to the repo otherwise."""
    if is_absolute_path(raw):
        return Path(raw)
    return (Path(repo_root) / raw).resolve()


def default_repo_root() -> Path:
    """The repository the running code came from.

    ``CLEFT_REPO_ROOT`` lets the cluster entrypoint point this at the worktree it
    checked out. It is deliberately not a CLI flag: it is not a scientific
    setting, and it cannot change what the run computes.
    """
    override = os.environ.get("CLEFT_REPO_ROOT")
    return Path(override) if override else _REPO_ROOT


# --------------------------------------------------------------------------
# atomic writes
# --------------------------------------------------------------------------


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` all-or-nothing.

    Run:AI can pause a workload mid-write. A partially written checkpoint that
    still loads is worse than one that fails, so the write goes to a temporary
    file in the same directory and is then renamed; ``os.replace`` is atomic on
    POSIX and Windows alike, and rename within a directory is atomic on NFS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str | None]:
    """Versions of the packages that can change a result. Absent ones are null.

    Cached: installed versions cannot change inside a running process, and this
    reads a dist-info directory per package. Uncached it was the single largest
    cost in the test suite -- roughly 150 RunContext constructions x 12 lookups.
    """
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "numpy",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "torch-geometric",
        "opencv-python-headless",
        "Pillow",
        "pandas",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _gpu_allocation() -> dict:
    """What Run:AI actually granted, which is not always a whole device.

    Measured on the cluster: ``RUNAI_NUM_OF_GPUS=0.11``. A fractional allocation
    shares one physical GPU and caps memory well below the device total, so both
    throughput and the largest workable batch size depend on it. Recording only
    the GPU name would make two runs on "RTX PRO 6000" look like the same
    hardware budget when one had a ninth of it.
    """
    raw = os.environ.get("RUNAI_NUM_OF_GPUS")
    try:
        requested = float(raw) if raw not in (None, "") else None
    except ValueError:
        requested = None
    return {
        "runai_num_of_gpus": raw,
        "gpus_requested": requested,
        "fractional_gpu": requested is not None and 0.0 < requested < 1.0,
    }


@lru_cache(maxsize=1)
def _torch_device_info() -> dict:
    """What torch reports about the hardware. Cached: it cannot change mid-process.

    Deliberately does NOT include the Run:AI allocation. That comes from an
    environment variable which a different workload sets differently, so caching
    it would make every run after the first report the first one's allocation.
    A test caught exactly that.
    """
    info = {
        "available": False,
        "version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_total_memory_bytes": None,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch absence must not break provenance
        return info
    info["version"] = getattr(torch.version, "cuda", None)
    try:
        if torch.cuda.is_available():
            info["available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception:  # noqa: BLE001 - a driver mismatch must be recorded, not fatal
        pass
    return info


def _cuda_info() -> dict:
    """Hardware facts plus the allocation. Cheap: only the torch half is cached."""
    return {**_torch_device_info(), **_gpu_allocation()}


class RunContext:
    """Created once at run start. Owns the run directory and all guards."""

    def __init__(
        self,
        config_path: str | Path,
        out_root: str | Path,
        *,
        repo_root: str | Path | None = None,
        tier: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.out_root = Path(out_root)
        self.repo_root = Path(repo_root) if repo_root else default_repo_root()
        self.warnings: list[str] = []

        # 1. Config first: an unknown key is fatal before anything is created.
        self.config = load_config(self.config_path)
        declared_tier = self.config["tier"]
        if tier is not None and tier != declared_tier:
            raise GuardError(
                f"tier argument {tier!r} disagrees with the config's tier "
                f"{declared_tier!r}. The config is the provenance record."
            )

        # 2. Environment.
        self.git = gitinfo.inspect_repo(self.repo_root)
        self.now = now or datetime.now(timezone.utc)
        self._python = self._check_python()

        # 3. Guard 1 - git state: clean, dirty, or unanswerable.
        self.tier = self._apply_git_guard(declared_tier)
        self._check_pinned_image()

        # 4./5. Run directory identity, then Guard 2 - clobber or resume.
        self.job_id, self.run_id_source = job_token()
        if self.run_id_source in NAME_ONLY_JOB_VARS:
            self._warn(
                f"run id derived from {self.run_id_source}={self.job_id!r}, which is a "
                "workload NAME, not a unique id. Two workloads with the same name at "
                "the same SHA would share this run directory and the second would be "
                "treated as a resume of the first. Set JOB_UUID, or set CLEFT_JOB_ID "
                "explicitly in the workload spec."
            )
        self.run_id = self._run_id()
        self.run_dir = self.out_root / self.tier / self.config["phase"] / self.run_id
        self.resumed = self.run_dir.exists()
        self.attempt = 0
        if self.resumed:
            self.attempt = self._verify_resume()

        # 6. Guard 3 - declared input hashes.
        self.inputs = self._verify_inputs()

        # 7. Create (or adopt), copy the config verbatim, write the records.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "log.txt"
        self._log_path.touch()
        if not self.resumed:
            shutil.copyfile(self.config_path, self.run_dir / "config.yaml")

        self._outputs: list[dict] = []
        self._finalized = False

        # 8. code/ as a worktree at the captured SHA.
        worktree_created, worktree_reason = self._create_worktree()

        self.env = self._build_env(worktree_created, worktree_reason)
        self._write_env_record()
        atomic_write_text(
            self.run_dir / "inputs.json",
            json.dumps({"inputs": self.inputs}, indent=2, sort_keys=True) + "\n",
        )

        if self.resumed:
            self.log(
                f"RESUME attempt {self.attempt} of run {self.run_id} "
                f"(job id from {self.run_id_source})"
            )
        else:
            self.log(
                f"run {self.run_id} tier={self.tier} sha={self.git.sha8} "
                f"id_source={self.run_id_source}"
            )
        for warning in self.warnings:
            self.log(f"WARNING {warning}")

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    @staticmethod
    def plan_run_dir(
        config_path: str | Path,
        out_root: str | Path,
        *,
        repo_root: str | Path | None = None,
        now: datetime | None = None,
    ) -> Path:
        """The directory a run with these arguments would claim. Creates nothing."""
        config = load_config(config_path)
        root = Path(repo_root) if repo_root else default_repo_root()
        git = gitinfo.inspect_repo(root)
        if git.state is gitinfo.GitState.UNAVAILABLE:
            raise GuardError(
                f"cannot plan a run directory: git did not answer for {root} "
                f"({git.failure.value if git.failure else 'unknown'}: {git.error})"
            )
        tier = "dev" if git.dirty else config["tier"]
        token, _ = job_token()
        suffix = token or _stamp(now or datetime.now(timezone.utc))
        run_id = f"{Path(config_path).stem}__{git.sha8}__{suffix}"
        return Path(out_root) / tier / config["phase"] / run_id

    def _run_id(self) -> str:
        if self.job_id:
            suffix = self.job_id
        else:
            # Laptop and dev runs have no job id. A timestamp cannot survive a
            # restart, so a run identified this way is never resumable and guard
            # 2 stays a hard abort for it.
            suffix = _stamp(self.now)
            self.run_id_source = "timestamp"
        return f"{self.config_path.stem}__{self.git.sha8}__{suffix}"

    # ------------------------------------------------------------------
    # guards
    # ------------------------------------------------------------------

    def _apply_git_guard(self, declared_tier: str) -> str:
        """Guard 1, over the three git states.

        Unknown is not the same as dirty and must not be treated as it. Dirty is
        a state git reported; unavailable is git declining to report anything.
        """
        if self.git.state is gitinfo.GitState.UNAVAILABLE:
            remedy = self.git.remedy
            raise GuardError(
                "cannot determine the code state: git did not answer.\n"
                f"  repository : {self.repo_root}\n"
                f"  problem    : {self.git.failure.value if self.git.failure else 'unknown'}\n"
                f"  git said   : {self.git.error}\n"
                + (f"\n  {remedy}\n" if remedy else "")
                + "\nThis is NOT a dirty tree. A dirty tree is a known state and only "
                "demotes the run to runs/dev/; an unanswerable git means nothing "
                "about this run's provenance can be recorded, so it stops at every "
                "tier."
            )

        if self.git.state is gitinfo.GitState.CLEAN:
            return declared_tier

        if declared_tier == "keeper":
            raise GuardError(
                "refusing a keeper run on a dirty tree: there are uncommitted or "
                "untracked changes in the work tree. Commit first, because a result "
                "whose code state is not described by its SHA is not citable.\n"
                "  (This is a real dirty tree, not a failed git probe -- git "
                "answered and reported changes.)"
            )
        return "dev"

    def _warn(self, message: str) -> None:
        """Record a provenance warning and make it impossible to miss."""
        self.warnings.append(message)
        banner = "=" * 78
        print(f"\n{banner}\nPROVENANCE WARNING\n  {message}\n{banner}\n", file=sys.stderr)

    def _check_pinned_image(self) -> None:
        """A keeper run outside the pinned image is a provenance hole.

        Not fatal: legitimate CPU-only keeper work such as building the manifest
        runs outside the image. But it happened on the very first cluster attempt
        -- workspace packages, numpy 1.24.4 against the pinned 1.26.4, torch and
        timm absent -- so it has to be loud and it has to be in env.json.
        """
        if self.tier != "keeper" or self._python["in_container"]:
            return
        self._warn(
            "keeper run OUTSIDE the pinned image (CLEFT_IN_CONTAINER is not set). "
            "Package versions come from whatever environment this is, not from "
            "docker/requirements.txt. Legitimate for CPU-only work; a provenance "
            "hole for anything trained. Recorded in env.json as "
            "keeper_outside_pinned_image."
        )

    def _check_python(self) -> dict:
        actual = "{}.{}.{}".format(*sys.version_info[:3])
        declared = DECLARED_PYTHON
        matches = declared is not None and actual == declared
        in_container = os.environ.get("CLEFT_IN_CONTAINER") == "1"
        if in_container and not matches:
            raise GuardError(
                f"python {actual} inside the image does not match the declared "
                f"{declared} in docker/requirements.txt. The build-image workflow "
                "measures this and fails if it drifts, so either the image was "
                "not built by that workflow or the declaration was edited by hand."
            )
        return {
            "version": actual,
            "version_full": sys.version,
            "executable": sys.executable,
            "declared": declared,
            "declaration_status": DECLARED_PYTHON_STATUS,
            "matches_declared": matches,
            "in_container": in_container,
        }

    def _verify_resume(self) -> int:
        """Guard 2, resume branch. Returns the attempt number.

        A directory that already exists is only ever adopted when the run id came
        from a stable job id and the code and config are identical. Anything else
        is a clobber.
        """
        if self.run_id_source == "timestamp":
            raise GuardError(
                f"run directory already exists, refusing to clobber: {self.run_dir}\n"
                "This id is timestamp-derived, so it cannot be a Run:AI restart."
            )

        env_path = self.run_dir / "env.json"
        if not env_path.is_file():
            raise GuardError(
                f"run directory {self.run_dir} exists but has no env.json. It is "
                "half-built or was created by something else; refusing to adopt it."
            )

        existing_config = self.run_dir / "config.yaml"
        if not existing_config.is_file():
            raise GuardError(
                f"run directory {self.run_dir} has no config.yaml; refusing to adopt it."
            )
        if existing_config.read_bytes() != self.config_path.read_bytes():
            raise GuardError(
                f"resume refused: {self.config_path} differs from the config.yaml "
                f"already in {self.run_dir}. A resumed job must run the same config; "
                "a changed config is a new run."
            )

        previous = json.loads(env_path.read_text(encoding="utf-8"))
        previous_sha = previous.get("git", {}).get("sha")
        if previous_sha != self.git.sha:
            raise GuardError(
                f"resume refused: {self.run_dir} was created at SHA {previous_sha} "
                f"but this process is at {self.git.sha}. Arms run at different code "
                "states are not comparable."
            )

        resumes = self._read_resumes()
        return len(resumes) + 1

    def _read_resumes(self) -> list[dict]:
        path = self.run_dir / "resumes.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("resumes", [])

    def _verify_inputs(self) -> list[dict]:
        resolved = []
        for entry in self.config["inputs"]:
            path = resolve_declared_path(entry["path"], self.repo_root)
            if not path.exists():
                raise GuardError(
                    f"declared input {entry['name']!r} does not exist: {path}"
                )
            actual = hash_path(path)
            if actual["rollup"] != entry["rollup_sha256"]:
                raise GuardError(
                    f"input hash mismatch for {entry['name']!r} at {path}\n"
                    f"  config declares {entry['rollup_sha256']}\n"
                    f"  directory hashes {actual['rollup']}\n"
                    f"  ({actual['file_count']} files, {actual['total_bytes']} bytes)\n"
                    "Data artifacts are immutable: create a new version, never edit one."
                )
            resolved.append(
                {
                    "name": entry["name"],
                    "path": str(path),
                    "declared_rollup": entry["rollup_sha256"],
                    "rollup": actual["rollup"],
                    "file_count": actual["file_count"],
                    "total_bytes": actual["total_bytes"],
                }
            )
        return resolved

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------

    def _create_worktree(self) -> tuple[bool, str | None]:
        code = self.run_dir / "code"
        if code.exists():
            return True, None
        # Only DIRTY can reach here: UNAVAILABLE aborted in the git guard, so
        # there is no "git failed" branch to confuse with dirtiness any more.
        if self.git.dirty or not self.git.sha:
            return False, "dirty tree: the SHA does not describe the code that ran"
        try:
            gitinfo.add_worktree(self.repo_root, self.git.sha, code)
        except Exception as exc:  # noqa: BLE001 - recorded, not silently dropped
            return False, f"worktree creation failed: {exc}"
        return True, None

    def _build_env(self, worktree_created: bool, worktree_reason: str | None) -> dict:
        return {
            "run_id": self.run_id,
            "run_id_source": self.run_id_source,
            "job_id": self.job_id,
            "attempt": self.attempt,
            "resumed": self.resumed,
            "tier": self.tier,
            "declared_tier": self.config["tier"],
            "phase": self.config["phase"],
            "seed": self.config["seed"],
            "utc_timestamp": self.now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git": {
                # The tri-state is the authoritative field. dirty_tree is kept
                # because readers and the plan document both use it, but it now
                # means only what it says: git answered, and reported changes.
                "state": self.git.state.value,
                "available": self.git.available,
                "sha": self.git.sha,
                "sha8": self.git.sha8,
                "dirty_tree": self.git.dirty,
                "branch": self.git.branch,
                "repo_root": str(self.repo_root),
                "worktree_created": worktree_created,
                "worktree_skipped_reason": worktree_reason,
            },
            "python": self._python,
            # A copy, because the package list is cached and shared. Embedding the
            # cached dict would let one run's env.json be mutated by another's.
            "packages": dict(_package_versions()),
            "cuda": _cuda_info(),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "image": {
                "digest": os.environ.get("CLEFT_IMAGE_DIGEST"),
                "base_digest": os.environ.get("CLEFT_BASE_IMAGE_DIGEST"),
                "in_container": self._python["in_container"],
            },
            # Which job-identity variables the scheduler actually set. Recorded
            # because the expected names were wrong once already: RUNAI_JOB_ID and
            # RUNAI_JOB_UUID do not exist on this cluster, JOB_UUID and jobName do.
            # An allowlist rather than a sweep of the environment, so no token or
            # credential can be captured by accident.
            "job_identity": {
                name: os.environ[name] for name in JOB_ID_ENV_VARS if name in os.environ
            },
            "keeper_outside_pinned_image": (
                self.tier == "keeper" and not self._python["in_container"]
            ),
            "provenance_warnings": list(self.warnings),
        }

    def _write_env_record(self) -> None:
        """First attempt writes env.json; resumes append to resumes.json.

        env.json stays the canonical first-attempt record. A resumed pod can land
        on a different node with a different GPU, so each attempt's environment is
        kept rather than overwritten.
        """
        payload = json.dumps(self.env, indent=2, sort_keys=True) + "\n"
        if not self.resumed:
            atomic_write_text(self.run_dir / "env.json", payload)
            return
        resumes = self._read_resumes()
        resumes.append({"attempt": self.attempt, "env": self.env})
        atomic_write_text(
            self.run_dir / "resumes.json",
            json.dumps({"resumes": resumes}, indent=2, sort_keys=True) + "\n",
        )

    # ------------------------------------------------------------------
    # writing outputs
    # ------------------------------------------------------------------

    def path(self, name: str, tier: str) -> Path:
        """Return a path inside the run directory, recording its sharing tier."""
        if tier not in OUTPUT_TIERS:
            raise ValueError(f"unknown output tier {tier!r}, expected one of {OUTPUT_TIERS}")
        if PATIENT_KEYED.search(name) and tier != "CLUSTER-ONLY":
            raise ValueError(
                f"{name!r} is patient-keyed and must be CLUSTER-ONLY: the clinical "
                "cohort never leaves EHU infrastructure."
            )
        if any(entry["name"] == name for entry in self._outputs):
            raise ValueError(f"output {name!r} already claimed by this run")
        self._outputs.append({"name": name, "tier": tier})
        return self.run_dir / name

    @contextmanager
    def atomic(self, name: str, tier: str):
        """Claim an output and yield a path to write atomically.

        The caller writes to the yielded temporary path; it is renamed into place
        only on clean exit. Use this for anything a paused pod could be halfway
        through -- checkpoints above all.
        """
        target = self.path(name, tier)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            yield tmp
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} [attempt {self.attempt}] {message}\n")

    def finalize(self) -> Path:
        """Write the output index. Idempotent."""
        if self._finalized:
            return self.run_dir

        merged: dict[str, dict] = {}
        for entry in self._read_previous_outputs():
            merged[entry["name"]] = entry
        for entry in self._outputs:
            path = self.run_dir / entry["name"]
            if path.is_file():
                entry["sha256"] = hash_file(path)
                entry["bytes"] = path.stat().st_size
            else:
                entry["sha256"] = None
                entry["bytes"] = None
                entry["missing"] = True
            entry["attempt"] = self.attempt
            merged[entry["name"]] = entry

        atomic_write_text(
            self.run_dir / "outputs.json",
            json.dumps(
                {"outputs": [merged[name] for name in sorted(merged)]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self.log("finalized")
        self._finalized = True
        return self.run_dir

    def _read_previous_outputs(self) -> list[dict]:
        path = self.run_dir / "outputs.json"
        if not self.resumed or not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("outputs", [])

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.log(f"FAILED {exc_type.__name__}: {exc}")
        self.finalize()
        return False
