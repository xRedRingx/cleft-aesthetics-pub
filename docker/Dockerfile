# Environment only. This image contains NO application code, ever.
#
# The previous pipeline's Dockerfile ended with `COPY . .`, so every code edit
# rebuilt gigabytes, so commits were avoided, so every run ever produced carried
# dirty_tree: true -- and code drift could never be separated from
# non-determinism. tests/test_workflow_hygiene.py now fails the build if any
# COPY source is anything other than requirements.txt.
#
# Code reaches the cluster by `git pull` on the NFS mount. Rebuilding this image
# is a separate manual workflow, triggered only when requirements.txt changes.
#
# Base: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
#   torch 2.8.0, CUDA 12.8, cuDNN 9. Includes sm_120 for the RTX PRO 6000
#   Blackwell and stays backward compatible with sm_86 (A40), so one image
#   serves both node pools.
# Pinned by digest, not by tag: a tag can be repointed upstream without notice,
# and then two runs at the same git SHA would no longer be the same experiment.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

# git is required at runtime, not just for convenience: the provenance module
# resolves the HEAD SHA and checks out code/ as a worktree beside every result.
# libgl1 and libglib2.0-0 are opencv's runtime shared libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# The version guard is deliberate. numpy 2 does not raise on import against the
# 1.x-compiled wheels below, it produces wrong numbers, so the failure has to be
# forced here where it is loud.
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -c "import numpy, torch; \
assert numpy.__version__.startswith('1.'), 'numpy upgraded to ' + numpy.__version__; \
assert torch.__version__.startswith('2.8'), 'torch changed to ' + torch.__version__; \
print('deps OK: torch', torch.__version__, 'numpy', numpy.__version__)"

# Measure the interpreter and enforce the declaration IN THIS LAYER.
#
# Doing it here rather than in CI after the fact means an image whose python
# disagrees with docker/requirements.txt cannot be BUILT, not merely cannot be
# pushed. It is also the only place the measurement is free: checking afterwards
# required `load: true`, which exported a multi-GB torch image and re-imported it
# on a runner with ~14 GB free, and died with "no space left on device".
#
# The declaration is read from the copied requirements.txt, so there is exactly
# one source of truth and it travels with the image.
#
# Written with no single quotes inside a single-quoted shell string, and with
# conditional expressions rather than `if` blocks, so nothing needs escaping.
RUN python -c 'import re, sys, pathlib; t = pathlib.Path("/tmp/requirements.txt").read_text(); m = re.search(r"^#\s*python:\s*(\d+\.\d+\.\d+)\s*$", t, re.M); d = m.group(1) if m else None; v = "%d.%d.%d" % sys.version_info[:3]; print("python measured", v, "/ declared", d); sys.exit(0 if d == v else "FATAL: python drift. This image ships " + v + " but docker/requirements.txt declares " + str(d) + ".\nSet the line to:  # python: " + v + "\nThen rebuild. env.json records this on every run, so a wrong declaration would silently mislabel every result.")' \
 && python -VV

# Marks the runtime as the container rather than a laptop. The provenance module
# turns a python version mismatch into a hard abort when this is set, and into a
# recorded warning when it is not.
ENV CLEFT_IN_CONTAINER=1

# Recorded into env.json on every run. The image's own digest is not knowable at
# build time; build-image.yml prints it, and the Run:AI workload passes it in as
# CLEFT_IMAGE_DIGEST.
ENV CLEFT_BASE_IMAGE_DIGEST=sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

# The NFS project path on the cluster. The repo is checked out here by git, not
# copied in by docker.
WORKDIR /home/user/codex/cleft-aesthetics
ENV PYTHONPATH=/home/user/codex/cleft-aesthetics/src

# No default command runs application code. Jobs are launched through
# scripts/entrypoint.sh from the repo, which pins the SHA it runs.
CMD ["python", "-c", "print('cleft-aesthetics environment image; launch via scripts/entrypoint.sh')"]
