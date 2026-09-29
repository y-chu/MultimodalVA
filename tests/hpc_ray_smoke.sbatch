#!/bin/bash
# Ray backend smoke test, as a SLURM job.
#
#   sbatch tests/hpc_ray_smoke.sbatch
#   sbatch --export=ALL,MMVA_QUICK=1 tests/hpc_ray_smoke.sbatch   # skip the text checks
#
# Edit the four SBATCH lines marked EDIT for your cluster, then submit from the
# package root. The log lands in ray_smoke_<jobid>.log next to where you submit.
#
# Why two GPUs: the script asks Ray for 0.5 GPU per trial with two trials at a
# time, and one of its checks is that trials really did overlap. On a single GPU
# that check fails for a reason that is not a bug, so request two — or run with
# MMVA_QUICK=1 on one GPU and accept the parallelism check reporting a genuine
# serial run.

#SBATCH --job-name=mmva-ray-smoke
#SBATCH --output=ray_smoke_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2          # EDIT: some clusters use --gres=gpu:2
##SBATCH --account=PAS0000         # EDIT: uncomment and set your account
##SBATCH --partition=gpu           # EDIT: uncomment and set your partition
##SBATCH --mail-type=END,FAIL      # EDIT: uncomment for mail

set -uo pipefail

echo "host          : $(hostname)"
echo "job id        : ${SLURM_JOB_ID:-none}"
echo "cpus per task : ${SLURM_CPUS_PER_TASK:-unset}"
echo "gpus          : ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-unset}}"
echo "started       : $(date -Is)"
echo

# --- EDIT: however this cluster activates the environment -------------------
# module load cuda/12.4
# source ~/venvs/multimodalva/bin/activate
# conda activate multimodalva

# One thread per process. Ray runs several trials at once and each would
# otherwise start as many BLAS threads as the node has cores, which oversubscribes
# it badly; the per-trial CPU budget is what num_cpus_per_trial is for.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Ray puts its session state here, and inside it a Unix domain socket whose full
# path must fit in 107 bytes — a hard limit in the kernel, not something Ray can
# work around. Ray itself appends about 65 characters
# (ray/session_<timestamp>_<pid>/sockets/plasma_store), so this directory has to
# stay under roughly 40. A $TMPDIR on project storage is typically far longer
# than that: on one cluster it was /gpfs/projects/<acct>/<user>/tmp, and every
# Ray start failed with "AF_UNIX path length cannot exceed 107 bytes". So prefer
# a short node-local path, and say so plainly if none is short enough.
for _candidate in "${SLURM_TMPDIR:-}" "/tmp" "${TMPDIR:-}"; do
    [ -n "$_candidate" ] || continue
    _try="$_candidate/ray_${SLURM_JOB_ID:-$$}"
    if [ ${#_try} -le 40 ] && mkdir -p "$_try" 2>/dev/null; then
        export RAY_TMPDIR="$_try"
        break
    fi
done
if [ -z "${RAY_TMPDIR:-}" ]; then
    echo "ERROR: no directory short enough for Ray's socket path (needs <= ~40" >&2
    echo "  characters). Set RAY_TMPDIR to a short node-local path and resubmit." >&2
    exit 1
fi
echo "ray tmpdir    : $RAY_TMPDIR (${#RAY_TMPDIR} chars)"

# The package unsets RAY_AIR_LOCAL_CACHE_DIR around its own tuner.fit() and
# forces RAY_AIR_NEW_OUTPUT=0; nothing here needs to set either.

echo "== python =="
python -c "import sys; print(sys.executable); print(sys.version)"
echo

echo "== is the [ray] extra installed? =="
python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("ray") is None:
    print("ray is NOT installed. Ray lives in an optional extra:")
    print("    pip install 'multimodalva[ray]'")
    sys.exit(1)
import ray
print("ray", ray.__version__)
PY
if [ $? -ne 0 ]; then
    echo "Aborting: install the [ray] extra and resubmit."
    exit 1
fi
echo

QUICK_FLAG=""
if [ "${MMVA_QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
    echo "MMVA_QUICK=1 — skipping the text checks"
fi

# Keep the run directories inside the job's scratch so a failure can be inspected.
OUTDIR="${SLURM_SUBMIT_DIR:-$PWD}/ray_smoke_out_${SLURM_JOB_ID:-local}"

echo "== smoke test =="
python tests/hpc_ray_smoke.py --outdir "$OUTDIR" $QUICK_FLAG
STATUS=$?

echo
echo "finished      : $(date -Is)"
echo "exit status   : $STATUS"
echo "run dirs      : $OUTDIR"
rm -rf "$RAY_TMPDIR"
exit $STATUS
