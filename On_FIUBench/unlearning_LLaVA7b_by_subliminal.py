"""Run FIUBench 7B Scheme-B unlearning with matched-projector SMFA.

Use --capacity-only and --capacity-checkpoint-dir for a fixed-M0-only sweep.
Use --rq1-sweep for 1/2/3/4/5/6 identity deletion with three independent M0 seeds;
the main run compares SMFA with the refusal/unlikelihood teacher and reports
all finite-epsilon carrier releases across independent Gaussian-noise repeats.
"""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from On_FIUBench import fiubench_llava_scheme_b_core as experiment


experiment.DEFAULT_MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
experiment.DEFAULT_OUTPUT_DIR = (
    "On_FIUBench/outputs/unlearning_LLaVA7b_smfa_dp_centered_v3_20261005"
)
experiment.DEFAULT_SUMMARY_PATH = (
    "On_FIUBench/unlearning_LLaVA7b_smfa_dp_centered_v3_20261005.txt"
)
experiment.DEFAULT_ADAPTER_RANK = 16
experiment.DEFAULT_EXPOSURE_STEPS = 400
experiment.DEFAULT_TEACHER_STEPS = 100
experiment.DEFAULT_TEACHER_METHOD = "smfa"
experiment.DEFAULT_RUN_TEACHER_COMPARISONS = True
experiment.DEFAULT_SMFA_RETAIN_STEPS = 100
experiment.DEFAULT_SMFA_RETAIN_LR = 5e-4
experiment.DEFAULT_SMFA_SCULPT_K = 5.0
experiment.DEFAULT_CARRIER_STEPS = 100
experiment.DEFAULT_DP_CARRIER_STEPS = 120
experiment.DEFAULT_FUNCTIONAL_INFLUENCE_SCALE = 0.08
experiment.DEFAULT_FUNCTIONAL_CLIP_NORM = 6.3
experiment.DEFAULT_FUNCTIONAL_CARRIER_COUNT = 32
experiment.DEFAULT_FUNCTIONAL_CONTRIBUTION_UNIT = "identity"
experiment.DEFAULT_FUNCTIONAL_LOCAL_STEPS = 20
experiment.DEFAULT_FUNCTIONAL_LOCAL_LR = 5e-4
experiment.DEFAULT_FUNCTIONAL_LOCAL_DELTA_SCALE = 2.8
experiment.DEFAULT_FUNCTIONAL_LOCAL_METHOD = "smfa"
experiment.DEFAULT_FUNCTIONAL_SMFA_AGGREGATE_BEFORE_SCULPT = True
experiment.DEFAULT_FUNCTIONAL_CENTER_ON_PUBLIC_RETAIN = True
experiment.DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_BLEND = 0.5
experiment.DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_REPAIR_STEPS = 0
experiment.DEFAULT_FUNCTIONAL_STUDENT_SOLVER = "ridge"
experiment.DEFAULT_FUNCTIONAL_RIDGE = 1e-6
experiment.DEFAULT_FUNCTIONAL_WHITEN = True
experiment.DEFAULT_FUNCTIONAL_PROJECTION_DIM = 6144
experiment.DEFAULT_FUNCTIONAL_PROJECTION_RECORDS = 24
experiment.DEFAULT_FUNCTIONAL_PROJECTION_MATCHED_LOCAL = True
experiment.DEFAULT_DP_EPSILONS = [
    "inf", "8192", "4096", "2048", "1024", "512"
]
experiment.DEFAULT_DP_NOISE_REPEATS = 5
experiment.DEFAULT_EVAL_QA_PER_IDENTITY = 4
experiment.DEFAULT_REUSE_BASE_CHECKPOINTS = True


if __name__ == "__main__":
    from rq1_experiments import dispatch_rq1_sweep
    if not dispatch_rq1_sweep(__file__, "llava"):
        experiment.main()
