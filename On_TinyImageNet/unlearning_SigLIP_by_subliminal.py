"""Post-release personalized SigLIP unlearning on TinyImageNet.

Default research mode saves outputs/unlearning_SigLIP_by_subliminal_20261006.txt
beside this script, on GPU 0. Audit artifacts are not publicly releasable DP data.
"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from On_CIFAR100 import unlearning_SigLIP_by_subliminal as siglip_common
from On_TinyImageNet import unlearning_clip_by_subliminal as tiny_common


GPU = 0  # Set to a visible GPU index, or -1 for CPU.


def parse_args():
    parser = tiny_common.build_parser(
        {
            "description": (
                "Portable carrier-based unlearning for post-release "
                "personalized SigLIP on TinyImageNet."
            ),
            "model_name": "ViT-B-16-SigLIP",
            "pretrained": "webli",
            "gpu": GPU,
            "unl_size_result": (
                "On_TinyImageNet/"
                "unlearning_SigLIP_by_subliminal_104_for_unl_size.txt"
            ),
            "dp_result": (
                "On_TinyImageNet/"
                "unlearning_SigLIP_by_subliminal_104_for_dp.txt"
            ),
            "comparison_result": (
                "On_TinyImageNet/"
                "unlearning_SigLIP_by_subliminal_104_for_unl_size_comparison.txt"
            ),
        }
    )
    parser.add_argument(
        "--encoder_microbatch_size",
        "--encoder-microbatch-size",
        type=int,
        default=512,
        help="Maximum frozen-encoder batch size; use 0 to disable chunking.",
    )
    return parser.parse_args()


def siglip_backbone_factory(visual, args):
    return siglip_common.FrozenSigLIPBackbone(
        visual, args.encoder_microbatch_size
    )


def main():
    args = parse_args()
    if args.encoder_microbatch_size < 0:
        raise ValueError("--encoder_microbatch_size must be nonnegative.")
    tiny_common.run_experiment(
        args,
        siglip_backbone_factory,
        siglip_common.initialize_adapter,
    )


if __name__ == "__main__":
    main()
