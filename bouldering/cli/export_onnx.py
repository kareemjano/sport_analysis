"""Export the SAM3 LiteText video model (detector + tracker) to ONNX graphs for `--backend onnx`.

Example:
    python -m bouldering export-onnx --output-dir checkpoints/onnx_sam3_litetext_s0
"""
from ..models.onnx_models import GRAPHS
from ..paths import DEFAULT_VIDEO, ONNX_DIR
from .common import add_model_args


def add_arguments(parser):
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="first frame is used as the reference input")
    parser.add_argument("--prompt", default="climbing hold", help="reference text prompt for the checks")
    parser.add_argument("--output-dir", default=str(ONNX_DIR))
    parser.add_argument("--only", nargs="+", choices=GRAPHS, help="export only these graphs")
    parser.add_argument("--attn-chunk", type=int, default=432,
                        help="query chunk size of the tracker memory attention (divides 5184); lower = less memory")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip comparing ONNX Runtime outputs against PyTorch")
    add_model_args(parser)


def run(args):
    from ..models.onnx_export import export

    if 5184 % args.attn_chunk != 0:
        raise SystemExit("--attn-chunk must divide 72*72=5184")
    export(args)
