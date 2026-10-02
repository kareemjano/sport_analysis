"""Segment climbing holds in a single image with a text prompt (SAM3 / EfficientSAM3 image model).

Examples:
    python -m bouldering segment
    python -m bouldering segment --model efficientsam3 --checkpoint checkpoints/efficientsam3_repvit.pt \\
        --backbone-type repvit --model-name m1.1 --text-encoder MobileCLIP-S0 --context-length 16
"""
import cv2
import numpy as np

from ..paths import CHECKPOINT_DIR, DEFAULT_IMAGE
from .common import add_model_args, resolve_output


def add_arguments(parser):
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--prompt", default="climbing hold")
    parser.add_argument("--output", default=None,
                        help="default: data/bouldering/output/segment_<model variant>[_p-<prompt>].png")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45, help="mask overlay opacity")
    add_model_args(parser)
    parser.set_defaults(checkpoint=str(CHECKPOINT_DIR / "sam3_litetext_mobileclip_s1_ctx32.pt"),
                        text_encoder="MobileCLIP-S1", context_length=32)


def run(args):
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_efficientsam3_image_model, build_sam3_image_model

    from ..common.visualization import draw_masks

    resolve_output(args, "segment", ext=".png")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = dict(checkpoint_path=args.checkpoint, text_encoder_type=args.text_encoder,
                  text_encoder_context_length=args.context_length, load_from_HF=False, device=device)
    if args.model == "sam3":
        model = build_sam3_image_model(**common)
    else:
        model = build_efficientsam3_image_model(backbone_type=args.backbone_type, model_name=args.model_name, **common)

    processor = Sam3Processor(model, device=device, confidence_threshold=args.score_thresh)
    image = Image.open(args.image).convert("RGB")
    state = processor.set_text_prompt(args.prompt, processor.set_image(image))
    masks = state["masks"][:, 0].cpu().numpy()
    scores = state["scores"].cpu().numpy()
    print(f"Found {len(masks)} masks for {args.prompt!r} (scores {scores.min():.2f}-{scores.max():.2f})"
          if len(masks) else f"Found no masks for {args.prompt!r}")

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis = draw_masks(frame, np.arange(1, len(masks) + 1), masks, alpha=args.alpha,
                     header=f"{len(masks)} x {args.prompt!r}")
    cv2.imwrite(args.output, vis)
    print(f"Wrote {args.output}")
