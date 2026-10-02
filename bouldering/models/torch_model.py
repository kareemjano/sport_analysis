"""PyTorch SAM3 / EfficientSAM3 video models (detector + SAM3 tracker)."""
from sam3.model_builder import build_efficientsam3_video_model, build_sam3_video_model

from ..paths import SAM3_CHECKPOINT


def build_model(model="sam3", checkpoint=SAM3_CHECKPOINT, text_encoder="MobileCLIP-S0", context_length=16,
                backbone_type="tinyvit", model_name="11m", tracker_checkpoint=SAM3_CHECKPOINT, device="cpu"):
    """SAM3 (ViT backbone) or EfficientSAM3 (student backbone) detector + SAM3 tracker.

    model: "sam3" (LiteText checkpoint) or "efficientsam3" (backbone_type / model_name).
    tracker_checkpoint: EfficientSAM3 only; SAM3 video checkpoint providing the tracker
        weights that EfficientSAM3 image checkpoints don't have.
    """
    if model == "sam3":
        net = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            text_encoder_type=text_encoder,
            text_encoder_context_length=context_length,
            device=device,
        )
    elif model == "efficientsam3":
        net = build_efficientsam3_video_model(
            checkpoint_path=str(checkpoint),
            backbone_type=backbone_type,
            model_name=model_name,
            text_encoder_type=text_encoder,
            text_encoder_context_length=context_length,
            tracker_checkpoint_path=str(tracker_checkpoint),
            device=device,
        )
    else:
        raise ValueError(f"unknown model {model!r}")
    return net.eval()
