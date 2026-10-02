"""Default locations of checkpoints, input data and outputs (independent of the working directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # sport_analysis/
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data" / "bouldering"
OUTPUT_DIR = DATA_DIR / "output"

DEFAULT_VIDEO = DATA_DIR / "VID-20250317-WA0000.mp4"
DEFAULT_IMAGE = DATA_DIR / "bouldering.png"
SAM3_CHECKPOINT = CHECKPOINT_DIR / "sam3_litetext_mobileclip_s0_ctx16.pt"
ONNX_DIR = CHECKPOINT_DIR / "onnx_sam3_litetext_s0"
