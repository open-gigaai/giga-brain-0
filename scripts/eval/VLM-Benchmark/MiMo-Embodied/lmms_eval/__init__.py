import os
from pathlib import Path


os.environ.setdefault(
    "MIMO_DATA_ROOT",
    str(Path(__file__).resolve().parents[1] / "datasets" / "public_datasets" / "VLM"),
)
