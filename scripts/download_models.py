"""Download an InsightFace model pack and keep only detection + recognition models.

    python scripts/download_models.py --pack buffalo_s
"""

import argparse

# TODO(phase 0): use insightface.utils.storage.ensure_available("models", pack),
# copy det_*.onnx -> models/detection/, w600k_*.onnx -> models/recognition/.


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="buffalo_s")
    args = p.parse_args()
    raise NotImplementedError(f"download {args.pack}")


if __name__ == "__main__":
    main()
