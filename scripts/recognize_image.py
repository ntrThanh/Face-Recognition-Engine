"""Recognize faces in an image.

    python scripts/recognize_image.py assets/images/test.jpg --out outputs/test.jpg
"""

import argparse

import cv2

from face_rec import FaceRecognizer
from face_rec.utils.draw import draw_results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", help="save annotated image here")
    args = p.parse_args()

    fr = FaceRecognizer.from_config(args.config)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    results = fr.recognize(img)
    for r in results:
        print(f"{r.name:<20} sim={r.similarity:.3f} bbox={r.face.bbox.astype(int).tolist()}")
    print("timing (ms):", {k: round(v, 1) for k, v in fr.timer.summary().items()})
    if args.out:
        cv2.imwrite(args.out, draw_results(img, results))


if __name__ == "__main__":
    main()
