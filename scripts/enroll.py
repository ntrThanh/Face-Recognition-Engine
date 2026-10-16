"""Manage enrolled people.

    python scripts/enroll.py --name ntrthanh --dir assets/ntrthanh_face   # add / extend a person
    python scripts/enroll.py --name ntrthanh --video "path/to/registration.mp4"
    python scripts/enroll.py --list
    python scripts/enroll.py --remove ntrthanh
    python scripts/enroll.py --rebuild     # re-embed everything in the gallery (after a model change)

Enrollment images are copied to <database.gallery>/<name>/ so the DB can be rebuilt later.
From a video, the best frames are picked automatically (exactly one face, large, sharp,
not turned too far) spread across the whole clip, and saved there as JPEGs.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from face_rec import FaceRecognizer
from face_rec.alignment import align_face
from face_rec.config import load_config, resolve_path
from face_rec.quality import pitch_ratio, sharpness, yaw_ratio

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def embed_person(fr: FaceRecognizer, name: str, images: list[Path]) -> int:
    embeddings, sources = [], []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip {path.name}: cannot read")
            continue
        faces = fr.detect(img)
        if not faces:
            print(f"  skip {path.name}: no face found")
            continue
        if len(faces) > 1:
            print(f"  {path.name}: {len(faces)} faces, using the largest")
        face = max(faces, key=lambda f: f.width * f.height)
        fr.embed_faces(img, [face])
        embeddings.append(face.embedding)
        sources.append(str(path))
        print(f"  ok   {path.name}: score={face.score:.2f} size={int(face.width)}x{int(face.height)}")
    if embeddings:
        fr.db.add(name, embeddings, sources)
    return len(embeddings)


def select_video_frames(fr: FaceRecognizer, video: Path, args) -> list[dict]:
    """Score sampled frames and pick the best one in each of `max_frames` time segments."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates, rejected, index = [], {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if index % args.step == 0:
            faces = fr.detect(img)
            reason = None
            if len(faces) != 1:
                reason = "no face" if not faces else "several faces"
            else:
                f = faces[0]
                yaw = yaw_ratio(f.landmarks)
                if f.score < args.min_score:
                    reason = "low detection score"
                elif min(f.width, f.height) < args.min_size:
                    reason = "face too small"
                elif abs(yaw) > args.max_yaw:
                    reason = "head turned"
                elif not args.min_pitch <= pitch_ratio(f.landmarks) <= args.max_pitch:
                    reason = "head tilted up/down"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
            else:
                crop = align_face(img, f.landmarks, fr.recognizer.align_template)
                candidates.append({"index": index, "image": img, "face": f, "yaw": yaw,
                                   "pitch": pitch_ratio(f.landmarks), "sharp": sharpness(crop)})
        index += 1
    cap.release()

    sampled = (index + args.step - 1) // args.step
    print(f"  sampled {sampled} of {index or total} frames: {len(candidates)} usable, "
          + ", ".join(f"{n} {r}" for r, n in sorted(rejected.items())))
    if not candidates:
        return []

    # Drop blurry frames relative to this video (absolute sharpness depends on resolution).
    median_sharp = float(np.median([c["sharp"] for c in candidates]))
    candidates = [c for c in candidates if c["sharp"] >= 0.5 * median_sharp]

    # One frame per time segment -> varied poses/expressions, best quality within each.
    segments = np.array_split(np.arange(len(candidates)), min(args.max_frames, len(candidates)))
    picked = []
    for seg in segments:
        group = [candidates[i] for i in seg]
        picked.append(max(group, key=lambda c: c["face"].score * (1 - abs(c["yaw"]))
                          * min(c["sharp"] / median_sharp, 1.5)))
    return picked


def enroll_video(fr: FaceRecognizer, name: str, video: Path, dest: Path | None, args) -> int:
    print(f"{name}: scanning {video.name}")
    picked = select_video_frames(fr, video, args)
    if not picked:
        print("  no usable frames; try a brighter video looking at the camera")
        return 0

    for c in picked:
        fr.embed_faces(c["image"], [c["face"]])
    embs = np.stack([c["face"].embedding for c in picked])

    # Sanity check: the clip should be one person and match any existing enrollment.
    self_sim = embs @ embs.T
    print(f"  consistency within video: min cosine {self_sim.min():.2f}")
    if name in fr.db.names():
        sims = [fr.db.search(e, top_k=len(fr.db.names())) for e in embs]
        own = [next((m.similarity for m in s if m.name == name), 0.0) for s in sims]
        print(f"  similarity to existing '{name}' enrollment: mean {np.mean(own):.2f}, "
              f"min {np.min(own):.2f}")
        if np.mean(own) < 0.25:
            sys.exit("  aborted: this video does not look like the enrolled person "
                     "(use --remove first to replace them)")

    sources = []
    for c in picked:
        path = str(video)
        if dest is not None:
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / f"{video.stem.strip().replace(' ', '_')}_f{c['index']:05d}.jpg"
            cv2.imwrite(str(out), c["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            path = str(out)
        sources.append(path)
        print(f"  ok   frame {c['index']:4d}: score={c['face'].score:.2f} "
              f"size={int(c['face'].width)}x{int(c['face'].height)} yaw={c['yaw']:+.2f} "
              f"sharp={c['sharp']:.0f}")
    fr.db.add(name, embs, sources)
    return len(picked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--name", help="person name (ASCII recommended for on-screen labels)")
    p.add_argument("--dir", help="folder of images of that person")
    p.add_argument("--video", help="video of that person (frames are picked automatically)")
    p.add_argument("--max-frames", type=int, default=10, help="video: frames to enroll")
    p.add_argument("--step", type=int, default=3, help="video: analyse every Nth frame")
    p.add_argument("--min-score", type=float, default=0.6, help="video: min detection score")
    p.add_argument("--min-size", type=int, default=80, help="video: min face size in pixels")
    p.add_argument("--max-yaw", type=float, default=0.35,
                   help="video: max head turn (nose offset / eye distance; 0 = frontal)")
    p.add_argument("--min-pitch", type=float, default=0.33,
                   help="video: nose height between eyes (0) and mouth (1); lower = chin up")
    p.add_argument("--max-pitch", type=float, default=0.70, help="video: higher = looking down")
    p.add_argument("--no-copy", action="store_true", help="don't save images into the gallery")
    p.add_argument("--rebuild", action="store_true", help="wipe the DB and re-embed the gallery")
    p.add_argument("--list", action="store_true")
    p.add_argument("--remove", metavar="NAME")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    gallery = resolve_path(cfg["database"].get("gallery", "data/gallery"))
    fr = FaceRecognizer(cfg, reset_db=args.rebuild)

    try:
        if args.list:
            for name in fr.db.names():
                print(f"{name}: {fr.db.count(name)} embeddings")
            print(f"total: {len(fr.db.names())} people, {fr.db.count()} embeddings "
                  f"(model {fr.db.model_id})")

        elif args.remove:
            if args.remove not in fr.db.names():
                sys.exit(f"'{args.remove}' is not enrolled")
            fr.db.remove(args.remove)
            print(f"removed '{args.remove}' from the database "
                  f"(gallery images in {gallery / args.remove} were kept)")

        elif args.rebuild:
            people = sorted(d for d in gallery.iterdir() if d.is_dir()) if gallery.exists() else []
            for person in people:
                print(f"{person.name}:")
                embed_person(fr, person.name, list_images(person))
            print(f"rebuilt: {len(fr.db.names())} people, {fr.db.count()} embeddings")

        elif args.name and args.video:
            video = Path(args.video)
            if not video.exists():
                sys.exit(f"video not found: {video}")
            dest = None if args.no_copy else gallery / args.name
            n = enroll_video(fr, args.name, video, dest, args)
            print(f"enrolled {n} frames; '{args.name}' now has "
                  f"{fr.db.count(args.name)} embeddings")

        elif args.name and args.dir:
            src = Path(args.dir)
            images = list_images(src)
            if not images:
                sys.exit(f"no images in {src}")
            if not args.no_copy:
                dest = gallery / args.name
                dest.mkdir(parents=True, exist_ok=True)
                if src.resolve() != dest.resolve():
                    copied = []
                    for img in images:
                        target = dest / img.name
                        if not target.exists():
                            shutil.copy2(img, target)
                        copied.append(target)
                    images = copied
            print(f"{args.name}:")
            n = embed_person(fr, args.name, images)
            print(f"enrolled {n}/{len(images)} images; '{args.name}' now has "
                  f"{fr.db.count(args.name)} embeddings")

        else:
            p.print_help()
    finally:
        fr.close()


if __name__ == "__main__":
    main()
