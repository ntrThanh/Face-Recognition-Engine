import numpy as np

from face_rec.alignment import TEMPLATES, align_face


def test_template_landmarks_map_to_identity_crop():
    dst, size = TEMPLATES["arcface_112"]
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    out = align_face(img, dst.copy(), "arcface_112")
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2
