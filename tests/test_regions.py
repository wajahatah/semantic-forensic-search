import numpy as np

from sfs.data.regions import (box_side, context_region, crop, full_region, lookup_union,
                              tight_region, union)

W, H = 1920, 1072


def _inside_and_square(r):
    x1, y1, x2, y2 = r
    return 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H and (x2 - x1) == (y2 - y1)


def test_tight_enforces_min_side_for_small_person():
    r = tight_region((1171, 296, 1277, 538), W, H)  # 106 x 242 person from the MEVA example
    assert _inside_and_square(r)
    assert r[2] - r[0] == round(242 * 1.5)  # padded height dominates, above 224 min


def test_tight_min_side_for_tiny_box():
    r = tight_region((500, 500, 510, 520), W, H)
    assert r[2] - r[0] == 224 and _inside_and_square(r)


def test_context_is_larger_than_tight_and_contains_it():
    box = (1171, 296, 1277, 538)
    t, c = tight_region(box, W, H), context_region(box, W, H)
    assert c[2] - c[0] > t[2] - t[0]
    assert c[0] <= t[0] and c[1] <= t[1] and c[2] >= t[2] and c[3] >= t[3]


def test_regions_shift_inside_at_image_edges():
    for box in [(0, 0, 50, 80), (1900, 1050, 1920, 1072), (0, 1000, 30, 1072)]:
        assert _inside_and_square(tight_region(box, W, H))
        assert _inside_and_square(context_region(box, W, H))


def test_context_capped_by_image_height():
    r = context_region((800, 100, 1100, 1000), W, H)  # 900 tall * 3 > image
    assert r[3] - r[1] == H and _inside_and_square(r)


def test_lookup_union_exact_nearest_and_missing():
    tracks = {0: {10: (0, 0, 10, 10)}, 1: {14: (20, 20, 30, 30)}}
    assert lookup_union(tracks, (0, 1), 10, window=5) == (0, 0, 30, 30)  # actor 1 via nearest
    assert lookup_union(tracks, (0,), 20, window=5) is None
    assert lookup_union(tracks, (9,), 10) is None


def test_union_and_box_side():
    assert union([(5, 5, 10, 10), (0, 7, 8, 20)]) == (0, 5, 10, 20)
    assert box_side((0, 0, 30, 50)) == 50 and box_side(None) == 0


def test_crop_shape():
    frame = np.zeros((H, W, 3), np.uint8)
    assert crop(frame, (10, 20, 234, 244)).shape == (224, 224, 3)
    assert crop(frame, full_region(W, H)).shape == (H, W, 3)


def test_regions_always_square_and_inside_random_boxes():
    """Regression: per-edge rounding once produced 364x363 'squares'."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x1, y1 = int(rng.integers(0, W - 2)), int(rng.integers(0, H - 2))
        box = (x1, y1, int(rng.integers(x1 + 1, W)), int(rng.integers(y1 + 1, H)))
        assert _inside_and_square(tight_region(box, W, H))
        assert _inside_and_square(context_region(box, W, H))
