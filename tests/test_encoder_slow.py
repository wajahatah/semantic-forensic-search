"""Real encoders. Downloads model weights (~0.8 GB each) on first run. Run with: pytest -m slow"""
import numpy as np
import pytest

from sfs.embed.encoder import Encoder

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", ["siglip", "siglip2"])
def test_real_encoder_shapes_and_text_image_alignment(name):
    enc = Encoder(name)
    red = np.zeros((224, 224, 3), np.uint8)
    red[..., 0] = 255
    blue = np.zeros((224, 224, 3), np.uint8)
    blue[..., 2] = 255

    img = enc.embed_images([red, blue])
    txt = enc.embed_texts(["a plain red image", "a plain blue image"])
    assert img.shape == (2, enc.dim) == (2, 768) and txt.shape == (2, 768)
    assert np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-3)

    sim = img @ txt.T  # each colour should match its own caption best
    assert sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0]
