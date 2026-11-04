import io

import numpy as np
import pytest
from PIL import Image

from weathercam.data.imagecodec import decode_rgb, load_rgb, save_png


def encode(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def noise_png(seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    return encode(Image.fromarray(pixels), "PNG")


def test_decode_png_values():
    pixels = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [255, 0, 128]]], dtype=np.uint8)
    out = decode_rgb(encode(Image.fromarray(pixels), "PNG"))
    assert out.shape == (2, 2, 3)
    expected = pixels.astype(np.float32) / np.float32(255.0)
    assert np.array_equal(out, expected)


def test_decode_dtype_and_range():
    out = decode_rgb(noise_png())
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_decode_grayscale_to_three_channels():
    out = decode_rgb(encode(Image.new("L", (3, 2), 100), "PNG"))
    assert out.shape == (2, 3, 3)
    assert np.all(out == np.float32(100) / np.float32(255.0))


def test_decode_rgba_drops_alpha():
    out = decode_rgb(encode(Image.new("RGBA", (2, 2), (10, 20, 30, 128)), "PNG"))
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[0, 0], np.array([10, 20, 30], dtype=np.float32) / np.float32(255.0))


def test_decode_palette_mode():
    image = Image.new("P", (2, 2))
    image.putpalette([255, 0, 0, 0, 255, 0])
    out = decode_rgb(encode(image, "PNG"))
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_decode_jpeg_shape():
    out = decode_rgb(encode(Image.new("RGB", (8, 4), (200, 100, 50)), "JPEG"))
    assert out.shape == (4, 8, 3)
    assert out.dtype == np.float32


def test_decode_ignores_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (4, 2), (200, 100, 50)).save(buffer, "JPEG", exif=exif)
    out = decode_rgb(buffer.getvalue())
    assert out.shape == (2, 4, 3)


def test_decode_empty_bytes_raises():
    with pytest.raises(ValueError):
        decode_rgb(b"")


def test_decode_garbage_raises():
    with pytest.raises(ValueError):
        decode_rgb(b"this is not an image at all")


@pytest.mark.parametrize("keep", [20, 60])
def test_decode_truncated_png_raises(keep):
    with pytest.raises(ValueError):
        decode_rgb(noise_png()[:keep])


@pytest.mark.parametrize("data", ["text", 5, None])
def test_decode_non_bytes_raises(data):
    with pytest.raises(ValueError):
        decode_rgb(data)


def test_decode_accepts_bytearray():
    data = noise_png()
    assert np.array_equal(decode_rgb(bytearray(data)), decode_rgb(data))


def test_load_rgb_matches_decode(tmp_path):
    data = noise_png()
    path = tmp_path / "noise.png"
    path.write_bytes(data)
    expected = decode_rgb(data)
    assert np.array_equal(load_rgb(path), expected)
    assert np.array_equal(load_rgb(str(path)), expected)


def test_load_rgb_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "missing.png")


def test_save_png_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    ints = rng.integers(0, 256, (5, 7, 3))
    image = ints.astype(np.float32) / np.float32(255.0)
    path = tmp_path / "round.png"
    save_png(path, image)
    assert np.array_equal(load_rgb(path), image)


def test_save_png_extremes(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[0, 0] = 1.0
    path = tmp_path / "extremes.png"
    save_png(path, image)
    with Image.open(path) as saved:
        pixels = np.asarray(saved)
    assert tuple(pixels[0, 0]) == (255, 255, 255)
    assert tuple(pixels[1, 1]) == (0, 0, 0)


def test_save_png_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "out.png"
    save_png(path, np.zeros((2, 2, 3), dtype=np.float32))
    assert path.is_file()


def test_save_png_rejects_other_suffix(tmp_path):
    path = tmp_path / "out.jpg"
    with pytest.raises(ValueError):
        save_png(path, np.zeros((2, 2, 3), dtype=np.float32))
    assert not path.exists()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 1.5, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_save_png_invalid_image_raises(tmp_path, image):
    path = tmp_path / "bad.png"
    with pytest.raises(ValueError):
        save_png(path, image)
    assert not path.exists()


def test_save_png_float64_input(tmp_path):
    image = np.full((2, 2, 3), 0.4, dtype=np.float64)
    path = tmp_path / "f64.png"
    save_png(path, image)
    with Image.open(path) as saved:
        assert tuple(np.asarray(saved)[0, 0]) == (102, 102, 102)


def test_save_png_input_not_modified(tmp_path):
    image = np.full((2, 2, 3), 0.25, dtype=np.float32)
    before = image.copy()
    save_png(tmp_path / "keep.png", image)
    assert np.array_equal(image, before)
