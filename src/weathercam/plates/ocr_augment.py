import numpy as np

from weathercam.data.augment import _check_image, add_fog, add_haze, add_rain, add_snow

BRIGHTNESS = (0.6, 1.4)
CONTRAST = (0.6, 1.4)
GAMMA = (0.7, 1.4)
SATURATION = (0.5, 1.5)
CHANNEL_GAIN = (0.92, 1.08)
BLUR_SIGMA = (0.5, 1.5)
NOISE_SIGMA = (0.0, 0.04)
WEATHER_SEVERITY = (0.05, 0.4)
PROBABILITY = {
    "weather": 0.3,
    "brightness": 0.8,
    "contrast": 0.8,
    "gamma": 0.5,
    "saturation": 0.5,
    "gain": 0.5,
    "gray": 0.05,
    "blur": 0.3,
    "noise": 0.3,
}
_WEATHER = (add_fog, add_haze, add_rain, add_snow)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def settings():
    return {
        "probability": dict(PROBABILITY),
        "brightness": list(BRIGHTNESS),
        "contrast": list(CONTRAST),
        "gamma": list(GAMMA),
        "saturation": list(SATURATION),
        "channel_gain": list(CHANNEL_GAIN),
        "blur_sigma": list(BLUR_SIGMA),
        "noise_sigma": list(NOISE_SIGMA),
        "weather_severity": list(WEATHER_SEVERITY),
        "weather_effects": [effect.__name__ for effect in _WEATHER],
    }


def _clip(image):
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _draw(rng, bounds):
    return float(rng.uniform(*bounds))


def _gray(image):
    return (image @ _LUMA)[:, :, None]


def _gaussian_blur(image, sigma):
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    out = image
    for axis in (0, 1):
        pad = [(0, 0)] * 3
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        size = out.shape[axis]
        acc = np.zeros_like(out)
        for k, weight in enumerate(kernel):
            acc += weight * (padded[k : k + size] if axis == 0 else padded[:, k : k + size])
        out = acc
    return out


def augment_crop(image, rng):
    out = _check_image(image)
    if rng.random() < PROBABILITY["weather"]:
        effect = _WEATHER[int(rng.integers(len(_WEATHER)))]
        out = effect(out, _draw(rng, WEATHER_SEVERITY), rng)
    if rng.random() < PROBABILITY["brightness"]:
        out = _clip(out * _draw(rng, BRIGHTNESS))
    if rng.random() < PROBABILITY["contrast"]:
        mean = float(out.mean())
        out = _clip((out - mean) * _draw(rng, CONTRAST) + mean)
    if rng.random() < PROBABILITY["gamma"]:
        out = _clip(out ** _draw(rng, GAMMA))
    if rng.random() < PROBABILITY["saturation"]:
        gray = _gray(out)
        out = _clip(gray + (out - gray) * _draw(rng, SATURATION))
    if rng.random() < PROBABILITY["gain"]:
        out = _clip(out * rng.uniform(*CHANNEL_GAIN, size=3).astype(np.float32))
    if rng.random() < PROBABILITY["gray"]:
        out = _clip(np.repeat(_gray(out), 3, axis=2))
    if rng.random() < PROBABILITY["blur"]:
        out = _clip(_gaussian_blur(out, _draw(rng, BLUR_SIGMA)))
    if rng.random() < PROBABILITY["noise"]:
        out = _clip(out + rng.normal(0.0, _draw(rng, NOISE_SIGMA), size=out.shape))
    return out
