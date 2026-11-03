import numpy as np

FOG_AIRLIGHT = 0.85
FOG_DENSITY_TOP = 1.0
FOG_DENSITY_BOTTOM = 0.4
FOG_NOISE_GRID = 8
FOG_NOISE_AMPLITUDE = 0.15
HAZE_DENSITY = 0.9
DEFAULT_TINT = (0.85, 0.75, 0.55)
RAIN_DENSITY = 0.01
RAIN_LENGTH = 6
RAIN_STRENGTH = 0.35
SNOW_DENSITY = 0.02
SNOW_SIZE = 2
SNOW_LEVEL = 0.95


def _check_image(image) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.dtype not in (np.float32, np.float64):
        raise ValueError(f"image must be float32 or float64, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains non-finite values")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must be in [0, 1]")
    return image.astype(np.float32, copy=True)


def _check_severity(severity: float) -> float:
    severity = float(severity)
    if not np.isfinite(severity) or severity < 0.0 or severity > 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    return severity


def _blend(image: np.ndarray, transmission: np.ndarray, airlight) -> np.ndarray:
    air = np.asarray(airlight, dtype=np.float32)
    out = image * transmission + air * (1.0 - transmission)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def add_fog(image, severity, rng, noise=False):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    h, w, _ = out.shape
    density = np.linspace(FOG_DENSITY_TOP, FOG_DENSITY_BOTTOM, h, dtype=np.float32)[:, None]
    density = np.broadcast_to(density, (h, w)).copy()
    if noise:
        grid = rng.random((FOG_NOISE_GRID, FOG_NOISE_GRID)).astype(np.float32)
        rows = -(-h // FOG_NOISE_GRID)
        cols = -(-w // FOG_NOISE_GRID)
        field = np.repeat(np.repeat(grid, rows, axis=0), cols, axis=1)[:h, :w]
        density = density * (1.0 + FOG_NOISE_AMPLITUDE * (2.0 * field - 1.0))
    transmission = np.clip(1.0 - severity * density, 0.0, 1.0)[:, :, None]
    return _blend(out, transmission, FOG_AIRLIGHT)


def add_haze(image, severity, rng, tint=DEFAULT_TINT):
    out = _check_image(image)
    severity = _check_severity(severity)
    tint = tuple(float(t) for t in tint)
    if len(tint) != 3 or any(not np.isfinite(t) or t < 0.0 or t > 1.0 for t in tint):
        raise ValueError(f"tint must be three values in [0, 1], got {tint}")
    if severity == 0.0:
        return out
    transmission = np.float32(1.0 - severity * HAZE_DENSITY)
    return _blend(out, transmission, tint)


def add_rain(image, severity, rng):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    h, w, _ = out.shape
    count = int(round(severity * RAIN_DENSITY * h * w))
    ys = rng.integers(0, h, size=count)
    xs = rng.integers(0, w, size=count)
    for step in range(RAIN_LENGTH):
        yy = np.clip(ys + step, 0, h - 1)
        xx = np.clip(xs + step // 3, 0, w - 1)
        out[yy, xx] = np.minimum(out[yy, xx] + RAIN_STRENGTH, 1.0)
    return out


def add_snow(image, severity, rng):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    h, w, _ = out.shape
    count = int(round(severity * SNOW_DENSITY * h * w))
    ys = rng.integers(0, h, size=count)
    xs = rng.integers(0, w, size=count)
    for dy in range(SNOW_SIZE):
        for dx in range(SNOW_SIZE):
            yy = np.clip(ys + dy, 0, h - 1)
            xx = np.clip(xs + dx, 0, w - 1)
            out[yy, xx] = np.maximum(out[yy, xx], SNOW_LEVEL)
    return out
