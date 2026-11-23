"""Free second reader (RapidOCR, ONNX). Runs inside the app, so no photo leaves the server.
Used only when the first reader is not confident."""
import logging
import time
from importlib.util import find_spec

import numpy as np

from image_reader import plate_tokens, vin_alternatives, vin_windows
from utils import is_valid_vin, norm_vin

log = logging.getLogger('fleetscan')
_ENGINE = None


def available():
    return find_spec('rapidocr') is not None and find_spec('onnxruntime') is not None


def _engine():
    global _ENGINE
    if _ENGINE is None:  # loaded once per server process, on first use only
        from rapidocr import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def vin_candidates(strings):
    strings = list(strings) + ([''.join(strings)] if len(strings) > 1 else [])
    direct, raw = [], []
    for s in strings:
        n = norm_vin(s)
        if 8 <= len(n) <= 24:
            raw.append(n)
        for w in vin_windows(s):
            if is_valid_vin(w) and w not in direct:
                direct.append(w)
    alts = []
    for n in dict.fromkeys(raw):
        for w in vin_windows(n):
            alts += [a for a in vin_alternatives(w) if a not in direct and a not in alts]
    near = [n for n in dict.fromkeys(raw) if 14 <= len(n) <= 20 and n not in direct]
    return direct + alts[:40] + near


def read_local(img, kind, emirate='Auto', timeout=12.0):
    """Return candidate strings (plate tokens or VINs); [] on any failure."""
    start = time.monotonic()
    arr = np.array(img.convert('RGB'))
    strings = []
    for k in ((0,) if kind == 'Plate' else (0, 1, 3)):  # VIN text may be vertical
        if time.monotonic() - start > timeout:
            break
        try:
            res = _engine()(np.ascontiguousarray(np.rot90(arr, k)))
        except Exception:
            log.exception('local reader failed')
            return []
        strings += [str(t) for t in (res.txts or ())]
        if kind == 'VIN' and any(is_valid_vin(w) for s in strings for w in vin_windows(s)):
            break
    return plate_tokens(strings, emirate) if kind == 'Plate' else vin_candidates(strings)
