"""Barcode, VIN-label and UAE-plate readers. OpenCV/Tesseract/zxing load lazily."""
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from importlib.util import find_spec

import numpy as np

from utils import is_valid_vin, norm_vin, timed

log = logging.getLogger('fleetscan')
ALNUM = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CONFUSABLE = {'O': '0', '0': 'O', 'I': '1', '1': 'I', 'Z': '2', '2': 'Z',
              'S': '5', '5': 'S', 'G': '6', '6': 'G', 'B': '8', '8': 'B'}
TIMEOUT = 20.0


@dataclass
class ReadResult:
    candidates: list = field(default_factory=list)
    source: str = ''
    confidence: float = 0.0
    timed_out: bool = False


def ocr_available():
    return bool(shutil.which('tesseract')) and find_spec('pytesseract') is not None


def barcode_available():
    return find_spec('zxingcpp') is not None


def vin_windows(text):
    s = norm_vin(text)
    return [s[i:i + 17] for i in range(len(s) - 16)] if len(s) >= 17 else []


def vin_alternatives(window):
    """Candidate alternatives from common OCR confusions; never accepted as truth."""
    base = window.replace('I', '1').replace('O', '0').replace('Q', '0')
    out = []

    def add(v):
        if is_valid_vin(v) and v not in out:
            out.append(v)
    add(base)
    for i, ch in enumerate(base):
        if ch in CONFUSABLE:
            add(base[:i] + CONFUSABLE[ch] + base[i + 1:])
    return out


def read_barcode(img, deadline):
    """Original + 90/180/270 rotations; returns (valid_vin_or_None, alternatives)."""
    import zxingcpp
    arr = np.array(img.convert('RGB'))
    alts = []
    for k in range(4):
        if time.monotonic() > deadline:
            break
        for res in zxingcpp.read_barcodes(np.ascontiguousarray(np.rot90(arr, k))):
            text = norm_vin(res.text)
            for w in vin_windows(text):
                if is_valid_vin(w):
                    return w, alts
            if text and text not in alts:
                alts.append(text)
    return None, alts


def _ocr(im, psm, whitelist, deadline):
    import pytesseract
    remaining = deadline - time.monotonic()
    if remaining <= 0.5:
        raise TimeoutError('deadline')
    cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
    d = pytesseract.image_to_data(im, lang='eng', config=cfg, output_type=pytesseract.Output.DICT,
                                  timeout=min(5, remaining))
    lines, words, confs = {}, [], []
    for t, c, b, p, l in zip(d['text'], d['conf'], d['block_num'], d['par_num'], d['line_num']):
        t = str(t).strip()
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = -1
        if t and c >= 0:
            lines.setdefault((b, p, l), []).append(t)
            words.append(t)
            confs.append(c)
    joined = [''.join(v) for v in lines.values()]
    return words + joined + ([''.join(words)] if len(words) > 1 else []), (float(np.mean(confs)) / 100 if confs else 0.0)


def locate_text_region(gray):
    """Crop to the densest wide/tall text-like cluster; falls back to the whole image."""
    import cv2
    h, w = gray.shape
    grad = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.bitwise_or(
        cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))),
        cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [b for b in boxes if b[2] * b[3] > .002 * w * h and max(b[2] / b[3], b[3] / b[2]) >= 3]
    if not boxes:
        return gray
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:3]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    mx, my = int(.04 * w), int(.04 * h)
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    if (x1 - x0) * (y1 - y0) > .85 * w * h:
        return gray
    crop = gray[y0:y1, x0:x1]
    if crop.shape[1] < 800:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return crop


def vin_variants(gray):
    import cv2
    clahe = cv2.createCLAHE(2.5, (8, 8)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 1.2), -.6, 0)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    capped = np.clip(gray, 0, np.percentile(gray, 96)).astype(np.float32)  # tame glare highlights
    deglare = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.normalize(capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return {'gray': gray, 'clahe': clahe, 'sharp': sharp, 'otsu': otsu, 'inv': 255 - otsu, 'deglare': deglare}


# (source, variant, rotation, tesseract psm): 7 = single line, 11 = sparse text. Bounded to 12 runs.
VIN_ATTEMPTS = [('crop', 'clahe', 0, 11), ('crop', 'clahe', 90, 11), ('crop', 'sharp', 0, 7),
                ('crop', 'otsu', 0, 11), ('crop', 'inv', 0, 11), ('crop', 'deglare', 0, 11),
                ('crop', 'clahe', 270, 11), ('crop', 'clahe', 180, 11), ('full', 'sharp', 0, 11),
                ('full', 'sharp', 90, 11), ('crop', 'sharp', 90, 7), ('crop', 'otsu', 90, 11)]


def read_vin_ocr(img, deadline, stop_when=None, progress=None):
    import cv2
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if progress:
        progress('Locating text region')
    crop = locate_text_region(gray)
    sources, cache = {'crop': crop, 'full': gray}, {}
    if progress:
        progress('Reading characters')
    direct, raw, confs = [], [], []
    for n, (src, name, angle, psm) in enumerate(VIN_ATTEMPTS):
        if time.monotonic() >= deadline - .5:
            break
        if src == 'full' and crop is gray:
            continue
        if src not in cache:
            cache[src] = vin_variants(sources[src])
        im = np.ascontiguousarray(np.rot90(cache[src][name], angle // 90))
        try:
            strings, conf = _ocr(im, psm, ALNUM, deadline)
        except Exception as exc:
            log.warning('VIN OCR attempt failed: %s', exc)
            continue
        confs.append(conf)
        found = []
        for s in strings:
            n_s = norm_vin(s)
            if 8 <= len(n_s) <= 24:
                raw.append(n_s)
            for w in vin_windows(s):
                if is_valid_vin(w) and w not in direct:
                    direct.append(w)
                    found.append(w)
        if stop_when and any(stop_when(v) for v in found):
            break
        if found and (not stop_when or n >= 5):
            break
    alts = []
    for s in dict.fromkeys(raw):
        for w in vin_windows(s):
            if not is_valid_vin(w) or not (stop_when and stop_when(w)):
                alts += [a for a in vin_alternatives(w) if a not in direct and a not in alts]
    near = [s for s in dict.fromkeys(raw) if 14 <= len(s) <= 20 and s not in direct]
    return direct + alts[:40] + near, (max(confs) if confs else 0.0)


def _order_points(pts):
    pts = pts.reshape(4, 2).astype('float32')
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype='float32')


def plate_regions(rgb):
    """Likely plate crops (perspective-corrected when 4 corners are found), best first.
    Lower 60% of the frame only; the bottom band (phone timestamp) is excluded."""
    import cv2
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.bilateralFilter(gray, 9, 60, 60), 60, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0 or not .004 * w * h <= bw * bh <= .5 * w * h or not 1.2 <= bw / bh <= 6.5:
            continue
        cy = y + bh / 2
        if cy < .4 * h or cy > .93 * h:
            continue
        scored.append((bw * bh * (1 + cy / h), (x, y, bw, bh), c))
    crops = []
    for _, (x, y, bw, bh), c in sorted(scored, key=lambda t: t[0], reverse=True)[:3]:
        hull = cv2.convexHull(c)
        quad = cv2.approxPolyDP(hull, .03 * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            o = _order_points(quad)
            wd = int(max(np.linalg.norm(o[0] - o[1]), np.linalg.norm(o[3] - o[2])))
            ht = int(max(np.linalg.norm(o[0] - o[3]), np.linalg.norm(o[1] - o[2])))
            if wd > 40 and ht > 15 and 1.2 <= wd / ht <= 6.5:
                m = cv2.getPerspectiveTransform(o, np.array([[0, 0], [wd, 0], [wd, ht], [0, ht]], dtype='float32'))
                crops.append(cv2.warpPerspective(rgb, m, (wd, ht)))
                continue
        ex, ey = int(.08 * bw), int(.08 * bh)
        crops.append(rgb[max(0, y - ey):min(h, y + bh + ey), max(0, x - ex):min(w, x + bw + ex)])
    crops.append(rgb[int(.4 * h):int(.92 * h)])  # fallback: lower region, timestamp band removed
    return crops


def plate_images(crop, invert=False):
    """Enhanced + thresholded crops. invert adds white-on-red/green/grey category panels."""
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    if gray.shape[1] < 600:
        s = min(3.0, 600 / gray.shape[1])
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(2.5, (8, 8)).apply(gray)
    otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return [enhanced, otsu] + ([255 - otsu] if invert else [])


def plate_tokens(strings, emirate='Auto'):
    """Main number first. Abu Dhabi (AUH) = category number + digits; other emirates = letter code + digits.
    Handles 'G77926', '5 24910', '90 21338', and a code printed apart from the number."""
    direct, digits, letters, smalls = [], [], [], []
    for s in strings:
        c = re.sub(r'[^A-Z0-9]', '', s.upper())
        if not c:
            continue
        if re.fullmatch(r'[A-Z]', c):
            if c not in letters:
                letters.append(c)
            continue
        if re.fullmatch(r'\d{1,2}', c) and c not in smalls:
            smalls.append(c)
        if re.fullmatch(r'[A-Z]{1,2}\d{2,5}|\d{2,5}[A-Z]{1,2}', c):
            direct.append(c)
        for run in re.findall(r'\d+', c):
            cands = [run[-5:], run[-4:]] if len(run) > 6 else ([run, run[-5:]] if len(run) == 6 else [run])
            digits += [d for d in cands if len(d) >= 2 and d not in digits]
    digits.sort(key=lambda t: 0 if len(t) in (4, 5) else 1)
    letter_combos = [l + d for d in digits[:3] for l in letters[:3]]
    cat_combos = [m + d for d in digits[:3] for m in smalls[:3] if m != d]
    if emirate == 'AUH':
        return list(dict.fromkeys(cat_combos + [d for d in direct if d.isdigit()] + digits))
    if emirate != 'Auto':
        return list(dict.fromkeys(direct + digits + letter_combos))
    return list(dict.fromkeys(direct + digits + letter_combos + cat_combos))


def read_plate_ocr(img, deadline, stop_when=None, progress=None, framed=False, emirate='Auto'):
    """framed=True: the photo was cropped to the on-screen guide frame, so read it whole first."""
    if progress:
        progress('Locating text region')
    rgb = np.array(img)
    crops = ([rgb] if framed else []) + plate_regions(rgb)
    if progress:
        progress('Reading characters')
    tokens, confs, done = [], [], False
    for n, crop in enumerate(crops):
        for im in plate_images(crop, invert=(n == 0)):
            if time.monotonic() >= deadline - .5:
                done = True
                break
            try:
                strings, conf = _ocr(im, 11, ALNUM, deadline)
            except Exception as exc:
                log.warning('plate OCR attempt failed: %s', exc)
                continue
            confs.append(conf)
            found = plate_tokens(strings, emirate)
            tokens += [t for t in found if t not in tokens]
            if stop_when and any(stop_when(t) for t in found):
                done = True
                break
        if done:
            break
    return tokens, (max(confs) if confs else 0.0)


def process(img, mode, stop_when=None, progress=None, timeout=TIMEOUT, framed=False, emirate='Auto'):
    """Barcode first for VIN modes (stop on a valid VIN, skip OCR), then OCR. Hard time limit."""
    deadline = time.monotonic() + timeout
    say = progress or (lambda s: None)
    say('Preparing photo')
    res = ReadResult()
    try:
        if mode in ('VIN label', 'VIN barcode / QR'):
            alts = []
            if barcode_available():
                with timed('barcode'):
                    vin, alts = read_barcode(img, deadline)
                if vin:
                    return ReadResult([vin], 'barcode', 1.0)
            if ocr_available():
                with timed('ocr_vin'):
                    cands, res.confidence = read_vin_ocr(img, deadline, stop_when, say)
                res.candidates, res.source = cands + [a for a in alts if a not in cands], 'ocr'
            else:
                res.candidates = alts
        elif ocr_available():
            with timed('ocr_plate'):
                res.candidates, res.confidence = read_plate_ocr(img, deadline, stop_when, say, framed, emirate)
            res.source = 'ocr'
    except Exception:
        log.exception('image processing failed')
    res.timed_out = time.monotonic() >= deadline
    return res
