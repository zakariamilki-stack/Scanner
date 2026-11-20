"""Shared helpers: normalisation, validation, naming and image preparation."""
import hashlib
import io
import logging
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

APP_VERSION = '2.0.0'
TZ = ZoneInfo('Asia/Dubai')
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_FORMATS = {'JPEG', 'PNG', 'MPO', 'WEBP'}
DEFAULT_PHOTO_ACTIONS = ('Stock Count',)
log = logging.getLogger('fleetscan')


def now_dubai():
    return datetime.now(TZ)


@contextmanager
def timed(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info('timing %s %.2fs', label, time.perf_counter() - start)


def txt(v):
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    x = str(v).strip()
    return x[:-2] if x.endswith('.0') and x[:-2].isdigit() else x


def norm_vin(v):
    return re.sub(r'[^A-Z0-9]', '', txt(v).upper())


def norm_plate(v):
    return re.sub(r'[^A-Z0-9\u0600-\u06FF]', '', txt(v).upper())


def is_valid_vin(v):
    """17 characters from A-H, J-N, P, R-Z, 0-9 (I, O and Q are never valid)."""
    return bool(re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}', v or ''))


def sanitize_username(username):
    """Immutable folder key: lowercase, spaces -> underscores, no unsafe characters."""
    s = txt(username).lower().replace(' ', '_')
    s = re.sub(r'[^a-z0-9_-]', '', s)
    s = re.sub(r'_+', '_', s).strip('_-')
    if not s:
        raise ValueError('Invalid username')
    return s


def safe_segment(name, default='folder'):
    s = re.sub(r'[\\/:*?"<>|#%~&{}]', '', txt(name)).strip(' .')
    return s or default


def safe_file_name(name):
    base = re.sub(r'[^A-Za-z0-9._-]', '_', txt(name).replace('\\', '/').split('/')[-1]).lstrip('.')
    return base or 'photo.jpg'


def photo_file_name(when, vin='', plate='', kind='PLATE'):
    """YYYYMMDD_HHMMSS_<plate-or-vin-or-UNKNOWN>_<scan-type>.jpg"""
    v = norm_vin(vin)
    ident = v if is_valid_vin(v) else re.sub(r'[^A-Z0-9]', '', norm_plate(plate))
    tag = 'VIN' if txt(kind).upper().startswith('VIN') else 'PLATE'
    return f"{when:%Y%m%d_%H%M%S}_{ident or 'UNKNOWN'}_{tag}.jpg"


def photo_required(action, required_actions=DEFAULT_PHOTO_ACTIONS):
    return action in set(required_actions)


def validate_save(action, has_photo, required_actions=DEFAULT_PHOTO_ACTIONS):
    """Return an error message, or '' when the save may proceed."""
    if photo_required(action, required_actions) and not has_photo:
        return f'A photo is required for "{action}". Take or upload one first.'
    return ''


def to_dict(obj):
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def image_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def load_image(raw, max_edge=1600):
    """Validate, fix EXIF orientation, convert to RGB, cap the long edge (never upscale)."""
    if not raw:
        raise ValueError('Empty image.')
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError('Image is larger than 15 MB.')
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Unsupported or corrupt image.') from exc
    if fmt not in ALLOWED_FORMATS:
        raise ValueError('Unsupported image format.')
    img = ImageOps.exif_transpose(img).convert('RGB')
    longest = max(img.size)
    if longest > max_edge:
        s = max_edge / longest
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def prepare_photo(raw):
    """Evidence photo: JPEG, oriented, long edge <= 1920, quality 82, no EXIF."""
    img = load_image(raw, 1920)
    out = io.BytesIO()
    img.save(out, 'JPEG', quality=82, optimize=True)
    return out.getvalue()
