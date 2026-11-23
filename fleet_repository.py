"""Fleet index loading, ownership rule, and transparent candidate ranking."""
import difflib
import io
import logging
import os
import re

import pandas as pd

from utils import is_valid_vin, norm_plate, norm_vin, timed

log = logging.getLogger('fleetscan')
RENAME = {'vin': '_vin', 'plate': '_plate', 'unit': '_unit', 'make': '_make', 'model': '_model',
          'year': '_year', 'branch': '_branch', 'location': '_location', 'status': '_status',
          'km': '_km', 'owner': '_owner', 'source_file': '_file', 'source_sheet': '_sheet',
          'priority': '_priority', 'plate_emirate': '_emirate', 'plate_code': '_pcode',
          'plate_number': '_pnum', 'vin_last5': '_vin5'}
FIELDS = list(RENAME.values())
OWNER_RANK = {'Moov': 0, 'Hertz': 1, 'AFF': 2}

# Alternative header names that are mapped to the standard ones above
HEADER_ALIASES = {
    'vin_no': 'vin', 'vin_number': 'vin', 'chassis': 'vin', 'chassis_no': 'vin',
    'chassis_number': 'vin', 'vehicle_chassis_number': 'vin',
    'plate_no': 'plate',
    'unit_no': 'unit', 'unit_number': 'unit', 'vehicle_unit_number_identifier': 'unit',
    'source': 'source_file', 'sheet': 'source_sheet',
}


def normalize_frame(df):
    df = df.rename(columns=RENAME).copy()
    for c in FIELDS:
        if c not in df:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str).str.strip()
    df['_vin'] = df['_vin'].map(norm_vin)
    df['_plate'] = df['_plate'].map(norm_plate)
    code, num = df['_pcode'].map(norm_plate), df['_pnum'].str.replace(r'\D', '', regex=True)
    df['_plate'] = df['_plate'].where(df['_plate'] != '', code + num)  # build the plate from code + number
    df['_emirate'] = df['_emirate'].str.upper()
    df['_pdigits'] = df['_plate'].str.replace(r'\D', '', regex=True)
    df['_pletters'] = df['_plate'].str.replace(r'[^A-Z]', '', regex=True)
    df['_pnum'] = num.where(num != '', df['_pdigits'])
    df['_vin5'] = df['_vin5'].map(norm_vin).where(df['_vin5'].map(norm_vin) != '', df['_vin'].str[-5:])
    return df[(df['_vin'] != '') | (df['_plate'] != '')].reset_index(drop=True)


def _clean_header(h):
    """Normalise a header: strip spaces/BOM, lowercase, underscores, then apply aliases."""
    h = str(h).replace('\ufeff', '').strip().lower()
    h = re.sub(r'[^a-z0-9]+', '_', h).strip('_')
    return HEADER_ALIASES.get(h, h)


def _decode(raw):
    """Handle UTF-16, UTF-8 (with or without BOM) and Excel's Windows-1252 'CSV'."""
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        return raw.decode('utf-16')
    for enc in ('utf-8-sig', 'cp1252'):
        try:
            return raw.decode(enc)
        except UnicodeError:
            pass
    return raw.decode('latin-1')


def _detect_sep(text):
    """Comma, semicolon, tab or pipe, taken from the header line."""
    first = next((ln for ln in text.splitlines() if ln.strip()), '')
    counts = {s: first.count(s) for s in (',', ';', '\t', '|')}
    sep = max(counts, key=counts.get)
    return sep if counts[sep] else ','


def read_fleet_csv(path):
    """Read the CSV whatever the encoding, delimiter or header style."""
    with open(path, 'rb') as f:
        raw = f.read()
    if not raw.strip():
        raise ValueError('fleet_index.csv is empty')
    if raw.startswith(b'version https://git-lfs'):
        raise ValueError('fleet_index.csv is a Git LFS pointer, not the real file')
    text = _decode(raw)
    df = pd.read_csv(io.StringIO(text), sep=_detect_sep(text), dtype=str,
                     keep_default_na=False, on_bad_lines='skip')
    df.columns = [_clean_header(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    if not ({'vin', 'plate', 'plate_number'} & set(df.columns)):
        raise ValueError('No vin/plate column found. Columns read: ' + ', '.join(df.columns))
    return df


def load_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with timed('fleet_index_load'):
        df = normalize_frame(read_fleet_csv(path))
    if df.empty:
        raise ValueError('fleet_index.csv has no rows with a VIN or plate')
    return df


def pick_owner(owners):
    """Moov wins over Hertz, Hertz over AFF."""
    return min(owners, key=lambda o: OWNER_RANK.get(o, 9)) if owners else ''


def resolve_record(fleet, row):
    """Ownership is decided by the final fleet record: same VIN in Moov and Hertz -> Moov."""
    v = row['_vin']
    if v:
        same = fleet[fleet['_vin'] == v]
        if len(same):
            best = pick_owner(set(same['_owner']))
            return same[same['_owner'] == best].iloc[0]
    return row


def owner_for_vin(fleet, vin):
    v = norm_vin(vin)
    if fleet.empty or not v:
        return ''
    return pick_owner(set(fleet.loc[fleet['_vin'] == v, '_owner']))


def confidence(observed, candidate, unique=True, exists=True):
    """Transparent score: similarity, positional matches, length, suffix, uniqueness, existence."""
    if not observed or not candidate:
        return 0.0
    if observed == candidate:
        return 1.0 if unique else 0.97
    sim = difflib.SequenceMatcher(None, observed, candidate).ratio()
    length = 1 - min(1, abs(len(observed) - len(candidate)) / max(len(candidate), 1))
    pos = sum(a == b for a, b in zip(observed, candidate)) / len(candidate)
    n = min(8, len(observed), len(candidate))
    suffix = 1.0 if observed[-n:] == candidate[-n:] else 0.0
    score = .55 * sim + .15 * length + .10 * pos + .10 * suffix + (.10 if unique else 0)
    return round(min(.99, score if exists else score * .5), 4)


def band(conf):
    return 'strong' if conf >= .95 else ('likely' if conf >= .80 else 'low')


def _digits_match(fleet, d):
    m = (fleet['_pdigits'] == d) | (fleet['_pnum'] == d)
    if len(d) >= 4:
        s = fleet['_pdigits']
        m = m | (s.str.endswith(d) & (s.str.len() - len(d) <= 2))
    return m


def _emirate_pool(fleet, kind, emirate):
    """Use the plate_emirate column when the data has it; otherwise infer AUH (no letters) / DXB (letter code)."""
    if kind != 'Plate' or emirate == 'Auto':
        return fleet
    if (fleet['_emirate'] != '').any():
        pool = fleet[fleet['_emirate'] == emirate]
    elif emirate == 'AUH':
        pool = fleet[fleet['_pletters'] == '']
    elif emirate == 'DXB':
        pool = fleet[fleet['_pletters'] != '']
    else:
        return fleet
    return pool if len(pool) else fleet


def find_candidates(fleet, observed, kind, manual=False, limit=3, emirate='Auto'):
    """Search the emirate's plate style first; if nothing matches there, search the whole fleet."""
    pool = _emirate_pool(fleet, kind, emirate)
    out = _find(pool, observed, kind, manual, limit, full=fleet)
    return out if out or pool is fleet else _find(fleet, observed, kind, manual, limit)


def _find(fleet, observed, kind, manual=False, limit=3, full=None):
    """Ranked fleet candidates: dicts with key, confidence, reason, owner, record."""
    if fleet.empty:
        return []
    found = {}
    best_conf = {}

    def add(idx, conf, reason):
        for i in idx:
            if conf > best_conf.get(i, -1.0):
                best_conf[i] = conf
                found[i] = (conf, reason)

    if kind == 'VIN':
        obs = [o for o in dict.fromkeys(norm_vin(x) for x in observed) if o]
        vins = fleet['_vin']
        for o in obs:
            if is_valid_vin(o):
                add(fleet.index[vins == o], 1.0, 'Manual entry' if manual else 'Exact VIN')
        if not found:
            for o in obs:
                if len(o) >= 8:
                    idx = fleet.index[vins.str.endswith(o[-8:])]
                    if len(set(vins[idx])) == 1:
                        best = max(confidence(o, vins[idx[0]]), .85)
                        add(idx, best, 'Unique VIN last-eight match')

        def last5(cands):
            for o in cands:
                idx = fleet.index[fleet['_vin5'] == o[-5:]]
                n = len(set(vins[idx]))
                if n == 1:
                    add(idx, .85, 'Unique VIN last-five match')
                elif 1 < n <= 6:
                    add(idx, .70, 'VIN last-five match (choose one)')

        if not found:
            last5([o for o in obs if 5 <= len(o) < 8])  # the employee typed only the last digits
        if not found:
            known = [v for v in vins.unique() if v]
            for o in obs:
                if len(o) >= 10:
                    for k in difflib.get_close_matches(o, known, n=3, cutoff=.7):
                        add(fleet.index[vins == k], confidence(o, k), 'Fleet-assisted OCR match')
        if not found:
            last5([o for o in obs if len(o) >= 8])  # tail of a long, partly misread VIN
    else:
        obs = [o for o in dict.fromkeys(norm_plate(x) for x in observed) if o]
        plates = fleet['_plate']
        for o in obs:
            add(fleet.index[plates == o], 1.0, 'Manual entry' if manual else 'Exact plate')
        if not found:
            for o in obs:
                d, letters = re.sub(r'\D', '', o), re.sub(r'[^A-Z]', '', o)
                if d and letters:
                    m = _digits_match(fleet, d) & fleet['_pletters'].str.contains(letters, regex=False)
                    add(fleet.index[m], .96, 'Numeric plate match')
        if not found:
            for o in obs:
                d = re.sub(r'\D', '', o)
                if len(d) >= 2:
                    idx = fleet.index[_digits_match(fleet, d)]
                    if len(idx):
                        uniq = len(set(plates[idx])) == 1
                        add(idx, .88 if uniq else .70, 'Numeric plate match')
        if not found:
            known = [p for p in plates.unique() if 2 <= len(p) <= 12]
            for o in obs:
                if len(o) >= 3:
                    for k in difflib.get_close_matches(o, known, n=3, cutoff=.6):
                        add(fleet.index[plates == k], confidence(o, k), 'Fleet-assisted OCR match')

    out, seen = [], set()
    cap = 6 if found and all('last-five' in r for _, r in found.values()) else limit
    for i, (conf, reason) in sorted(found.items(), key=lambda kv: kv[1][0], reverse=True):
        rec = resolve_record(full if full is not None else fleet, fleet.loc[i])
        key = rec['_vin'] or (rec['_plate'] + '|' + rec['_unit'])
        if key in seen:
            continue
        seen.add(key)
        out.append({'key': key, 'confidence': conf, 'reason': reason,
                    'owner': rec['_owner'], 'record': {f: str(rec[f]) for f in FIELDS}})
        if len(out) >= cap:
            break
    return out


def make_stop_fn(fleet, kind):
    """Cheap membership test used by OCR for early stopping."""
    if fleet.empty:
        return None
    if kind == 'VIN':
        known = set(fleet['_vin'])
        return lambda v: v in known
    plates = set(fleet['_plate'])
    digits = set(fleet['_pdigits']) | set(fleet['_pnum']) | {d[-5:] for d in fleet['_pdigits'] if len(d) > 5}
    return lambda t: t in plates or (len(re.sub(r'\D', '', t)) >= 4 and re.sub(r'\D', '', t) in digits)


def merge_matches(base, extra, tag):
    """Combine candidate lists from two readers; keep the best score per vehicle, top 3."""
    best = {m['key']: m for m in base}
    for m in extra:
        if m['key'] not in best or m['confidence'] > best[m['key']]['confidence']:
            best[m['key']] = dict(m, reason=m['reason'] + f' ({tag})')
    return sorted(best.values(), key=lambda m: m['confidence'], reverse=True)[:3]


def data_quality(fleet):
    """Counts that help when cleaning fleet_index.csv."""
    v = fleet['_vin']
    has_vin, has_plate = fleet[v != ''], fleet[fleet['_plate'] != '']
    return {
        'rows': len(fleet),
        'missing VIN': int((v == '').sum()),
        'VIN not 17 valid characters': int(((v != '') & ~v.map(is_valid_vin)).sum()),
        'missing plate': int((fleet['_plate'] == '').sum()),
        'missing plate_emirate': int((fleet['_emirate'] == '').sum()),
        'duplicate VIN + owner rows': int(has_vin.duplicated(['_vin', '_owner']).sum()),
        'VINs listed under more than one owner': int((has_vin.groupby('_vin')['_owner'].nunique() > 1).sum()),
        'plates shared by different VINs': int((has_plate.groupby('_plate')['_vin'].nunique() > 1).sum()),
    }
