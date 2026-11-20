"""Fleet index loading, ownership rule, and transparent candidate ranking."""
import difflib
import logging
import os
import re

import pandas as pd

from utils import is_valid_vin, norm_plate, norm_vin, timed

log = logging.getLogger('fleetscan')
RENAME = {'vin': '_vin', 'plate': '_plate', 'unit': '_unit', 'make': '_make', 'model': '_model',
          'year': '_year', 'branch': '_branch', 'location': '_location', 'status': '_status',
          'km': '_km', 'owner': '_owner', 'source_file': '_file', 'source_sheet': '_sheet',
          'priority': '_priority'}
FIELDS = list(RENAME.values())
OWNER_RANK = {'Moov': 0, 'Hertz': 1, 'AFF': 2}


def normalize_frame(df):
    df = df.rename(columns=RENAME).copy()
    for c in FIELDS:
        if c not in df:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str).str.strip()
    df['_vin'] = df['_vin'].map(norm_vin)
    df['_plate'] = df['_plate'].map(norm_plate)
    df['_pdigits'] = df['_plate'].str.replace(r'\D', '', regex=True)
    df['_pletters'] = df['_plate'].str.replace(r'[^A-Z]', '', regex=True)
    return df[(df['_vin'] != '') | (df['_plate'] != '')].reset_index(drop=True)


def load_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with timed('fleet_index_load'):
        return normalize_frame(pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig'))


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


def _digits_match(series, d):
    m = series == d
    if len(d) >= 4:
        m = m | (series.str.endswith(d) & (series.str.len() - len(d) <= 2))
    return m


def find_candidates(fleet, observed, kind, manual=False, limit=3):
    """Ranked fleet candidates: dicts with key, confidence, reason, owner, record."""
    if fleet.empty:
        return []
    found = {}

    def add(idx, conf, reason):
        for i in idx:
            if i not in found or conf > found[i][0]:
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
        if not found:
            known = [v for v in vins.unique() if v]
            for o in obs:
                if len(o) >= 10:
                    for k in difflib.get_close_matches(o, known, n=3, cutoff=.7):
                        add(fleet.index[vins == k], confidence(o, k), 'Fleet-assisted OCR match')
    else:
        obs = [o for o in dict.fromkeys(norm_plate(x) for x in observed) if o]
        plates = fleet['_plate']
        for o in obs:
            add(fleet.index[plates == o], 1.0, 'Manual entry' if manual else 'Exact plate')
        if not found:
            for o in obs:
                d, letters = re.sub(r'\D', '', o), re.sub(r'[^A-Z]', '', o)
                if d and letters:
                    m = _digits_match(fleet['_pdigits'], d) & fleet['_pletters'].str.contains(letters, regex=False)
                    add(fleet.index[m], .96, 'Numeric plate match')
        if not found:
            for o in obs:
                d = re.sub(r'\D', '', o)
                if len(d) >= 2:
                    idx = fleet.index[_digits_match(fleet['_pdigits'], d)]
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
    for i, (conf, reason) in sorted(found.items(), key=lambda kv: kv[1][0], reverse=True):
        rec = resolve_record(fleet, fleet.loc[i])
        key = rec['_vin'] or (rec['_plate'] + '|' + rec['_unit'])
        if key in seen:
            continue
        seen.add(key)
        out.append({'key': key, 'confidence': conf, 'reason': reason,
                    'owner': rec['_owner'], 'record': {f: str(rec[f]) for f in FIELDS}})
        if len(out) >= limit:
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
    digits = set(fleet['_pdigits']) | {d[-5:] for d in fleet['_pdigits'] if len(d) > 5}
    return lambda t: t in plates or (len(re.sub(r'\D', '', t)) >= 4 and re.sub(r'\D', '', t) in digits)


def merge_matches(base, extra, tag):
    """Combine candidate lists from two readers; keep the best score per vehicle, top 3."""
    best = {m['key']: m for m in base}
    for m in extra:
        if m['key'] not in best or m['confidence'] > best[m['key']]['confidence']:
            best[m['key']] = dict(m, reason=m['reason'] + f' ({tag})')
    return sorted(best.values(), key=lambda m: m['confidence'], reverse=True)[:3]          
