"""Fleet Scan - mobile-first fleet stock count and vehicle identification (Streamlit)."""
import html
import io
import logging
import os
import re
import uuid

import pandas as pd
import streamlit as st

import ai_reader
import auth
from camera_component import frame_camera
import fleet_repository as fr
import image_reader as ir
import photo_storage as ps
import utils
from scan_repository import SQLiteScanRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
log = logging.getLogger('fleetscan')
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = 'fleet_index.csv'
DB_FILE = 'fleet_scan.db'
ACTIONS = ['Stock Count', 'Vehicle Received', 'Vehicle Moved', 'Inspection Completed',
           'Vehicle Collected', 'Auction Preparation', 'Hayaza Check', 'Other']
MODES = ['Plate', 'VIN label', 'VIN barcode / QR', 'Manual']
UNMATCHED = 'Not in fleet (save as unmatched)'

st.set_page_config(page_title='Fleet Scan', page_icon='🚘', layout='centered', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#f3f6fa}.block-container{max-width:760px;padding:.6rem .7rem 5rem;overflow-x:hidden}
.hero{background:linear-gradient(135deg,#052c55,#0876cf);color:#fff;padding:14px 16px;border-radius:15px;margin-bottom:8px}
.hero h1{font-size:24px;margin:0;color:#fff}.hero p{font-size:13px;margin:3px 0 0;opacity:.92}
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button{min-height:50px;border-radius:11px;font-weight:750}
[data-testid="stCameraInput"] button{min-height:58px}
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div{min-height:48px}
[data-testid="stImage"] img{max-height:300px;object-fit:contain}
.card{background:#fff;border:1px solid #dce5ef;border-radius:13px;padding:11px 13px;margin:6px 0;font-size:14px;line-height:1.5;word-break:break-word}
.ok{background:#e3f6ea;border-left:6px solid #1f7a35;padding:12px;border-radius:9px;color:#0b3d1a}
.warn{background:#fff4d6;border-left:6px solid #b7791f;padding:12px;border-radius:9px;color:#5a3a00}
.bad{background:#ffe5e5;border-left:6px solid #c52b2b;padding:12px;border-radius:9px;color:#6b0f0f}
</style>''', unsafe_allow_html=True)


def secrets_dict():
    try:
        return utils.to_dict(st.secrets)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def cached_fleet(path):
    try:
        return fr.load_index(path), ''
    except Exception:
        log.exception('fleet index load failed')
        return pd.DataFrame(), 'Fleet index could not be loaded.'


@st.cache_resource(show_spinner=False)
def get_storage():
    return ps.build_storage(secrets_dict())


@st.cache_resource(show_spinner=False)
def get_repo():
    try:
        return SQLiteScanRepository(DB_FILE)
    except Exception:
        log.exception('scan repository init failed')
        return None


def box(kind, text):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def to_excel(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False)
    return out.getvalue()


# ------------------------------------------------------------------ login
def login_page(users):
    st.markdown('<div class="hero"><h1>Fleet Scan</h1><p>Sign in to start scanning.</p></div>', unsafe_allow_html=True)
    if not users:
        box('bad', 'No users are configured. Ask an administrator to add users to the app secrets.')
    with st.form('login_form'):
        u = st.text_input('Employee ID or username')
        p = st.text_input('Password', type='password')
        go = st.form_submit_button('Login', type='primary', width='stretch')
    if go:
        res = auth.authenticate(u, p, users)
        if res.user:
            auth.login(st.session_state, res.user)
            st.rerun()
        st.error(res.message)


# ------------------------------------------------------------------ scan
def reset_scan():
    n = st.session_state.get('cam_n', 0) + 1
    for k in ('matches', 'searched', 'read_key', 'ocr_conf', 'photo_raw', 'photo_framed', 'basic_seen',
              'value_input', 'notice'):
        st.session_state.pop(k, None)
    st.session_state['cam_n'] = n


def clear_results():
    for k in ('matches', 'searched', 'read_key', 'ocr_conf', 'notice'):
        st.session_state.pop(k, None)


def get_image(raw):
    h = utils.image_hash(raw)
    cache = st.session_state.get('img_cache')
    if cache and cache[0] == h:
        return h, cache[1], ''
    try:
        img = utils.load_image(raw)
    except ValueError as exc:
        log.warning('image rejected: %s', exc)
        return h, None, str(exc)
    st.session_state['img_cache'] = (h, img)
    return h, img, ''


def merge_matches(base, ai):
    best = {m['key']: m for m in base}
    for m in ai:
        m = dict(m, reason=m['reason'] + ' (AI-assisted)')
        if m['key'] not in best or m['confidence'] > best[m['key']]['confidence']:
            best[m['key']] = m
    return sorted(best.values(), key=lambda m: m['confidence'], reverse=True)[:3]


def run_read(img, h, mode, kind, fleet, ai_cfg=None):
    ss = st.session_state
    if ss.get('read_key') == (h, mode):
        return  # same photo and mode already read: never reprocess on rerun
    status = st.status('Reading photo...', expanded=True)
    try:
        stop = fr.make_stop_fn(fleet, kind)
        res = ir.process(img, mode, stop, status.write, framed=bool(ss.get('photo_framed')))
        status.write('Matching fleet records')
        with utils.timed('fleet_match'):
            matches = fr.find_candidates(fleet, res.candidates[:60], kind)
        ai_cands = []
        if (ai_cfg and res.source != 'barcode' and ss.get('ai_calls', 0) < ai_reader.CALL_LIMIT
                and (not matches or fr.band(matches[0]['confidence']) != 'strong')):
            status.write('Double-checking with AI')
            ss['ai_calls'] = ss.get('ai_calls', 0) + 1
            with utils.timed('ai_read'):
                ai_cands = ai_reader.read_with_ai(img, kind, ai_cfg)
            if ai_cands:
                matches = merge_matches(matches, fr.find_candidates(fleet, ai_cands, kind))
        status.update(label='Done', state='complete', expanded=False)
    except Exception:
        log.exception('read failed')
        status.update(label='Failed', state='error')
        st.error('Camera image could not be processed.')
        return
    ss.update(read_key=(h, mode), matches=matches, searched=True, ocr_conf=res.confidence,
              photo_raw=ss.get('photo_raw'), sel_nonce=ss.get('sel_nonce', 0) + 1)
    all_cands = ai_cands + res.candidates
    ss['value_input'] = all_cands[0] if all_cands else ''
    notes = []
    if not all_cands:
        notes.append('No reliable characters were detected. Move closer, clean the surface, or retry with/without flash.')
    if res.timed_out:
        notes.append('Time limit reached; showing the best candidates found. Correct manually if needed.')
    ss['notice'] = ' '.join(notes)


def card(c):
    r, e = c['record'], html.escape
    rows = [('VIN', r['_vin']), ('Plate', r['_plate']), ('Unit', r['_unit']),
            ('Make / model', f"{r['_make']} {r['_model']}".strip()), ('Year', r['_year']),
            ('Fleet owner', r['_owner']), ('Branch', r['_branch']), ('Location', r['_location']),
            ('Status', r['_status']), ('Confidence', f"{c['confidence']:.0%}"), ('Match reason', c['reason'])]
    return '<div class="card">' + '<br>'.join(f'<b>{k}:</b> {e(v) or "-"}' for k, v in rows) + '</div>'


def scan_tab(user, fleet, storage, repo, required_actions, ai_cfg=None):
    ss = st.session_state
    ss.setdefault('cam_n', 0)
    if ss.get('flash'):
        st.success(ss.pop('flash'))
    mode = st.segmented_control('Scan type', MODES, default='Plate', key='mode') or 'Plate'
    kind = 'VIN' if mode.startswith('VIN') else 'Plate'
    n = ss['cam_n']
    if mode != 'Manual':  # live camera with a guide frame; the browser crops to the frame
        shot = frame_camera(kind, n)
        if shot and shot['id'] != ss.get('cam_seen'):
            ss['cam_seen'] = shot['id']
            ss['photo_raw'], ss['photo_framed'] = shot['bytes'], True
            clear_results()
    with st.expander('Basic camera or upload instead'):
        use_basic = st.toggle('Use the basic camera', key=f'basic_{n}')
        cam = st.camera_input('Take photo', key=f'cam_{n}') if use_basic else None
        up = st.file_uploader('Choose image', type=['jpg', 'jpeg', 'png', 'webp'], key=f'up_{n}',
                              label_visibility='collapsed')
    f = cam or up
    if f is not None:
        b = f.getvalue()
        if utils.image_hash(b) != ss.get('basic_seen'):
            ss['basic_seen'], ss['photo_raw'], ss['photo_framed'] = utils.image_hash(b), b, False
            clear_results()
    raw = ss.get('photo_raw')
    img, h = None, ''
    if raw:
        h, img, err = get_image(raw)
        if img is None:
            st.error('Camera image could not be processed. ' + err)
        else:
            ss['photo_raw'] = raw
            st.image(img, width='stretch')
    if mode != 'Manual':
        needs = ir.barcode_available() if mode == 'VIN barcode / QR' else ir.ocr_available()
        if not needs and not ir.ocr_available():
            box('warn', 'Text reader is not available on this server. Use Manual mode.')
        if st.button('Read photo', type='primary', width='stretch', disabled=img is None):
            run_read(img, h, mode, kind, fleet, ai_cfg)
        if ai_cfg:
            st.caption('If the text reader is unsure, the cropped photo is sent to an AI service for a second opinion.')
    if st.button('Retake / reset', width='stretch'):
        reset_scan()
        st.rerun()
    if ss.get('notice'):
        box('warn', html.escape(ss['notice']))

    value = st.text_input('Confirm or correct ' + kind, key='value_input')
    if st.button('Find vehicle', type='primary', width='stretch', disabled=not value.strip()):
        ss.update(matches=fr.find_candidates(fleet, [value], kind, manual=True), searched=True,
                  sel_nonce=ss.get('sel_nonce', 0) + 1)
    if not ss.get('searched'):
        return

    matches, nonce = ss.get('matches', []), ss.get('sel_nonce', 0)
    rec, owner, conf, reason = {f: '' for f in fr.FIELDS}, 'Unmatched', None, 'Unmatched'
    if not matches:
        box('bad', '<b>No fleet match found.</b> Correct the value above or save as unmatched.')
    else:
        top = fr.band(matches[0]['confidence'])
        box({'strong': 'ok', 'likely': 'warn', 'low': 'bad'}[top],
            {'strong': '<b>Strong match.</b> Please confirm it is the right vehicle.',
             'likely': '<b>Likely match.</b> Check the candidates below.',
             'low': '<b>Low confidence.</b> Select a vehicle manually or correct the value.'}[top])
        for c in matches:
            st.markdown(card(c), unsafe_allow_html=True)
        labels = [f"{i + 1}. {c['record']['_plate'] or '-'} | {c['record']['_vin'] or '-'} | {c['confidence']:.0%}"
                  for i, c in enumerate(matches)]
        choice = st.radio('Confirm vehicle', labels + [UNMATCHED], key=f'choice_{nonce}',
                          index=0 if top == 'strong' else None)
        if choice is None:
            st.info('Select a vehicle or "Not in fleet" to continue.')
            return
        if choice != UNMATCHED:
            sel = matches[labels.index(choice)]
            rec, owner, conf, reason = sel['record'], sel['owner'], sel['confidence'], sel['reason']

    q = ss.get('value_input', '')
    with st.form('save_form'):
        action = st.selectbox('Action', ACTIONS)
        cv = st.text_input('Confirmed VIN', value=rec['_vin'] or (utils.norm_vin(q) if kind == 'VIN' else ''),
                           key=f'cv_{nonce}_{rec["_vin"]}')
        cp = st.text_input('Confirmed plate', value=rec['_plate'] or (utils.norm_plate(q) if kind == 'Plate' else ''),
                           key=f'cp_{nonce}_{rec["_plate"]}')
        loc = st.text_input('Location', value=rec['_location'], key=f'loc_{nonce}_{rec["_vin"]}')
        notes = st.text_area('Notes', key=f'notes_{nonce}')
        st.caption('Stock Count requires a photo that is stored successfully.' if storage.configured
                   else 'Photos are not stored in this version. The scan record is saved.')
        go = st.form_submit_button('Confirm and save', type='primary', width='stretch')
    if go:
        save_scan(user, fleet, storage, repo, required_actions, mode, kind, action, q, utils.norm_vin(cv),
                  cp, loc, notes, rec, owner, reason, conf)


def save_scan(user, fleet, storage, repo, required_actions, mode, kind, action, query, vin, plate, loc,
              notes, rec, owner, reason, conf):
    ss = st.session_state
    raw = ss.get('photo_raw')
    err = utils.validate_save(action, bool(raw), required_actions)
    if err:
        st.error(err)
        return
    if repo is None:
        st.error('Scan log is temporarily unavailable.')
        return
    if vin and vin != rec['_vin']:  # the confirmed VIN decides ownership, not raw OCR
        owner = fr.owner_for_vin(fleet, vin) or owner
    photo = {'photo_file_name': '', 'photo_folder': '', 'photo_url': '',
             'photo_status': 'not_attempted' if storage.configured else 'not_stored'}
    must = utils.photo_required(action, required_actions)
    if raw and (must or storage.configured):
        h = utils.image_hash(raw)
        done = ss.setdefault('uploads', {})
        if h not in done:  # an earlier successful upload is reused on retry (no duplicates)
            try:
                jpg = utils.prepare_photo(raw)
            except ValueError:
                st.error('Camera image could not be processed.')
                return
            when = utils.now_dubai()
            name = utils.photo_file_name(when, vin, plate, kind)
            with st.spinner('Uploading photo...'), utils.timed('photo_upload'):
                res = storage.upload_photo(user['username'], name, jpg, {'when': when})
            if res.ok:
                done[h] = res
            elif must:
                st.error(('Photo storage not configured. ' if 'not configured' in res.message else '')
                         + 'Photo upload failed. The stock count was not saved. Press Confirm and save to retry; your photo is kept.')
                return
            else:
                st.warning('Photo could not be stored; the scan was saved without it.')
                photo['photo_status'] = 'failed'
        if h in done:
            r = done[h]
            photo = {'photo_file_name': r.file_name, 'photo_folder': r.folder_path,
                     'photo_url': r.url, 'photo_status': 'uploaded'}
    record = {
        'scan_id': uuid.uuid4().hex, 'username': user['username'], 'display_name': user['display_name'],
        'login_time': ss.get('login_time', ''), 'scanned_on': utils.now_dubai().strftime('%Y-%m-%d %H:%M:%S'),
        'scanned_by': user['display_name'], 'action': action, 'scan_type': mode, 'scanned_value': query,
        'confirmed_vin': vin, 'confirmed_plate': plate, 'owner': owner, 'unit': rec['_unit'],
        'make': rec['_make'], 'model': rec['_model'], 'year': rec['_year'], 'branch': rec['_branch'],
        'location': loc, 'status': rec['_status'], 'match_method': reason, 'match_confidence': conf,
        'ocr_confidence': ss.get('ocr_conf'), 'source_file': rec['_file'], 'notes': notes, **photo}
    try:
        with utils.timed('db_save'):
            repo.add(record)
    except Exception:
        log.exception('scan save failed')
        st.error('Scan log is temporarily unavailable. Press Confirm and save to retry; the photo will not be uploaded twice.')
        return
    ss['flash'] = 'Saved. Ready for the next vehicle.'
    reset_scan()
    st.rerun()


# ------------------------------------------------------------------ history / admin
def history_tab(repo, user, everyone=False):
    key = 'all' if everyone else 'mine'
    if everyone and user['role'] != 'admin':
        return
    if repo is None:
        st.error('Scan log is temporarily unavailable.')
        return
    st.caption('History is loaded on request. SQLite history is not permanent on Streamlit Community Cloud.')
    if st.button('Load history', key=f'load_{key}', width='stretch'):
        st.session_state[f'show_{key}'] = True
    if not st.session_state.get(f'show_{key}'):
        return
    try:
        df = repo.history(None if everyone else user['username'])
    except Exception:
        log.exception('history failed')
        st.error('Scan log is temporarily unavailable.')
        return
    if everyone and not df.empty:
        who = st.selectbox('Employee', ['All'] + sorted(df['username'].dropna().unique().tolist()), key='emp_filter')
        if who != 'All':
            df = df[df['username'] == who]
    term = st.text_input('Search history', key=f'term_{key}')
    if term and not df.empty:
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask |= df[c].fillna('').astype(str).str.contains(re.escape(term), case=False, na=False)
        df = df[mask]
    st.dataframe(df, width='stretch', hide_index=True)
    if not df.empty:
        st.download_button('Download Excel', to_excel(df), f'fleet_scan_{key}.xlsx', width='stretch', key=f'dl_{key}')


def status_tab(fleet, ferr, storage):
    st.write(('✅' if not ferr else '❌') + f' **Fleet index:** {INDEX_FILE} ({len(fleet):,} vehicle rows)')
    if ferr:
        st.error(ferr)
    if storage.configured:
        st.write(f'✅ **Photo storage:** {storage.name}')
        if not storage.persistent:
            st.warning('Local photo storage is for development only and is lost on Streamlit Cloud restarts.')
    else:
        st.write('📷 **Photo storage is off.** Scan records are saved without photos.')
    st.write('🤖 **AI backup reader:** ' + ('on' if ai_cfg else 'off'))
    st.caption('Ownership: Moov when a VIN exists in both Moov and Hertz; otherwise Hertz, then AFF.')


def admin_tab(fleet, ferr, storage, repo, users):
    st.write(f'**Version:** {utils.APP_VERSION}')
    st.write(('✅' if not ferr else '❌') + f' Fleet index loaded - {len(fleet):,} rows')
    if not fleet.empty:
        st.write(fleet['_owner'].value_counts().to_dict())
    st.write(('✅' if ir.ocr_available() else '❌') + ' Tesseract OCR executable')
    st.write(('✅' if ir.barcode_available() else '❌') + ' Barcode library (zxing-cpp)')
    st.write(('✅' if storage.configured else '⚠️') + f' Storage provider: {storage.name}')
    if st.button('Run storage and database health checks', width='stretch'):
        ok, msg = storage.health_check()
        st.write(('✅ ' if ok else '❌ ') + 'Storage: ' + msg)
        ok, msg = repo.health() if repo else (False, 'Scan log unavailable')
        st.write(('✅ ' if ok else '❌ ') + 'Database: ' + msg)
    st.write('**Users:** ' + ', '.join(f"{u['username']} ({u['role']}{'' if u['active'] else ', inactive'})" for u in users.values()))


# ------------------------------------------------------------------ main
secrets = secrets_dict()
users = auth.load_users(secrets)
ai_cfg = ai_reader.settings(secrets)
if not auth.is_authenticated(st.session_state):
    login_page(users)
    st.stop()

user = {k: st.session_state[k] for k in ('username', 'display_name', 'role')}
fleet, ferr = cached_fleet(os.path.join(HERE, INDEX_FILE))
storage, repo = get_storage(), get_repo()
required_actions = (set(secrets.get('storage', {}).get('photo_required_actions', [])) | {'Stock Count'}
                    if storage.configured else set())

fleet_txt = f'🟢 Fleet {len(fleet):,}' if not ferr else '🔴 Fleet index missing'
photo_txt = '📷 Photos ready' if storage.configured else '📷 Photos off (scan log only)'
st.markdown(f'<div class="hero"><h1>Fleet Scan</h1><p>{html.escape(user["display_name"])} · {user["role"]}'
            f'<br>{fleet_txt} · {photo_txt}</p></div>', unsafe_allow_html=True)
if st.button('Logout', width='stretch'):
    auth.logout(st.session_state)
    st.rerun()
if ferr:
    st.error(ferr)

names = ['Scan', 'My History', 'Status'] + (['All History', 'Administration'] if user['role'] == 'admin' else [])
tabs = st.tabs(names)
with tabs[0]:
    scan_tab(user, fleet, storage, repo, required_actions, ai_cfg)
with tabs[1]:
    history_tab(repo, user)
with tabs[2]:
    status_tab(fleet, ferr, storage)
if user['role'] == 'admin':
    with tabs[3]:
        history_tab(repo, user, everyone=True)
    with tabs[4]:
        admin_tab(fleet, ferr, storage, repo, users)
