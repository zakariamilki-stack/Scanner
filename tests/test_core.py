import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt
import pandas as pd
import pytest

import auth
import fleet_repository as fr
import photo_storage as ps
import utils
from scan_repository import SQLiteScanRepository

MOOV_VIN, AFF_VIN, HERTZ_VIN = 'MR2BF9C33P0023888', 'WDDGF8AB9CA654321', 'JTDKN3DU5A0123456'


@pytest.fixture
def fleet():
    rows = [
        (MOOV_VIN, 'G 77926', 'M1', 'Moov'), (MOOV_VIN, 'G-77926', 'H1', 'Hertz'),
        (HERTZ_VIN, '5 24910', 'H2', 'Hertz'), (AFF_VIN, 'H81595', 'A1', 'AFF'),
    ]
    return fr.normalize_frame(pd.DataFrame(rows, columns=['vin', 'plate', 'unit', 'owner']))


def test_vin_normalization_and_validation():
    assert utils.norm_vin(' mr2-bf9c33 p0023888 ') == MOOV_VIN
    assert utils.is_valid_vin(MOOV_VIN)
    for bad in ('MR2BF9C33P002388I', 'MR2BF9C33P002388O', 'MR2BF9C33P002388Q', 'SHORT', ''):
        assert not utils.is_valid_vin(bad)


def test_plate_normalization():
    assert utils.norm_plate(' g-779 26 ') == 'G77926'
    assert utils.norm_plate('5 24910') == '524910'


def test_username_sanitization():
    assert utils.sanitize_username('Employee 001') == 'employee_001'
    assert '.' not in utils.sanitize_username('../../etc/passwd') and '/' not in utils.sanitize_username('../a/b')
    with pytest.raises(ValueError):
        utils.sanitize_username('../..')


def test_folder_reuse_across_sessions(tmp_path):
    a = ps.LocalPhotoStorageProvider('Fleet Scan Photos', str(tmp_path))
    first = a.ensure_user_folder('Employee001')
    second = ps.LocalPhotoStorageProvider('Fleet Scan Photos', str(tmp_path)).ensure_user_folder('employee001')
    assert first == second == 'Fleet Scan Photos/employee001'
    assert os.listdir(tmp_path / 'Fleet Scan Photos') == ['employee001']
    res = a.upload_photo('employee001', '../evil.jpg', b'x', {})
    assert res.ok and res.file_name == 'evil.jpg' and os.path.exists(res.url)


def test_unconfigured_storage_never_reports_success():
    s = ps.build_storage({'storage': {'provider': 'sharepoint'}, 'graph': {}})
    assert not s.configured and not s.upload_photo('u1', 'a.jpg', b'x').ok


def test_moov_priority(fleet):
    assert fr.find_candidates(fleet, [MOOV_VIN], 'VIN')[0]['owner'] == 'Moov'
    assert fr.find_candidates(fleet, ['G77926'], 'Plate')[0]['owner'] == 'Moov'
    assert fr.find_candidates(fleet, [HERTZ_VIN], 'VIN')[0]['owner'] == 'Hertz'
    assert fr.find_candidates(fleet, [AFF_VIN], 'VIN')[0]['owner'] == 'AFF'
    assert fr.owner_for_vin(fleet, MOOV_VIN) == 'Moov'


def test_exact_vin(fleet):
    c = fr.find_candidates(fleet, [MOOV_VIN], 'VIN')
    assert len(c) == 1 and c[0]['reason'] == 'Exact VIN' and c[0]['confidence'] == 1.0


def test_last_eight_vin(fleet):
    c = fr.find_candidates(fleet, ['ZZZZZZZZZ' + HERTZ_VIN[-8:]], 'VIN')
    assert c[0]['reason'] == 'Unique VIN last-eight match' and c[0]['record']['_vin'] == HERTZ_VIN


def test_plate_numeric(fleet):
    c = fr.find_candidates(fleet, ['24910'], 'Plate')
    assert c[0]['reason'] == 'Numeric plate match' and c[0]['record']['_unit'] == 'H2' and c[0]['confidence'] >= .8
    assert fr.find_candidates(fleet, ['H81595'], 'Plate')[0]['reason'] == 'Exact plate'


def test_confidence_scoring():
    assert fr.confidence(MOOV_VIN, MOOV_VIN) >= .95
    assert fr.confidence('MR2BF9C33P0023889', MOOV_VIN) < fr.confidence(MOOV_VIN, MOOV_VIN)
    assert fr.confidence('AAAAAAAAAAAAAAAAA', MOOV_VIN) < .8
    assert fr.band(.96) == 'strong' and fr.band(.85) == 'likely' and fr.band(.5) == 'low'


def test_db_schema_migration(tmp_path):
    path = str(tmp_path / 'old.db')
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE scan_log (id INTEGER PRIMARY KEY, scanned_on TEXT, scanned_by TEXT)')
    conn.execute("INSERT INTO scan_log(scanned_on, scanned_by) VALUES ('2026-01-01', 'Old User')")
    conn.commit(); conn.close()
    repo = SQLiteScanRepository(path)
    repo.add({'username': 'employee001', 'scanned_by': 'E1', 'action': 'Stock Count', 'bogus': 'ignored'})
    repo.add({'username': 'employee002', 'action': 'Other'})
    assert len(repo.history()) == 3
    assert list(repo.history('employee001')['username']) == ['employee001']
    assert 'photo_url' in repo.history().columns and repo.health()[0]


def test_file_name_generation():
    when = datetime(2026, 11, 19, 9, 15, 30, tzinfo=utils.TZ)
    assert utils.photo_file_name(when, plate=' 24910', kind='Plate') == '20261119_091530_24910_PLATE.jpg'
    assert utils.photo_file_name(when, vin=MOOV_VIN, kind='VIN label') == f'20261119_091530_{MOOV_VIN}_VIN.jpg'
    assert utils.photo_file_name(when).endswith('_UNKNOWN_PLATE.jpg')


def test_photo_required_for_stock_count():
    assert utils.validate_save('Stock Count', False)
    assert utils.validate_save('Stock Count', True) == ''
    assert utils.validate_save('Vehicle Moved', False) == ''
    assert utils.validate_save('Vehicle Moved', False, ['Vehicle Moved'])


def test_auth():
    h = bcrypt.hashpw(b'secret', bcrypt.gensalt(rounds=4)).decode()
    users = auth.load_users({'users': {'Zakaria': {'display_name': 'Z', 'password_hash': h, 'role': 'admin'},
                                       'gone': {'password_hash': h, 'active': False}}})
    assert auth.authenticate('zakaria', 'secret', users).user['role'] == 'admin'
    assert auth.authenticate('zakaria', 'bad', users).user is None
    assert auth.authenticate('gone', 'secret', users).user is None
    assert auth.authenticate('nobody', 'x', users).message == 'Invalid username or password.'
