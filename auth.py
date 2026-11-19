"""Login using bcrypt hashes stored in Streamlit secrets."""
import time
from dataclasses import dataclass

import bcrypt

from utils import now_dubai, sanitize_username

MAX_FAILS = 5
LOCK_SECONDS = 300
ROLES = {'admin', 'scanner'}
_FAILS = {}  # process-wide: username -> (count, first_failure_time)
_DUMMY = bcrypt.hashpw(b'dummy', bcrypt.gensalt(rounds=10))


@dataclass
class AuthResult:
    user: dict = None
    message: str = ''


def load_users(secrets):
    users = {}
    for key, v in (secrets.get('users') or {}).items():
        try:
            name = sanitize_username(key)
        except ValueError:
            continue
        role = str(v.get('role', 'scanner')).lower()
        users[name] = {
            'username': name,
            'display_name': str(v.get('display_name') or key),
            'password_hash': str(v.get('password_hash', '')),
            'active': bool(v.get('active', True)),
            'role': role if role in ROLES else 'scanner',
        }
    return users


def verify_password(password, password_hash):
    try:
        return bcrypt.checkpw(password.encode()[:72], password_hash.encode())
    except (ValueError, TypeError, AttributeError):
        return False


def _locked(name):
    count, first = _FAILS.get(name, (0, 0))
    if count >= MAX_FAILS and time.time() - first < LOCK_SECONDS:
        return True
    if count and time.time() - first >= LOCK_SECONDS:
        _FAILS.pop(name, None)
    return False


def authenticate(username, password, users):
    try:
        name = sanitize_username(username)
    except ValueError:
        name = ''
    if name and _locked(name):
        return AuthResult(None, 'Too many failed attempts. Try again in a few minutes.')
    user = users.get(name)
    ok = verify_password(password or '', user['password_hash'] if user else _DUMMY.decode())
    if user and user['active'] and ok:
        _FAILS.pop(name, None)
        return AuthResult({k: user[k] for k in ('username', 'display_name', 'role')}, '')
    if name:
        count, first = _FAILS.get(name, (0, time.time()))
        _FAILS[name] = (count + 1, first)
    return AuthResult(None, 'Invalid username or password.')


def login(state, user):
    state['authenticated'] = True
    state['username'] = user['username']
    state['display_name'] = user['display_name']
    state['role'] = user['role']
    state['login_time'] = now_dubai().strftime('%Y-%m-%d %H:%M:%S')


def logout(state):
    state.clear()


def is_authenticated(state):
    return bool(state.get('authenticated')) and bool(state.get('username'))
