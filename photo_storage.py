"""Photo storage providers. SharePoint (Microsoft Graph) is the production provider."""
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

from utils import now_dubai, safe_file_name, safe_segment, sanitize_username

log = logging.getLogger('fleetscan')
GRAPH = 'https://graph.microsoft.com/v1.0'
_TOKEN = {}  # process-wide, in memory only (never in session state)


@dataclass
class UploadResult:
    ok: bool
    folder_path: str = ''
    file_name: str = ''
    url: str = ''
    message: str = ''  # safe to show to users


class PhotoStorageProvider:
    name = 'base'
    persistent = False
    configured = False

    def __init__(self, root_folder='Fleet Scan Photos'):
        self.root = safe_segment(root_folder, 'Fleet Scan Photos')

    def user_folder(self, username):
        return f'{self.root}/{sanitize_username(username)}'

    def day_folder(self, username, when=None):
        return f'{self.user_folder(username)}/{(when or now_dubai()):%Y-%m-%d}'

    def ensure_user_folder(self, username):
        raise NotImplementedError

    def upload_photo(self, username, file_name, image_bytes, metadata=None):
        raise NotImplementedError

    def get_photo_url(self, folder_path, file_name):
        return ''

    def health_check(self):
        raise NotImplementedError


class NotConfiguredProvider(PhotoStorageProvider):
    name = 'off (scan log only)'

    def ensure_user_folder(self, username):
        return self.user_folder(username)

    def upload_photo(self, username, file_name, image_bytes, metadata=None):
        return UploadResult(False, message='Photo storage not configured.')

    def health_check(self):
        return False, 'Photo storage not configured'


class LocalPhotoStorageProvider(PhotoStorageProvider):
    """Development only. NOT persistent on Streamlit Community Cloud."""
    name = 'local (development, nonpersistent on Streamlit Cloud)'
    configured = True

    def __init__(self, root_folder='Fleet Scan Photos', base_dir='local_photos'):
        super().__init__(root_folder)
        self.base_dir = base_dir

    def _path(self, rel):
        return os.path.join(self.base_dir, *rel.split('/'))

    def ensure_user_folder(self, username):
        rel = self.user_folder(username)
        os.makedirs(self._path(rel), exist_ok=True)
        return rel

    def upload_photo(self, username, file_name, image_bytes, metadata=None):
        try:
            rel = self.day_folder(username, (metadata or {}).get('when'))
            self.ensure_user_folder(username)
            os.makedirs(self._path(rel), exist_ok=True)
            name = safe_file_name(file_name)
            with open(os.path.join(self._path(rel), name), 'wb') as fh:
                fh.write(image_bytes)
            return UploadResult(True, rel, name, self.get_photo_url(rel, name))
        except Exception:
            log.exception('local upload failed')
            return UploadResult(False, message='Photo upload failed.')

    def get_photo_url(self, folder_path, file_name):
        return os.path.join(self._path(folder_path), file_name)

    def health_check(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            return os.access(self.base_dir, os.W_OK), 'Local storage writable (development only)'
        except Exception:
            return False, 'Local storage not writable'


class SharePointPhotoStorageProvider(PhotoStorageProvider):
    name = 'sharepoint'
    persistent = True
    configured = True

    def __init__(self, root_folder, graph_cfg, shared_folder_url=''):
        super().__init__(root_folder)
        self.cfg = graph_cfg
        self.shared_folder_url = shared_folder_url
        self._drive = graph_cfg.get('drive_id') or ''
        self._known = set()

    def _token(self):
        import requests
        key = self.cfg['client_id']
        t = _TOKEN.get(key)
        if t and t['exp'] > time.time() + 60:
            return t['value']
        r = requests.post(
            f"https://login.microsoftonline.com/{self.cfg['tenant_id']}/oauth2/v2.0/token",
            data={'grant_type': 'client_credentials', 'client_id': self.cfg['client_id'],
                  'client_secret': self.cfg['client_secret'],
                  'scope': 'https://graph.microsoft.com/.default'}, timeout=(5, 10))
        r.raise_for_status()
        j = r.json()
        _TOKEN[key] = {'value': j['access_token'], 'exp': time.time() + int(j.get('expires_in', 3000))}
        return _TOKEN[key]['value']

    def _req(self, method, path, timeout=(5, 20), **kw):
        import requests
        headers = {'Authorization': f'Bearer {self._token()}', **kw.pop('headers', {})}
        return requests.request(method, GRAPH + path, headers=headers, timeout=timeout, **kw)

    def _drive_id(self):
        if not self._drive:
            r = self._req('GET', f"/sites/{self.cfg['site_id']}/drive", timeout=(5, 10))
            r.raise_for_status()
            self._drive = r.json()['id']
        return self._drive

    def _ensure_path(self, path):
        cur = ''
        for seg in path.split('/'):
            parent, cur = cur, f'{cur}/{seg}' if cur else seg
            if cur in self._known:
                continue
            target = (f'/drives/{self._drive_id()}/root/children' if not parent else
                      f'/drives/{self._drive_id()}/root:/{quote(parent)}:/children')
            r = self._req('POST', target, json={'name': seg, 'folder': {},
                                                '@microsoft.graph.conflictBehavior': 'fail'})
            if r.status_code not in (200, 201, 409):  # 409 = already exists (reuse)
                r.raise_for_status()
            self._known.add(cur)

    def ensure_user_folder(self, username):
        path = self.user_folder(username)
        self._ensure_path(path)
        return path

    def upload_photo(self, username, file_name, image_bytes, metadata=None):
        try:
            folder = self.day_folder(username, (metadata or {}).get('when'))
            self._ensure_path(folder)
            name = safe_file_name(file_name)
            r = self._req('PUT', f"/drives/{self._drive_id()}/root:/{quote(folder + '/' + name)}:/content"
                          '?@microsoft.graph.conflictBehavior=rename',
                          timeout=(5, 25), data=image_bytes, headers={'Content-Type': 'image/jpeg'})
            r.raise_for_status()
            j = r.json()
            return UploadResult(True, folder, j.get('name', name), j.get('webUrl', ''))
        except Exception:
            log.exception('SharePoint upload failed')  # full detail only in logs
            return UploadResult(False, message='Photo upload failed.')

    def health_check(self):
        try:
            r = self._req('GET', f'/drives/{self._drive_id()}', timeout=(3, 5))
            return (r.status_code == 200), ('SharePoint reachable' if r.status_code == 200 else 'SharePoint rejected the request')
        except Exception:
            log.exception('SharePoint health check failed')
            return False, 'SharePoint not reachable'


def build_storage(secrets):
    cfg, graph = secrets.get('storage') or {}, secrets.get('graph') or {}
    provider = str(cfg.get('provider', '')).lower()
    root = cfg.get('root_folder', 'Fleet Scan Photos')
    if provider == 'local' and cfg.get('allow_local_dev'):
        return LocalPhotoStorageProvider(root, cfg.get('local_dir', 'local_photos'))
    if (provider == 'sharepoint' and all(graph.get(k) for k in ('tenant_id', 'client_id', 'client_secret'))
            and (graph.get('drive_id') or graph.get('site_id'))):
        return SharePointPhotoStorageProvider(root, graph, cfg.get('shared_folder_url', ''))
    return NotConfiguredProvider(root)
