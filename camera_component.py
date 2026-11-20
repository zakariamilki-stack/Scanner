"""Live camera with a guide frame. The browser crops to the frame before sending the photo."""
import base64
import os

import streamlit.components.v1 as components

_component = components.declare_component(
    'frame_camera', path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frame_camera'))


def frame_camera(kind, reset, key='frame_camera'):
    """Return {'id': str, 'bytes': bytes} for the latest capture, or None."""
    value = _component(kind=kind, reset=reset, key=key, default=None)
    if not value or 'data' not in value:
        return None
    try:
        return {'id': value['id'], 'bytes': base64.b64decode(value['data'].split(',', 1)[1])}
    except (IndexError, ValueError):
        return None
