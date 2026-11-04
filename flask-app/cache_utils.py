"""Cache and model persistence helpers.

Forecast results are expensive to compute (Prophet refits the model on every
call), so they are persisted as JSON under ``CACHE_DIR`` and served until they
expire or are refreshed via ``POST /api/internal/refresh``. Files are written
atomically so concurrent gunicorn workers never read a half-written payload.
"""

import json
import os
import tempfile
import time

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(MODEL_DIR, 'cache'))

os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_key(key):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in key)


def _cache_path(key):
    return os.path.join(CACHE_DIR, f'{_safe_key(key)}.json')


def read_cache(key, max_age=None):
    """Return the cached payload for ``key`` or ``None`` if missing/expired."""
    path = _cache_path(key)
    try:
        if not os.path.exists(path):
            return None
        if max_age is not None and time.time() - os.path.getmtime(path) > max_age:
            return None
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def write_cache(key, payload):
    """Persist ``payload`` for ``key`` atomically. Returns True on success."""
    try:
        text = json.dumps(payload, default=str, allow_nan=True)
        _atomic_write(_cache_path(key), text)
        return True
    except Exception:
        return False


def cache_age(key):
    """Age of the cached entry in seconds, or ``None`` when absent."""
    try:
        return round(time.time() - os.path.getmtime(_cache_path(key)), 1)
    except OSError:
        return None


def save_joblib_atomic(obj, filename):
    """Write a joblib model without exposing a half-written file to readers."""
    import joblib

    path = os.path.join(MODEL_DIR, filename)
    os.makedirs(MODEL_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=MODEL_DIR, suffix='.tmp')
    os.close(fd)
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write(path, text):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
