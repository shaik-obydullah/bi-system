"""Internal operational endpoints.

These are not meant for the dashboards. They are called by the ERP scheduler
(see ``POST /api/internal/refresh``) so that expensive forecast models are
computed off the request path instead of on every dashboard load.
"""

import threading
import time

from flask import Blueprint, jsonify, request, current_app

from cache_utils import cache_age
from routes import prophet_forecast, sales_forecast

internal_bp = Blueprint('internal', __name__)

_recommendation_train = None
_refresh_lock = threading.Lock()


def _train_recommendations():
    global _recommendation_train
    if _recommendation_train is None:
        from routes.product_recommendations import train_models
        _recommendation_train = train_models
    return _recommendation_train()


def _refresh_sales():
    payload = sales_forecast.refresh()
    detail = {'status': 'success' if payload.get('status') == 'success' else 'error'}
    if payload.get('message'):
        detail['message'] = payload['message']
    return detail


def _refresh_prophet():
    payload, status_code = prophet_forecast.refresh(periods=prophet_forecast.DEFAULT_PERIODS)
    detail = {'status': 'success' if status_code == 200 and payload.get('status') == 'success' else 'error'}
    if payload.get('message'):
        detail['message'] = payload['message']
    return detail


def _refresh_recommendations():
    payload = _train_recommendations()
    detail = {'status': payload.get('status', 'error')}
    if payload.get('message'):
        detail['message'] = payload['message']
    return detail


REFRESHERS = {
    'sales': _refresh_sales,
    'prophet': _refresh_prophet,
    'recommendations': _refresh_recommendations,
}


@internal_bp.route('/internal/refresh', methods=['POST'])
def refresh():
    """Recompute the expensive payloads and persist them to the cache dir."""
    raw_targets = request.args.get('targets', '')
    if raw_targets.strip():
        targets = [name.strip() for name in raw_targets.split(',') if name.strip()]
        unknown = [name for name in targets if name not in REFRESHERS]
        if unknown:
            return jsonify({
                'status': 'error',
                'message': f'unknown target(s): {unknown}, expected one of: {sorted(REFRESHERS)}'
            }), 400
    else:
        targets = list(REFRESHERS)

    if not _refresh_lock.acquire(blocking=False):
        return jsonify({
            'status': 'error',
            'message': 'a refresh is already in progress'
        }), 409

    started = time.time()
    results = {}
    try:
        for name in targets:
            target_started = time.time()
            try:
                detail = REFRESHERS[name]()
            except Exception:
                current_app.logger.exception('refresh target %s failed', name)
                detail = {'status': 'error', 'message': 'internal error'}
            detail['duration_seconds'] = round(time.time() - target_started, 2)
            results[name] = detail
    finally:
        _refresh_lock.release()

    succeeded = sum(1 for detail in results.values() if detail.get('status') == 'success')
    if succeeded == len(results):
        overall = 'success'
    elif succeeded:
        overall = 'partial'
    else:
        overall = 'error'

    body = {
        'status': overall,
        'refreshed': results,
        'duration_seconds': round(time.time() - started, 2),
    }
    return jsonify(body), 500 if overall == 'error' else 200


@internal_bp.route('/internal/cache-status', methods=['GET'])
def cache_status():
    """Report the age of each cached forecast payload."""
    keys = {
        'sales': 'sales_forecast',
        'prophet_default': prophet_forecast.cache_key(prophet_forecast.DEFAULT_PERIODS),
    }
    return jsonify({
        'status': 'success',
        'data': {
            name: {'key': key, 'age_seconds': cache_age(key)}
            for name, key in keys.items()
        },
        'model_version': '2.0',
    })
