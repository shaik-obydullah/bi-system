import hmac
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from models import db

# Paths that may be reached without an API key (used by container healthchecks).
PUBLIC_PATHS = {'/', '/health'}


def create_app():
    load_dotenv()
    app = Flask(__name__)

    api_key = os.getenv('BI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError(
            'BI_API_KEY is not set. The BI API refuses to start without an API key. '
            'Copy flask-app/.env.example to flask-app/.env and set a strong value.'
        )
    app.config['BI_API_KEY'] = api_key

    db_user = os.getenv('DB_USER', 'root')
    db_pass = os.getenv('DB_PASSWORD', 'root')
    db_host = os.getenv('DB_HOST', 'erp_mariadb')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME', 'laravel_db')

    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        "?charset=utf8mb4"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_recycle': 280,
        'pool_pre_ping': True,
    }

    db.init_app(app)

    @app.before_request
    def attach_request_id():
        g.request_id = request.headers.get('X-Request-ID') or uuid.uuid4().hex

    @app.before_request
    def require_api_key():
        if request.path in PUBLIC_PATHS:
            return None
        supplied = request.headers.get('X-API-Key', '')
        if not hmac.compare_digest(supplied.encode('utf-8'), app.config['BI_API_KEY'].encode('utf-8')):
            return jsonify({'status': 'error', 'message': 'unauthorized'}), 401
        return None

    @app.after_request
    def add_request_id_header(response):
        request_id = getattr(g, 'request_id', None)
        if request_id:
            response.headers['X-Request-ID'] = request_id
        return response

    @app.errorhandler(Exception)
    def handle_exception(error):
        if isinstance(error, HTTPException):
            return error
        app.logger.exception(
            'Unhandled exception [request_id=%s path=%s]',
            getattr(g, 'request_id', None), request.path,
        )
        return jsonify({
            'status': 'error',
            'message': 'internal error',
            'request_id': getattr(g, 'request_id', None),
        }), 500

    @app.errorhandler(404)
    def handle_404(error):
        return jsonify({'status': 'error', 'message': 'not found'}), 404

    @app.errorhandler(405)
    def handle_405(error):
        return jsonify({'status': 'error', 'message': 'method not allowed'}), 405

    from routes.employee_performance import employee_bp
    from routes.product_analysis import product_analysis_bp
    from routes.product_recommendations import recommendation_bp
    from routes.sales_forecast import sales_bp
    from routes.product_combos import combo_bp
    from routes.prophet_forecast import prophet_bp
    from routes.internal import internal_bp

    blueprints = [
        employee_bp,
        product_analysis_bp,
        recommendation_bp,
        sales_bp,
        combo_bp,
        prophet_bp,
        internal_bp,
    ]

    api_prefix = os.getenv('API_PREFIX', '/api').rstrip('/') or '/api'
    legacy_enabled = os.getenv('ENABLE_LEGACY_API', '1') != '0'

    for blueprint in blueprints:
        app.register_blueprint(blueprint, url_prefix=f'{api_prefix}/v1', name=f'{blueprint.name}_v1')
        if legacy_enabled:
            # Temporary alias so existing ERP callers keep working during rollout.
            app.register_blueprint(blueprint, url_prefix=api_prefix, name=f'{blueprint.name}_legacy')

    @app.route('/health')
    def health():
        checks = {}
        healthy = True

        try:
            db.session.execute(text('SELECT 1'))
            checks['database'] = 'ok'
        except Exception:
            app.logger.exception('health check: database unavailable')
            checks['database'] = 'error'
            healthy = False

        model_dir = os.getenv('MODEL_DIR', 'ml_models')
        try:
            os.makedirs(model_dir, exist_ok=True)
            checks['model_dir'] = 'ok' if os.access(model_dir, os.W_OK) else 'error'
        except OSError:
            checks['model_dir'] = 'error'
        if checks['model_dir'] != 'ok':
            healthy = False

        return jsonify({
            'status': 'ok' if healthy else 'error',
            'checks': checks,
            'model_version': '2.0',
        }), 200 if healthy else 503

    @app.route('/')
    def index():
        base = f'{api_prefix}/v1'
        return {
            'status': 'ok',
            'service': 'BI System',
            'auth': 'Send the X-API-Key header on every request except / and /health',
            'api_prefix': base,
            'legacy_prefix': api_prefix if legacy_enabled else None,
            'endpoints': [
                f'POST {base}/analyze (Employee Performance)',
                f'POST {base}/product-analysis (Product Analysis)',
                f'POST {base}/recommendations/products (Recommendations)',
                f'POST {base}/recommendations/train (Train Models)',
                f'GET {base}/sales-forecast (Linear Regression Forecast)',
                f'GET {base}/prophet-forecast (Facebook Prophet Forecast)',
                f'GET {base}/product-combos (Product Combos)',
                f'POST {base}/internal/refresh (Refresh cached forecasts)',
                f'GET {base}/internal/cache-status (Cache ages)',
                'GET /health (no API key)',
            ],
        }

    return app
