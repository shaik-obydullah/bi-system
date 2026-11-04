import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from models import db


def create_app():
    load_dotenv()
    app = Flask(__name__)

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
    CORS(app)

    from routes.employee_performance import employee_bp
    from routes.product_analysis import product_analysis_bp
    from routes.product_recommendations import recommendation_bp
    from routes.sales_forecast import sales_bp
    from routes.product_combos import combo_bp
    from routes.prophet_forecast import prophet_bp

    app.register_blueprint(employee_bp, url_prefix='/api')
    app.register_blueprint(product_analysis_bp, url_prefix='/api')
    app.register_blueprint(recommendation_bp, url_prefix='/api')
    app.register_blueprint(sales_bp, url_prefix='/api')
    app.register_blueprint(combo_bp, url_prefix='/api')
    app.register_blueprint(prophet_bp, url_prefix='/api')

    @app.route('/')
    def index():
        return {'status': 'ok', 'service': 'BI System', 'endpoints': [
            'POST /api/analyze (Employee Performance)',
            'POST /api/product-analysis (Product Analysis)',
            'POST /api/recommendations/products (Recommendations)',
            'POST /api/recommendations/train (Train Models)',
            'GET /api/sales-forecast (Linear Regression Forecast)',
            'GET /api/prophet-forecast (Facebook Prophet Forecast)',
            'GET /api/product-combos (Product Combos)',
        ]}

    return app
