import os
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from models import db, Product, Stock, SaleDetail, Sale, Review
from cache_utils import save_joblib_atomic

product_analysis_bp = Blueprint('product_analysis', __name__)

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_product_data():
    products = Product.query.filter_by(deleted_at=None, status='active').all()
    rows = []
    for p in products:
        active_stock = Stock.query.filter_by(
            fk_product_id=p.id, status='active', deleted_at=None
        ).first()

        total_sold = db.session.query(
            db.func.coalesce(db.func.sum(SaleDetail.sale_stock), 0)
        ).join(Sale, Sale.id == SaleDetail.fk_sale_id).filter(
            SaleDetail.fk_stock_id == Stock.id,
            Stock.fk_product_id == p.id,
            Sale.status == 'completed',
            Sale.deleted_at == None,
            SaleDetail.deleted_at == None,
        ).scalar()

        total_revenue = db.session.query(
            db.func.coalesce(db.func.sum(SaleDetail.subtotal), 0)
        ).join(Sale, Sale.id == SaleDetail.fk_sale_id).filter(
            SaleDetail.fk_stock_id == Stock.id,
            Stock.fk_product_id == p.id,
            Sale.status == 'completed',
            Sale.deleted_at == None,
            SaleDetail.deleted_at == None,
        ).scalar()

        avg_rating = db.session.query(
            db.func.avg(Review.rating)
        ).filter(
            Review.fk_product_id == p.id,
            Review.deleted_at == None
        ).scalar() or 0

        review_count = Review.query.filter_by(
            fk_product_id=p.id, deleted_at=None
        ).count()

        stock_qty = active_stock.quantity if active_stock else 0
        sale_price = float(active_stock.sale_price) if active_stock else 0
        buy_price = float(active_stock.buy_price) if active_stock else 0

        rows.append({
            'product_id': p.id,
            'name': p.name,
            'sku': p.sku,
            'category': p.category.name if p.category else 'Uncategorized',
            'brand': p.brand.name if p.brand else 'Unknown',
            'price': sale_price,
            'buy_price': buy_price,
            'stock': stock_qty,
            'units_sold': int(total_sold),
            'revenue': float(total_revenue),
            'avg_rating': round(float(avg_rating), 2),
            'review_count': review_count,
        })

    return pd.DataFrame(rows)


def analyze_products(df):
    if df.empty or len(df) < 2:
        df['performance_score'] = 50.0
        df['performance_tier'] = 'Medium'
        return df

    df['stock_ratio'] = df['units_sold'] / (df['stock'] + df['units_sold'] + 1)
    df['sales_velocity'] = df['units_sold'] / (df['price'] + 1)
    df['profit_margin'] = np.where(
        df['buy_price'] > 0,
        (df['price'] - df['buy_price']) / df['buy_price'],
        0
    )

    features = ['revenue', 'stock_ratio', 'sales_velocity', 'profit_margin']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features].fillna(0))

    weights = np.array([0.4, 0.2, 0.2, 0.2])
    df['performance_score'] = np.round(np.dot(scaled, weights) * 100, 1)

    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled)

    tier_map = {}
    cluster_means = df.groupby('cluster')['performance_score'].mean().sort_values()
    for i, cluster_id in enumerate(cluster_means.index):
        tier_map[cluster_id] = ['Low', 'Medium', 'High'][min(i, 2)]
    df['performance_tier'] = df['cluster'].map(tier_map)

    try:
        save_joblib_atomic(scaler, 'product_scaler.joblib')
        save_joblib_atomic(kmeans, 'product_kmeans.joblib')
    except Exception:
        pass

    return df


@product_analysis_bp.route('/product-analysis', methods=['GET', 'POST'])
def analyze():
    try:
        df = fetch_product_data()
        if df.empty:
            return jsonify({
                'status': 'success',
                'data': [],
                'summary': {'total_products': 0},
                'message': 'No product data found'
            })

        df = analyze_products(df)

        summary = {
            'total_products': len(df),
            'avg_score': round(df['performance_score'].mean(), 1),
            'total_revenue': round(float(df['revenue'].sum()), 2),
            'top_product': df.loc[df['performance_score'].idxmax(), 'name'] if len(df) > 0 else None,
            'high_tier': int((df['performance_tier'] == 'High').sum()),
            'medium_tier': int((df['performance_tier'] == 'Medium').sum()),
            'low_tier': int((df['performance_tier'] == 'Low').sum()),
            'total_stock': int(df['stock'].sum()),
            'total_sold': int(df['units_sold'].sum()),
        }

        result = df.replace({np.nan: None}).to_dict('records')

        return jsonify({
            'status': 'success',
            'data': result,
            'summary': summary,
            'model_version': '2.0'
        })

    except Exception:
        current_app.logger.exception('product analysis failed')
        return jsonify({'status': 'error', 'message': 'internal error'}), 500
