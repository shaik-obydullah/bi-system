import os
import numpy as np
import pandas as pd
import joblib
from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from models import db, Product, Stock, SaleDetail, Sale, Review, User

try:
    from surprise import Dataset, Reader, KNNBasic
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

recommendation_bp = Blueprint('recommendations', __name__)

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

_models = {'knn': None, 'collab': None, 'scaler': None, 'trained': False}


def prepare_product_features():
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

        avg_rating = db.session.query(
            db.func.avg(Review.rating)
        ).filter(
            Review.fk_product_id == p.id,
            Review.deleted_at == None
        ).scalar() or 0

        review_count = Review.query.filter_by(
            fk_product_id=p.id, deleted_at=None
        ).count()

        rows.append({
            'product_id': p.id,
            'name': p.name,
            'price': float(active_stock.sale_price) if active_stock else 0,
            'avg_rating': float(avg_rating),
            'review_count': review_count,
            'total_sales': int(total_sold),
            'popularity': int(total_sold) * 0.5 + float(avg_rating) * 2,
        })

    return pd.DataFrame(rows)


def prepare_ratings_data():
    sales_rows = db.session.query(
        SaleDetail.fk_stock_id,
        Sale.fk_user_id,
        db.func.count(SaleDetail.id).label('purchase_count')
    ).join(Sale, Sale.id == SaleDetail.fk_sale_id).filter(
        Sale.status == 'completed',
        Sale.deleted_at == None,
        SaleDetail.deleted_at == None,
    ).group_by(
        SaleDetail.fk_stock_id, Sale.fk_user_id
    ).all()

    review_rows = Review.query.filter_by(
        deleted_at=None, status='published'
    ).all()

    ratings = []
    seen = set()
    for r in review_rows:
        stock = Stock.query.filter_by(
            fk_product_id=r.fk_product_id, status='active', deleted_at=None
        ).first()
        if stock:
            key = (stock.id, r.fk_user_id)
            ratings.append({
                'stock_id': stock.id,
                'user_id': r.fk_user_id,
                'rating': r.rating,
            })
            seen.add(key)

    for sr in sales_rows:
        key = (sr.fk_stock_id, sr.fk_user_id)
        if key not in seen and sr.fk_user_id:
            ratings.append({
                'stock_id': sr.fk_stock_id,
                'user_id': sr.fk_user_id,
                'rating': min(4.0 + sr.purchase_count * 0.5, 5.0),
            })
            seen.add(key)

    return pd.DataFrame(ratings) if ratings else pd.DataFrame(
        columns=['stock_id', 'user_id', 'rating']
    )


def train_models():
    product_df = prepare_product_features()
    if product_df.empty:
        return {'status': 'error', 'message': 'No products found'}

    features = ['price', 'avg_rating', 'review_count', 'total_sales', 'popularity']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(product_df[features].fillna(0))

    n_neighbors = min(10, len(product_df) - 1)
    if n_neighbors < 1:
        n_neighbors = 1
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(scaled)

    collab_model = None
    if SURPRISE_AVAILABLE:
        ratings_df = prepare_ratings_data()
        if len(ratings_df) >= 5:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(ratings_df[['user_id', 'stock_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            sim_options = {'name': 'cosine', 'user_based': False}
            collab_model = KNNBasic(sim_options=sim_options)
            collab_model.fit(trainset)

    _models['knn'] = knn
    _models['collab'] = collab_model
    _models['scaler'] = scaler
    _models['product_df'] = product_df
    _models['trained'] = True

    try:
        joblib.dump({'knn': knn, 'scaler': scaler}, os.path.join(MODEL_DIR, 'recommendation.joblib'))
        if collab_model:
            joblib.dump(collab_model, os.path.join(MODEL_DIR, 'collab_filter.joblib'))
    except Exception:
        pass

    return {'status': 'success', 'products_indexed': len(product_df)}


def load_or_train():
    if _models['trained']:
        return True

    model_path = os.path.join(MODEL_DIR, 'recommendation.joblib')
    if os.path.exists(model_path):
        try:
            saved = joblib.load(model_path)
            _models['knn'] = saved['knn']
            _models['scaler'] = saved['scaler']
            _models['product_df'] = prepare_product_features()

            collab_path = os.path.join(MODEL_DIR, 'collab_filter.joblib')
            if os.path.exists(collab_path):
                _models['collab'] = joblib.load(collab_path)

            _models['trained'] = True
            return True
        except Exception:
            pass

    result = train_models()
    return result.get('status') == 'success'


def get_popular_products(limit=10):
    products = prepare_product_features()
    if products.empty:
        return []
    products = products.sort_values('popularity', ascending=False).head(limit)
    return products[['product_id', 'name', 'price', 'total_sales', 'avg_rating', 'popularity']].to_dict('records')


def get_trending_products(limit=10):
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=30)

    recent = db.session.query(
        SaleDetail.fk_stock_id,
        db.func.count(SaleDetail.id).label('count'),
        db.func.avg(Review.rating).label('avg_rating')
    ).join(Sale, Sale.id == SaleDetail.fk_sale_id).outerjoin(
        Review, Review.fk_product_id == Stock.fk_product_id
    ).filter(
        Sale.created_at >= cutoff,
        Sale.status == 'completed',
        Sale.deleted_at == None,
        SaleDetail.deleted_at == None,
    ).group_by(SaleDetail.fk_stock_id).order_by(
        db.desc('count')
    ).limit(limit).all()

    results = []
    for r in recent:
        stock = Stock.query.get(r.fk_stock_id)
        if stock and stock.fk_product_id:
            product = Product.query.get(stock.fk_product_id)
            if product:
                results.append({
                    'product_id': product.id,
                    'name': product.name,
                    'price': float(stock.sale_price) if stock.sale_price else 0,
                    'recent_sales': r.count,
                    'avg_rating': round(float(r.avg_rating or 0), 2),
                })
    return results


def get_content_based_recommendations(product_id, limit=10):
    if not _models['trained'] or _models['knn'] is None:
        return []

    product_df = _models['product_df']
    idx_match = product_df[product_df['product_id'] == product_id].index
    if len(idx_match) == 0:
        return []

    idx = idx_match[0]
    features = ['price', 'avg_rating', 'review_count', 'total_sales', 'popularity']
    scaled = _models['scaler'].transform(product_df.loc[idx:idx, features].fillna(0))
    distances, indices = _models['knn'].kneighbors(scaled)

    results = []
    for i, dist in zip(indices[0][1:limit+1], distances[0][1:limit+1]):
        row = product_df.iloc[i]
        results.append({
            'product_id': int(row['product_id']),
            'name': row['name'],
            'price': row['price'],
            'similarity': round(1 - float(dist), 3),
        })
    return results


def get_collaborative_recommendations(user_id, limit=10):
    if not _models['trained'] or _models['collab'] is None:
        return get_popular_products(limit)

    product_df = _models['product_df']
    stock_ids = Stock.query.filter(
        Stock.fk_product_id.in_(product_df['product_id'].tolist()),
        Stock.status == 'active',
        Stock.deleted_at == None,
    ).with_entities(Stock.id, Stock.fk_product_id).all()

    stock_map = {s.id: s.fk_product_id for s in stock_ids}
    predictions = []
    for stock_id, product_id in stock_map.items():
        try:
            pred = _models['collab'].predict(user_id, stock_id)
            predictions.append({
                'product_id': product_id,
                'estimated_rating': round(pred.est, 2),
            })
        except Exception:
            continue

    predictions.sort(key=lambda x: x['estimated_rating'], reverse=True)
    seen = set()
    results = []
    for p in predictions[:limit * 2]:
        if p['product_id'] not in seen:
            seen.add(p['product_id'])
            product = Product.query.get(p['product_id'])
            if product:
                results.append({
                    'product_id': product.id,
                    'name': product.name,
                    'estimated_rating': p['estimated_rating'],
                })
        if len(results) >= limit:
            break
    return results


def get_personalized_recommendations(user_id, limit=10):
    products = prepare_product_features()
    if products.empty:
        return []

    user_reviews = Review.query.filter_by(
        fk_user_id=user_id, deleted_at=None
    ).all()
    if not user_reviews:
        return get_popular_products(limit)

    avg_user_rating = np.mean([r.rating for r in user_reviews])
    products['rating_diff'] = abs(products['avg_rating'] - avg_user_rating)
    products['score'] = products['popularity'] * 0.6 + (5 - products['rating_diff']) * 0.4 * 20
    results = products.sort_values('score', ascending=False).head(limit)

    return results[['product_id', 'name', 'price', 'score', 'avg_rating']].to_dict('records')


@recommendation_bp.route('/recommendations/train', methods=['POST'])
@cross_origin()
def train():
    result = train_models()
    status_code = 200 if result.get('status') == 'success' else 400
    return jsonify(result), status_code


@recommendation_bp.route('/recommendations/products', methods=['POST'])
@cross_origin()
def recommend():
    try:
        data = request.get_json() or {}
        strategy = data.get('strategy', 'popular')
        product_id = data.get('product_id')
        user_id = data.get('user_id')
        limit = min(data.get('limit', 10), 50)

        load_or_train()

        if strategy == 'popular':
            results = get_popular_products(limit)
        elif strategy == 'trending':
            results = get_trending_products(limit)
        elif strategy == 'content_based' and product_id:
            results = get_content_based_recommendations(product_id, limit)
        elif strategy == 'collaborative' and user_id:
            results = get_collaborative_recommendations(user_id, limit)
        elif strategy == 'personalized' and user_id:
            results = get_personalized_recommendations(user_id, limit)
        elif strategy == 'similar_users' and user_id:
            results = get_popular_products(limit)
        else:
            results = get_popular_products(limit)

        return jsonify({
            'status': 'success',
            'strategy': strategy,
            'count': len(results),
            'data': results,
            'model_version': '2.0'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
