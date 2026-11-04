import os
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, current_app
from models import db, Transaction
from cache_utils import read_cache, write_cache, save_joblib_atomic

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

prophet_bp = Blueprint('prophet_forecast', __name__)

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_PERIODS = 12
CACHE_TTL = int(os.getenv('FORECAST_CACHE_TTL', '21600'))

_refresh_lock = threading.Lock()


def cache_key(periods):
    return f'prophet_forecast_{periods}'


def fetch_monthly_sales():
    income_types = ['saleIncome', 'income', 'miscIncome']
    transactions = Transaction.query.filter(
        Transaction.type.in_(income_types),
        Transaction.deleted_at == None,
    ).order_by(Transaction.date).all()

    if not transactions:
        return pd.DataFrame()

    rows = []
    for t in transactions:
        rows.append({
            'date': t.date,
            'amount': float(t.amount or 0),
        })

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    monthly = df.resample('M').agg({
        'amount': 'sum'
    }).rename(columns={'amount': 'y'})
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={'date': 'ds'})
    return monthly


def prophet_forecast(monthly_sales, periods=12):
    if not PROPHET_AVAILABLE:
        return None, 'Prophet not installed'

    if monthly_sales.empty or len(monthly_sales) < 2:
        return None, 'Need at least 2 months of data'

    df = monthly_sales[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna()
    df = df[df['y'] >= 0]

    if len(df) < 2:
        return None, 'Insufficient data points'

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast_df = model.predict(future)

    forecast_only = forecast_df[forecast_df['ds'] > df['ds'].max()].copy()

    historical_df = forecast_df[forecast_df['ds'] <= df['ds'].max()].copy()

    result = {
        'historical': [
            {
                'ds': row['ds'].strftime('%Y-%m-%d'),
                'y': round(float(row['yhat']), 2),
            }
            for _, row in historical_df.iterrows()
        ],
        'forecast': [
            {
                'ds': row['ds'].strftime('%Y-%m-%d'),
                'yhat': round(float(max(0, row['yhat'])), 2),
                'yhat_lower': round(float(max(0, row['yhat_lower'])), 2),
                'yhat_upper': round(float(row['yhat_upper']), 2),
            }
            for _, row in forecast_only.iterrows()
        ],
        'trend': [
            {
                'ds': row['ds'].strftime('%Y-%m-%d'),
                'trend': round(float(row['trend']), 2),
            }
            for _, row in forecast_df.iterrows()
        ],
    }

    forecast_values = [f['yhat'] for f in result['forecast']]
    historical_values = df['y'].tolist()

    if historical_values:
        last_month = historical_values[-1]
        avg_monthly = np.mean(historical_values)
        total_historical = sum(historical_values)
    else:
        last_month = 0
        avg_monthly = 0
        total_historical = 0

    if forecast_values:
        avg_forecast = np.mean(forecast_values)
        peak_forecast = max(forecast_values)
        peak_month_idx = forecast_values.index(peak_forecast)
        peak_month = result['forecast'][peak_month_idx]['ds']
    else:
        avg_forecast = 0
        peak_forecast = 0
        peak_month = None

    growth_rate = 0
    if len(historical_values) >= 2 and historical_values[-2] != 0:
        growth_rate = ((historical_values[-1] - historical_values[-2]) / historical_values[-2]) * 100

    forecast_growth = 0
    if last_month > 0 and forecast_values:
        forecast_growth = ((forecast_values[0] - last_month) / last_month) * 100

    try:
        save_joblib_atomic(model, 'prophet_model.joblib')
    except Exception:
        pass

    result['summary'] = {
        'total_historical_months': len(historical_values),
        'total_historical_revenue': round(total_historical, 2),
        'avg_monthly_revenue': round(avg_monthly, 2),
        'last_month_revenue': round(last_month, 2),
        'forecast_months': len(forecast_values),
        'avg_forecast': round(avg_forecast, 2),
        'peak_forecast': round(peak_forecast, 2),
        'peak_month': peak_month,
        'current_growth_rate': round(growth_rate, 2),
        'forecast_growth_rate': round(forecast_growth, 2),
        'model_components': ['trend', 'yearly_seasonality'],
    }

    return result, None


def build_payload(periods):
    """Return ``(payload, http_status)`` for the given forecast horizon."""
    monthly = fetch_monthly_sales()

    if monthly.empty:
        return {
            'status': 'success',
            'data': None,
            'message': 'No sales data available for forecasting',
            'model_version': '2.0'
        }, 200

    result, error = prophet_forecast(monthly, periods=periods)

    if error:
        return {
            'status': 'error',
            'message': error
        }, 400

    return {
        'status': 'success',
        'data': result,
        'model_version': '2.0'
    }, 200


def refresh(periods=DEFAULT_PERIODS):
    """Recompute and cache the forecast for ``periods`` months."""
    with _refresh_lock:
        payload, status_code = build_payload(periods)
        if status_code == 200 and payload.get('status') == 'success':
            write_cache(cache_key(periods), payload)
        return payload, status_code


@prophet_bp.route('/prophet-forecast', methods=['GET'])
def forecast():
    try:
        periods = request.args.get('periods', DEFAULT_PERIODS, type=int)
        periods = min(max(periods, 1), 60)

        cached = read_cache(cache_key(periods), max_age=CACHE_TTL)
        if cached is not None:
            return jsonify(cached)

        if not PROPHET_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Facebook Prophet is not installed'
            }), 503

        payload, status_code = refresh(periods)
        return jsonify(payload), status_code

    except Exception:
        current_app.logger.exception('prophet-forecast failed')
        return jsonify({'status': 'error', 'message': 'internal error'}), 500
