import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Blueprint, jsonify
from flask_cors import cross_origin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from models import db, Transaction

sales_bp = Blueprint('sales_forecast', __name__)

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)


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
    }).rename(columns={'amount': 'total_sales'})
    monthly['month'] = monthly.index.strftime('%b %Y')
    monthly['period'] = monthly.index.strftime('%Y-%m')
    monthly = monthly.reset_index()
    return monthly


def calculate_forecast(monthly_sales):
    if len(monthly_sales) < 2:
        return {
            'next_month_forecast': 0,
            'confidence_level': 0,
            'expected_range_low': 0,
            'expected_range_high': 0,
            'growth_rate': 0,
            'model_accuracy': 0,
            'history': monthly_sales.to_dict('records') if not monthly_sales.empty else [],
        }

    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y = monthly_sales['total_sales'].values

    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mean_y = np.mean(y_test)
        accuracy = max(0, 100 - (mae / mean_y) * 100) if mean_y > 0 else 0
        std_dev = float(np.std(y_test - y_pred))
    else:
        accuracy = 50
        std_dev = float(np.std(y_train - model.predict(X_train_scaled)))

    next_idx = np.array([[len(monthly_sales)]])
    next_scaled = scaler.transform(next_idx)
    forecast = float(model.predict(next_scaled)[0])
    forecast = max(0, forecast)

    if len(monthly_sales) >= 2:
        current = monthly_sales.iloc[-1]['total_sales']
        previous = monthly_sales.iloc[-2]['total_sales']
        growth_rate = ((current - previous) / previous) * 100 if previous > 0 else 0
    else:
        growth_rate = 0

    range_low = max(0, forecast - 1.5 * std_dev)
    range_high = forecast + 1.5 * std_dev
    confidence = min(90 + (len(monthly_sales) * 2), 95)

    try:
        joblib.dump({
            'model': model,
            'scaler': scaler
        }, os.path.join(MODEL_DIR, 'sales_forecast.joblib'))
    except Exception:
        pass

    return {
        'next_month_forecast': round(forecast, 2),
        'confidence_level': round(confidence, 2),
        'expected_range_low': round(range_low, 2),
        'expected_range_high': round(range_high, 2),
        'growth_rate': round(float(growth_rate), 2),
        'model_accuracy': round(float(accuracy), 2),
        'history': monthly_sales.to_dict('records'),
    }


@sales_bp.route('/sales-forecast', methods=['GET'])
@cross_origin()
def forecast():
    try:
        monthly = fetch_monthly_sales()

        if monthly.empty:
            return jsonify({
                'status': 'success',
                'data': {
                    'actual_sales': [],
                    'forecast': {
                        'next_month_forecast': 0,
                        'confidence_level': 0,
                        'expected_range_low': 0,
                        'expected_range_high': 0,
                        'growth_rate': 0,
                        'model_accuracy': 0,
                    }
                },
                'message': 'No sales data available',
                'model_version': '2.0'
            })

        forecast_result = calculate_forecast(monthly)

        summary = {
            'total_months': len(monthly),
            'total_revenue': round(float(monthly['total_sales'].sum()), 2),
            'avg_monthly': round(float(monthly['total_sales'].mean()), 2),
            'best_month': monthly.loc[monthly['total_sales'].idxmax(), 'month'] if len(monthly) > 0 else None,
            'worst_month': monthly.loc[monthly['total_sales'].idxmin(), 'month'] if len(monthly) > 0 else None,
        }

        return jsonify({
            'status': 'success',
            'data': {
                'actual_sales': forecast_result.pop('history', []),
                'forecast': forecast_result,
            },
            'summary': summary,
            'model_version': '2.0'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
