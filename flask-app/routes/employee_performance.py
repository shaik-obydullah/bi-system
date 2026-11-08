import os
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, current_app
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from models import db, Employee, TaskManagement
from cache_utils import save_joblib_atomic

employee_bp = Blueprint('employee_performance', __name__)

MODEL_DIR = os.getenv('MODEL_DIR', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_employee_data():
    employees = Employee.query.filter_by(deleted_at=None, status='active').all()
    rows = []
    for emp in employees:
        tasks = TaskManagement.query.filter_by(
            fk_employee_id=emp.id, deleted_at=None
        ).all()
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == 'completed')
        in_progress = sum(1 for t in tasks if t.status == 'in_progress')
        pending = sum(1 for t in tasks if t.status == 'pending')
        canceled = sum(1 for t in tasks if t.status == 'canceled')

        due_tasks = [t for t in tasks if t.due_date and t.start_date]
        if due_tasks:
            avg_days = np.mean([(t.due_date - t.start_date).days for t in due_tasks])
        else:
            avg_days = 0

        completion_rate = completed / total if total > 0 else 0

        rows.append({
            'employee_id': emp.id,
            'name': f"{emp.first_name} {emp.last_name}",
            'job_title': emp.job_title,
            'total_tasks': total,
            'completed_tasks': completed,
            'in_progress_tasks': in_progress,
            'pending_tasks': pending,
            'canceled_tasks': canceled,
            'completion_rate': round(completion_rate, 3),
            'avg_task_days': round(avg_days, 1),
        })

    return pd.DataFrame(rows)


def analyze_performance(df):
    if df.empty or len(df) < 2:
        df['cluster'] = 0
        df['anomaly'] = False
        df['performance_score'] = 50.0
        df['performance_tier'] = 'Medium'
        return df

    features = ['completed_tasks', 'total_tasks', 'avg_task_days', 'completion_rate']
    X = df[features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    iso = IsolationForest(contamination=min(0.05, max(0.01, 1 / len(df))), random_state=42)
    df['anomaly'] = iso.fit_predict(X_scaled) == -1

    # Scale each signal independently to 0..1, then combine into a 0..100 score.
    # Reusing one scaler (and a 0..100 range) made the weighted sum negative and
    # then multiplied it by 100 again, producing scores like -1418 or 1860.
    days_scaler = MinMaxScaler(feature_range=(0, 1))
    tasks_scaler = MinMaxScaler(feature_range=(0, 1))
    days_scaled = days_scaler.fit_transform(df[['avg_task_days']].fillna(0)).flatten()
    tasks_scaled = tasks_scaler.fit_transform(df[['completed_tasks']].fillna(0)).flatten()

    raw_scores = (
        0.4 * df['completion_rate'].fillna(0) +
        0.3 * (1 - days_scaled) +
        0.3 * tasks_scaled
    )
    df['performance_score'] = np.round(raw_scores * 100, 1)

    df['performance_tier'] = pd.cut(
        df['performance_score'],
        bins=[-1, 33, 66, 101],
        labels=['Low', 'Medium', 'High']
    )
    df['performance_tier'] = df['performance_tier'].astype(str)

    try:
        save_joblib_atomic(scaler, 'emp_scaler.joblib')
        save_joblib_atomic(kmeans, 'emp_kmeans.joblib')
        save_joblib_atomic(iso, 'emp_isolation.joblib')
    except Exception:
        pass

    return df


@employee_bp.route('/analyze', methods=['POST'])
def analyze():
    try:
        df = fetch_employee_data()
        if df.empty:
            return jsonify({
                'status': 'success',
                'data': [],
                'summary': {'total_employees': 0},
                'message': 'No employee data found'
            })

        df = analyze_performance(df)

        summary = {
            'total_employees': len(df),
            'avg_score': round(df['performance_score'].mean(), 1),
            'high_performers': int((df['performance_tier'] == 'High').sum()),
            'medium_performers': int((df['performance_tier'] == 'Medium').sum()),
            'low_performers': int((df['performance_tier'] == 'Low').sum()),
            'anomalies_detected': int(df['anomaly'].sum()),
            'avg_completion_rate': round(df['completion_rate'].mean(), 3),
        }

        result = df.replace({np.nan: None}).to_dict('records')

        return jsonify({
            'status': 'success',
            'data': result,
            'summary': summary,
            'model_version': '2.0'
        })

    except Exception:
        current_app.logger.exception('employee performance analysis failed')
        return jsonify({'status': 'error', 'message': 'internal error'}), 500
