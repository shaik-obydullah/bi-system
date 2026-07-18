import numpy as np
import pandas as pd
from flask import Blueprint, jsonify
from flask_cors import cross_origin
from mlxtend.frequent_patterns import apriori, association_rules
from models import db, Sale, SaleDetail, Stock

combo_bp = Blueprint('product_combos', __name__)


def fetch_transaction_baskets():
    sales = Sale.query.filter(
        Sale.status == 'completed',
        Sale.deleted_at == None,
    ).all()

    if not sales:
        return pd.DataFrame(), []

    sale_ids = [s.id for s in sales]

    details = SaleDetail.query.filter(
        SaleDetail.fk_sale_id.in_(sale_ids),
        SaleDetail.deleted_at == None,
    ).all()

    if not details:
        return pd.DataFrame(), []

    product_names = {}
    for d in details:
        if d.fk_stock_id not in product_names:
            stock = Stock.query.filter_by(
                id=d.fk_stock_id, deleted_at=None
            ).first()
            if stock and stock.fk_product_id:
                from models import Product
                product = Product.query.filter_by(
                    id=stock.fk_product_id, deleted_at=None
                ).first()
                if product:
                    product_names[d.fk_stock_id] = product.name
                else:
                    product_names[d.fk_stock_id] = d.stock_name
            else:
                product_names[d.fk_stock_id] = d.stock_name

    basket_data = {}
    for d in details:
        sale_id = d.fk_sale_id
        product_name = product_names.get(d.fk_stock_id, f'Product-{d.fk_stock_id}')
        if sale_id not in basket_data:
            basket_data[sale_id] = set()
        basket_data[sale_id].add(product_name)

    all_products = sorted(set(
        name for names in basket_data.values() for name in names
    ))

    rows = []
    for sale_id, products in basket_data.items():
        row = {p: (p in products) for p in all_products}
        rows.append(row)

    return pd.DataFrame(rows), all_products


def run_apriori(basket_df, all_products):
    if basket_df.empty or len(basket_df) < 2:
        return {
            'frequent_itemsets': [],
            'association_rules': [],
            'summary': {
                'total_transactions': 0,
                'total_products': len(all_products),
                'frequent_combos': 0,
                'rules_found': 0,
            }
        }

    min_support = max(0.01, 1 / len(basket_df))
    frequent = apriori(basket_df, min_support=min_support, use_colnames=True, max_len=3)

    if frequent.empty:
        return {
            'frequent_itemsets': [],
            'association_rules': [],
            'summary': {
                'total_transactions': len(basket_df),
                'total_products': len(all_products),
                'frequent_combos': 0,
                'rules_found': 0,
            }
        }

    frequent['itemsets'] = frequent['itemsets'].apply(lambda x: list(x))
    frequent['support'] = frequent['support'].round(4)

    rules_list = []
    if len(frequent) >= 2:
        try:
            rules = association_rules(frequent, metric='confidence', min_threshold=0.3, num_itemsets=len(frequent))
            rules = rules.sort_values('lift', ascending=False).head(50)

            for _, r in rules.iterrows():
                rules_list.append({
                    'antecedents': list(r['antecedents']),
                    'consequents': list(r['consequents']),
                    'support': round(float(r['support']), 4),
                    'confidence': round(float(r['confidence']), 4),
                    'lift': round(float(r['lift']), 4),
                })
        except Exception:
            pass

    itemsets_list = frequent.head(30).to_dict('records')

    return {
        'frequent_itemsets': itemsets_list,
        'association_rules': rules_list,
        'summary': {
            'total_transactions': len(basket_df),
            'total_products': len(all_products),
            'frequent_combos': len(frequent),
            'rules_found': len(rules_list),
            'min_support_used': round(min_support, 4),
        }
    }


@combo_bp.route('/product-combos', methods=['GET'])
@cross_origin()
def combos():
    try:
        basket_df, all_products = fetch_transaction_baskets()
        result = run_apriori(basket_df, all_products)

        return jsonify({
            'status': 'success',
            'data': result,
            'model_version': '2.0'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
