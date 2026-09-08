from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_category_id = db.Column(db.BigInteger, db.ForeignKey('categories.id'))
    fk_brand_id = db.Column(db.BigInteger, db.ForeignKey('brands.id'))
    fk_supplier_id = db.Column(db.BigInteger, db.ForeignKey('suppliers.id'))
    name = db.Column(db.String(200), nullable=False)
    sku = db.Column(db.String(50))
    review_number = db.Column(db.SmallInteger)
    review_avg = db.Column(db.SmallInteger)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)

    category = db.relationship('Category', backref='products', lazy='joined')
    brand = db.relationship('Brand', backref='products', lazy='joined')
    stocks = db.relationship('Stock', backref='product', lazy='dynamic',
                             primaryjoin="and_(Product.id==Stock.fk_product_id, Stock.deleted_at.is_(None))")
    reviews = db.relationship('Review', backref='product', lazy='dynamic',
                              primaryjoin="and_(Product.id==Review.fk_product_id, Review.deleted_at.is_(None))")


class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_category_id = db.Column(db.BigInteger)
    name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)


class Brand(db.Model):
    __tablename__ = 'brands'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)


class Stock(db.Model):
    __tablename__ = 'stocks'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_product_id = db.Column(db.BigInteger, db.ForeignKey('products.id'))
    batch = db.Column(db.String(100))
    quantity = db.Column(db.Integer)
    buy_price = db.Column(db.Numeric(10, 2))
    sale_price = db.Column(db.Numeric(10, 2))
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)


class Sale(db.Model):
    __tablename__ = 'sales'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_user_id = db.Column(db.BigInteger)
    invoice_id = db.Column(db.String(30), unique=True)
    type = db.Column(db.String(20))
    net_price = db.Column(db.Numeric(10, 2))
    grand_total = db.Column(db.Numeric(10, 2))
    paid_amount = db.Column(db.Numeric(10, 2))
    buy_price = db.Column(db.Numeric(10, 2))
    sale_due = db.Column(db.Numeric(10, 2))
    status = db.Column(db.String(20))
    deleted_at = db.Column(db.DateTime)

    details = db.relationship('SaleDetail', backref='sale', lazy='dynamic',
                              primaryjoin="and_(Sale.id==SaleDetail.fk_sale_id, SaleDetail.deleted_at.is_(None))")


class SaleDetail(db.Model):
    __tablename__ = 'sale_details'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_sale_id = db.Column(db.BigInteger, db.ForeignKey('sales.id'))
    fk_stock_id = db.Column(db.BigInteger, db.ForeignKey('stocks.id'))
    stock_name = db.Column(db.String(200))
    sale_stock = db.Column(db.Integer)
    subtotal = db.Column(db.Numeric(10, 2))
    deleted_at = db.Column(db.DateTime)


class Review(db.Model):
    __tablename__ = 'reviews'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_product_id = db.Column(db.BigInteger, db.ForeignKey('products.id'))
    fk_user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'))
    rating = db.Column(db.SmallInteger, nullable=False)
    review = db.Column(db.Text)
    status = db.Column(db.String(20))
    deleted_at = db.Column(db.DateTime)


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True)


class Customer(db.Model):
    __tablename__ = 'customers'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    balance = db.Column(db.Numeric(10, 2), default=0)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)


class Employee(db.Model):
    __tablename__ = 'employees'
    id = db.Column(db.BigInteger, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True)
    job_title = db.Column(db.String(255), nullable=False)
    salary = db.Column(db.Numeric(8, 2), nullable=False)
    hire_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)

    tasks = db.relationship('TaskManagement', backref='employee', lazy='dynamic',
                            primaryjoin="and_(Employee.id==TaskManagement.fk_employee_id, TaskManagement.deleted_at.is_(None))")


class TaskManagement(db.Model):
    __tablename__ = 'task_management'
    id = db.Column(db.BigInteger, primary_key=True)
    fk_employee_id = db.Column(db.BigInteger, db.ForeignKey('employees.id'))
    task_name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20))
    start_date = db.Column(db.Date)
    due_date = db.Column(db.Date)
    deleted_at = db.Column(db.DateTime)


class Supplier(db.Model):
    __tablename__ = 'suppliers'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    balance = db.Column(db.Numeric(10, 2), default=0)
    status = db.Column(db.String(20), default='active')
    deleted_at = db.Column(db.DateTime)


class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.BigInteger, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    fk_reference_id = db.Column(db.BigInteger)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    paid_amount = db.Column(db.Numeric(10, 2))
    due_amount = db.Column(db.Numeric(10, 2))
    deleted_at = db.Column(db.DateTime)


class Expense(db.Model):
    __tablename__ = 'expenses'
    id = db.Column(db.BigInteger, primary_key=True)
    table_name = db.Column(db.String(100))
    fk_transaction_id = db.Column(db.BigInteger, db.ForeignKey('transactions.id'))
    description = db.Column(db.Text, nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    deleted_at = db.Column(db.DateTime)


class Income(db.Model):
    __tablename__ = 'incomes'
    id = db.Column(db.BigInteger, primary_key=True)
    table_name = db.Column(db.String(100))
    fk_transaction_id = db.Column(db.BigInteger, db.ForeignKey('transactions.id'))
    description = db.Column(db.Text, nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    deleted_at = db.Column(db.DateTime)


class Shipment(db.Model):
    __tablename__ = 'shipments'
    id = db.Column(db.BigInteger, primary_key=True)
    tracking_number = db.Column(db.String(255), unique=True)
    received_date = db.Column(db.Date)
    status = db.Column(db.String(20))
    deleted_at = db.Column(db.DateTime)
