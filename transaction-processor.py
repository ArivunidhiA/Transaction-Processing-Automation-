import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class TransactionProcessor:
    def __init__(self, db_path='transactions.db'):
        """Initialize the transaction processor with database connection"""
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()
        
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            filename='transaction_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Create necessary database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Transactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id TEXT PRIMARY KEY,
                        timestamp DATETIME,
                        amount FLOAT,
                        account_from TEXT,
                        account_to TEXT,
                        transaction_type TEXT,
                        status TEXT,
                        risk_score FLOAT,
                        processing_time FLOAT,
                        validation_status TEXT
                    )
                ''')
                
                # Transaction logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transaction_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id TEXT,
                        timestamp DATETIME,
                        action TEXT,
                        status TEXT,
                        message TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database setup completed successfully")
        except Exception as e:
            self.logger.error(f"Database setup failed: {str(e)}")
            raise
            
    def validate_transaction(self, transaction):
        """
        Validate transaction according to business rules
        Returns: (is_valid, risk_score, validation_message)
        """
        risk_score = 0
        validation_errors = []
        
        # Check transaction amount limits
        if transaction['amount'] <= 0:
            validation_errors.append("Invalid amount: Must be greater than 0")
        elif transaction['amount'] > 1000000:
            risk_score += 50
            validation_errors.append("High-value transaction: Additional approval required")
            
        # Check account format
        if not (transaction['account_from'].startswith('ACC') and 
                transaction['account_to'].startswith('ACC')):
            validation_errors.append("Invalid account format")
            
        # Check transaction type
        valid_types = ['TRANSFER', 'PAYMENT', 'DEPOSIT', 'WITHDRAWAL']
        if transaction['transaction_type'] not in valid_types:
            validation_errors.append("Invalid transaction type")
            
        # Calculate final risk score based on amount and type
        risk_score += min(transaction['amount'] / 10000, 30)
        if transaction['transaction_type'] in ['WITHDRAWAL', 'TRANSFER']:
            risk_score += 10
            
        is_valid = len(validation_errors) == 0
        return is_valid, risk_score, '; '.join(validation_errors) if validation_errors else "Valid"
    
    def process_transaction(self, transaction):
        """Process a single transaction with validation and logging"""
        start_time = datetime.now()
        transaction_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # Validate transaction
            is_valid, risk_score, validation_message = self.validate_transaction(transaction)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Determine status based on validation and risk score
            if is_valid:
                status = 'APPROVED' if risk_score < 70 else 'PENDING_REVIEW'
            else:
                status = 'REJECTED'
            
            # Store transaction
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert transaction record
                cursor.execute('''
                    INSERT INTO transactions (
                        transaction_id, timestamp, amount, account_from, account_to,
                        transaction_type, status, risk_score, processing_time, validation_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    transaction['amount'],
                    transaction['account_from'],
                    transaction['account_to'],
                    transaction['transaction_type'],
                    status,
                    risk_score,
                    processing_time,
                    validation_message
                ))
                
                # Log the transaction
                cursor.execute('''
                    INSERT INTO transaction_logs (transaction_id, timestamp, action, status, message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    transaction_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'PROCESS',
                    status,
                    f"Risk Score: {risk_score}; {validation_message}"
                ))
                
                conn.commit()
                
            self.logger.info(f"Transaction {transaction_id} processed: {status}")
            return {
                'transaction_id': transaction_id,
                'status': status,
                'risk_score': risk_score,
                'processing_time': processing_time,
                'message': validation_message
            }
            
        except Exception as e:
            self.logger.error(f"Transaction processing failed: {str(e)}")
            raise
            
    def get_transaction_metrics(self, start_date=None, end_date=None):
        """Calculate transaction processing metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT 
                        COUNT(*) as total_transactions,
                        AVG(processing_time) as avg_processing_time,
                        AVG(risk_score) as avg_risk_score,
                        SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) as approved_count,
                        SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
                        SUM(CASE WHEN status = 'PENDING_REVIEW' THEN 1 ELSE 0 END) as pending_count
                    FROM transactions
                '''
                
                if start_date and end_date:
                    query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
                
                metrics = pd.read_sql_query(query, conn)
                return metrics.to_dict('records')[0]
                
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
            raise
            
    def detect_anomalies(self, lookback_days=30):
        """Detect anomalous transactions using Isolation Forest"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent transactions
                query = f'''
                    SELECT amount, risk_score, processing_time
                    FROM transactions
                    WHERE timestamp >= datetime('now', '-{lookback_days} days')
                '''
                df = pd.read_sql_query(query, conn)
                
                if len(df) > 10:  # Need minimum samples for meaningful analysis
                    # Train anomaly detector
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = iso_forest.fit_predict(df)
                    
                    # Calculate anomaly percentage
                    anomaly_rate = (anomalies == -1).mean()
                    
                    return {
                        'anomaly_rate': anomaly_rate,
                        'total_transactions': len(df),
                        'anomalous_transactions': (anomalies == -1).sum()
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            raise
            
    def generate_sla_report(self, target_processing_time=5.0):
        """Generate SLA compliance report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT 
                        strftime('%Y-%m-%d', timestamp) as date,
                        COUNT(*) as total_transactions,
                        AVG(processing_time) as avg_processing_time,
                        SUM(CASE WHEN processing_time <= ? THEN 1 ELSE 0 END) as within_sla_count
                    FROM transactions
                    GROUP BY date
                    ORDER BY date DESC
                    LIMIT 30
                '''
                
                df = pd.read_sql_query(query, conn, params=[target_processing_time])
                df['sla_compliance_rate'] = df['within_sla_count'] / df['total_transactions'] * 100
                
                return df.to_dict('records')
                
        except Exception as e:
            self.logger.error(f"SLA report generation failed: {str(e)}")
            raise

def main():
    # Initialize processor
    processor = TransactionProcessor()
    
    # Example transaction
    transaction = {
        'amount': 5000.0,
        'account_from': 'ACC123456',
        'account_to': 'ACC789012',
        'transaction_type': 'TRANSFER'
    }
    
    # Process transaction
    result = processor.process_transaction(transaction)
    print(f"Transaction processed: {result}")
    
    # Get metrics
    metrics = processor.get_transaction_metrics()
    print(f"\nTransaction Metrics: {metrics}")
    
    # Check for anomalies
    anomalies = processor.detect_anomalies()
    if anomalies:
        print(f"\nAnomaly Detection Results: {anomalies}")
    
    # Generate SLA report
    sla_report = processor.generate_sla_report()
    print(f"\nSLA Report (last entry): {sla_report[0] if sla_report else 'No data'}")

if __name__ == "__main__":
    main()
