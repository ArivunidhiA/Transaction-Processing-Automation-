import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
from sklearn.ensemble import IsolationForest
import random
import warnings
warnings.filterwarnings('ignore')

class TransactionDataGenerator:
    @staticmethod
    def generate_data(num_records=10000, start_date='2024-01-01'):
        """Generate sample transaction data"""
        np.random.seed(42)
        
        # Generate dates
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(minutes=x) for x in range(num_records)]
        
        # Generate account numbers
        def generate_account():
            prefix = random.choice(['ACC', 'SAV', 'CHK'])
            return f"{prefix}{random.randint(100000, 999999)}"
        
        # Create base dataset
        data = {
            'transaction_id': [f'TXN{i:010d}' for i in range(num_records)],
            'timestamp': dates,
            'account_from': [generate_account() for _ in range(num_records)],
            'account_to': [generate_account() for _ in range(num_records)],
            'transaction_type': np.random.choice(
                ['TRANSFER', 'PAYMENT', 'DEPOSIT', 'WITHDRAWAL'],
                size=num_records,
                p=[0.4, 0.3, 0.2, 0.1]
            )
        }
        
        # Generate amounts based on transaction type
        amounts = []
        for tx_type in data['transaction_type']:
            base_amount = {
                'TRANSFER': (5000, 2000),
                'PAYMENT': (1000, 500),
                'DEPOSIT': (2000, 1000),
                'WITHDRAWAL': (500, 200)
            }[tx_type]
            amount = np.random.normal(base_amount[0], base_amount[1])
            amounts.append(abs(round(amount, 2)))
        
        data['amount'] = amounts
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add processing times and risk scores
        df['processing_time'] = np.random.exponential(2, num_records)
        df['risk_score'] = df.apply(lambda row: min(
            70 + (row['amount'] / 1000) + 
            (20 if row['transaction_type'] == 'WITHDRAWAL' else 0),
            100
        ), axis=1)
        
        # Add status
        df['status'] = df.apply(lambda row: 
            'PENDING_REVIEW' if row['risk_score'] >= 70
            else 'SLA_BREACH' if row['processing_time'] > 5
            else 'APPROVED', axis=1
        )
        
        return df

class TransactionProcessor:
    def __init__(self, db_path='transactions.db'):
        """Initialize the transaction processor"""
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            filename='transaction_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Setup database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create transactions table
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
                    processing_time FLOAT
                )
            ''')
            
            # Create logs table
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
    
    def process_transaction(self, transaction):
        """Process a single transaction"""
        try:
            # Validate and calculate risk
            risk_score = min(
                70 + (transaction['amount'] / 1000) +
                (20 if transaction['transaction_type'] == 'WITHDRAWAL' else 0),
                100
            )
            
            # Determine status
            status = (
                'PENDING_REVIEW' if risk_score >= 70
                else 'APPROVED'
            )
            
            # Log transaction
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO transaction_logs 
                    (transaction_id, timestamp, action, status, message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    transaction['transaction_id'],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'PROCESS',
                    status,
                    f"Risk Score: {risk_score:.2f}"
                ))
                conn.commit()
            
            return {
                'transaction_id': transaction['transaction_id'],
                'status': status,
                'risk_score': risk_score
            }
            
        except Exception as e:
            self.logger.error(f"Transaction processing failed: {str(e)}")
            raise

    def get_metrics(self, start_date=None, end_date=None):
        """Get transaction metrics"""
        query = '''
            SELECT 
                COUNT(*) as total_transactions,
                AVG(processing_time) as avg_processing_time,
                AVG(risk_score) as avg_risk_score,
                SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) as approved_count
            FROM transactions
        '''
        
        if start_date and end_date:
            query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn).to_dict('records')[0]

def main():
    """Main function to demonstrate the system"""
    print("Starting Transaction Processing System...")
    
    # Generate sample data
    print("\nGenerating sample transaction data...")
    data = TransactionDataGenerator.generate_data(10000)
    
    # Initialize processor
    processor = TransactionProcessor()
    
    # Save data to database
    print("\nSaving to database...")
    with sqlite3.connect('transactions.db') as conn:
        data.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Process some transactions
    print("\nProcessing sample transactions...")
    sample_size = min(5, len(data))
    for _, transaction in data.head(sample_size).iterrows():
        result = processor.process_transaction(transaction.to_dict())
        print(f"Processed transaction {result['transaction_id']}: {result['status']}")
    
    # Get metrics
    print("\nCalculating metrics...")
    metrics = processor.get_metrics()
    print("\nTransaction Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
