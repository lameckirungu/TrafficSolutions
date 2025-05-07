import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

# Database configuration
DB_PATH = "traffic_data.db"

def init_db():
    """Initialize the database with necessary tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create traffic data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS traffic_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        intersection_id TEXT NOT NULL,
        lane_id TEXT NOT NULL,
        direction TEXT,
        vehicle_count REAL,
        avg_speed REAL,
        congestion_level REAL,
        wait_time REAL,
        is_green INTEGER,
        has_emergency INTEGER
    )
    ''')
    
    # Create signal states table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signal_states (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        intersection_id TEXT NOT NULL,
        phase INTEGER,
        state_data TEXT
    )
    ''')
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        prediction_timestamp TEXT NOT NULL,
        intersection_id TEXT NOT NULL,
        lane_id TEXT NOT NULL,
        predicted_congestion REAL,
        predicted_wait_time REAL,
        confidence REAL
    )
    ''')
    
    # Create performance metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        scenario_name TEXT,
        control_type TEXT,
        avg_wait_time REAL,
        throughput REAL,
        congestion_level REAL,
        emergency_response_time REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_simulation_data(traffic_data):
    """
    Save traffic simulation data to database.
    
    Args:
        traffic_data (DataFrame): Traffic data to save
    """
    if traffic_data.empty:
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # Convert DataFrame to SQL-compatible format
    df_to_save = traffic_data.copy()
    
    # Convert timestamp to string if it's a datetime
    if pd.api.types.is_datetime64_any_dtype(df_to_save['timestamp']):
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert boolean columns to integers
    if 'is_green' in df_to_save.columns:
        df_to_save['is_green'] = df_to_save['is_green'].astype(int)
    
    if 'has_emergency' in df_to_save.columns:
        df_to_save['has_emergency'] = df_to_save['has_emergency'].astype(int)
    
    # Save to database
    df_to_save.to_sql('traffic_data', conn, if_exists='append', index=False)
    
    conn.close()

def save_signal_state(signal_state, intersection_id):
    """
    Save signal state to database.
    
    Args:
        signal_state (dict): Signal state data
        intersection_id (str): Intersection identifier
    """
    if not signal_state:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert signal state to JSON
    state_json = json.dumps(signal_state)
    
    # Insert into database
    cursor.execute(
        '''
        INSERT INTO signal_states (timestamp, intersection_id, phase, state_data)
        VALUES (?, ?, ?, ?)
        ''',
        (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            intersection_id,
            signal_state.get('current_phase', 0),
            state_json
        )
    )
    
    conn.commit()
    conn.close()

def save_prediction(prediction_data):
    """
    Save prediction data to database.
    
    Args:
        prediction_data (DataFrame): Prediction data to save
    """
    if prediction_data.empty:
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # Convert DataFrame to SQL-compatible format
    df_to_save = prediction_data.copy()
    
    # Convert timestamp columns to strings
    if pd.api.types.is_datetime64_any_dtype(df_to_save['timestamp']):
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if pd.api.types.is_datetime64_any_dtype(df_to_save['prediction_timestamp']):
        df_to_save['prediction_timestamp'] = df_to_save['prediction_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select only needed columns
    columns_to_save = [
        'timestamp', 'prediction_timestamp', 'intersection_id', 'lane_id',
        'predicted_congestion', 'predicted_wait_time', 'confidence'
    ]
    
    # Save to database only columns that exist
    existing_columns = [col for col in columns_to_save if col in df_to_save.columns]
    df_to_save[existing_columns].to_sql('predictions', conn, if_exists='append', index=False)
    
    conn.close()

def save_performance_metrics(metrics, scenario_name, control_type):
    """
    Save performance metrics to database.
    
    Args:
        metrics (dict): Performance metrics data
        scenario_name (str): Name of the scenario
        control_type (str): Type of control (traditional or ML-based)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        '''
        INSERT INTO performance_metrics (
            timestamp, scenario_name, control_type, avg_wait_time, 
            throughput, congestion_level, emergency_response_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            scenario_name,
            control_type,
            metrics.get('avg_wait_time', 0),
            metrics.get('throughput', 0),
            metrics.get('congestion_level', 0),
            metrics.get('emergency_response_time', 0)
        )
    )
    
    conn.commit()
    conn.close()

def get_historical_data(time_limit=None):
    """
    Retrieve historical traffic data from database.
    
    Args:
        time_limit (str): Optional time limit (e.g., '1 hour', '30 minutes')
    
    Returns:
        DataFrame: Historical traffic data
    """
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM traffic_data"
    
    if time_limit:
        query += f" WHERE timestamp >= datetime('now', '-{time_limit}')"
    
    query += " ORDER BY timestamp"
    
    # Load data
    df = pd.read_sql_query(query, conn)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns and not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    conn.close()
    
    return df

def get_prediction_accuracy():
    """
    Calculate prediction accuracy by comparing historical predictions with actual data.
    
    Returns:
        dict: Prediction accuracy metrics
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Get predictions
    predictions = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp",
        conn
    )
    
    # Get actual data (using prediction_timestamp to match with actual)
    actual_data = pd.read_sql_query(
        "SELECT * FROM traffic_data ORDER BY timestamp",
        conn
    )
    
    conn.close()
    
    if predictions.empty or actual_data.empty:
        return {
            'congestion_accuracy': 0,
            'wait_time_accuracy': 0,
            'sample_size': 0
        }
    
    # Convert timestamps to datetime
    predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
    predictions['prediction_timestamp'] = pd.to_datetime(predictions['prediction_timestamp'])
    actual_data['timestamp'] = pd.to_datetime(actual_data['timestamp'])
    
    # Match predictions with actual data
    matched_data = []
    
    for _, pred_row in predictions.iterrows():
        # Find actual data points close to the prediction timestamp
        actual_at_time = actual_data[
            (actual_data['timestamp'] >= pred_row['prediction_timestamp'] - pd.Timedelta(seconds=30)) &
            (actual_data['timestamp'] <= pred_row['prediction_timestamp'] + pd.Timedelta(seconds=30)) &
            (actual_data['intersection_id'] == pred_row['intersection_id']) &
            (actual_data['lane_id'] == pred_row['lane_id'])
        ]
        
        if not actual_at_time.empty:
            # Use the closest actual data point
            actual_at_time['time_diff'] = abs(actual_at_time['timestamp'] - pred_row['prediction_timestamp'])
            closest_actual = actual_at_time.loc[actual_at_time['time_diff'].idxmin()]
            
            matched_data.append({
                'intersection_id': pred_row['intersection_id'],
                'lane_id': pred_row['lane_id'],
                'timestamp': pred_row['prediction_timestamp'],
                'predicted_congestion': pred_row['predicted_congestion'],
                'actual_congestion': closest_actual['congestion_level'],
                'predicted_wait_time': pred_row['predicted_wait_time'],
                'actual_wait_time': closest_actual['wait_time']
            })
    
    if not matched_data:
        return {
            'congestion_accuracy': 0,
            'wait_time_accuracy': 0,
            'sample_size': 0
        }
    
    # Create DataFrame with matched data
    matched_df = pd.DataFrame(matched_data)
    
    # Calculate mean absolute percentage error
    matched_df['congestion_error'] = abs(matched_df['predicted_congestion'] - matched_df['actual_congestion'])
    matched_df['wait_time_error'] = abs(matched_df['predicted_wait_time'] - matched_df['actual_wait_time'])
    
    # Calculate accuracy (1 - normalized error)
    congestion_accuracy = 1 - min(1, matched_df['congestion_error'].mean())
    
    wait_time_accuracy = 0
    if matched_df['actual_wait_time'].mean() > 0:
        wait_time_accuracy = 1 - min(1, matched_df['wait_time_error'].mean() / 
                                    max(1, matched_df['actual_wait_time'].mean()))
    
    return {
        'congestion_accuracy': congestion_accuracy * 100,
        'wait_time_accuracy': wait_time_accuracy * 100,
        'sample_size': len(matched_df)
    }

def clear_simulation_data():
    """Clear all simulation data from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tables = ['traffic_data', 'signal_states', 'predictions', 'performance_metrics']
    
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
    
    conn.commit()
    conn.close()
