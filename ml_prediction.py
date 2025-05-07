import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta

class TrafficPredictor:
    """
    Implements machine learning models for traffic prediction.
    """
    
    def __init__(self):
        """Initialize the traffic predictor with default settings."""
        self.congestion_model = None
        self.wait_time_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'vehicle_count', 'avg_speed', 'wait_time', 'congestion_level',
            'hour', 'minute', 'is_rush_hour', 'is_weekend'
        ]
        self.target_columns = ['congestion_level', 'wait_time']
        self.prediction_window = 5  # minutes
        self.historical_data = pd.DataFrame()
        self.model_type = "random_forest"  # or "linear"
        self.trained = False
    
    def load_model(self, model_config):
        """
        Load or initialize ML models based on configuration.
        
        Args:
            model_config (dict): Model configuration parameters
        """
        self.model_type = model_config.get('type', 'random_forest')
        self.prediction_window = model_config.get('prediction_window', 5)
        
        # Initialize models based on type
        if self.model_type == 'random_forest':
            self.congestion_model = RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 50),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
            self.wait_time_model = RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 50),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
        elif self.model_type == 'linear':
            self.congestion_model = LinearRegression()
            self.wait_time_model = LinearRegression()
        
        # Reset historical data
        self.historical_data = pd.DataFrame()
        self.trained = False
    
    def _prepare_features(self, data):
        """
        Prepare feature data for prediction.
        
        Args:
            data (DataFrame): Raw traffic data
        
        Returns:
            DataFrame: Processed features for ML models
        """
        if data.empty:
            return pd.DataFrame()
        
        # Copy to avoid modifying original
        df = data.copy()
        
        # Extract time features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Define rush hours (8-9 AM and 5-6 PM)
            rush_hours_morning = (df['hour'] >= 8) & (df['hour'] < 9)
            rush_hours_evening = (df['hour'] >= 17) & (df['hour'] < 18)
            df['is_rush_hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
            
            # Weekend flag
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # One-hot encode direction
        if 'direction' in df.columns:
            direction_dummies = pd.get_dummies(df['direction'], prefix='direction')
            df = pd.concat([df, direction_dummies], axis=1)
        
        # Create binary indicator for emergency vehicles
        if 'has_emergency' in df.columns:
            df['has_emergency'] = df['has_emergency'].astype(int)
        
        # Select features present in the data
        available_features = [col for col in self.feature_columns if col in df.columns]
        features = df[available_features]
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _train_models(self):
        """
        Train ML models on accumulated historical data.
        """
        if len(self.historical_data) < 10:  # Need minimum data points
            return False
        
        # Prepare features
        features = self._prepare_features(self.historical_data)
        
        # Prepare targets
        congestion_target = self.historical_data['congestion_level']
        wait_time_target = self.historical_data['wait_time']
        
        # Scale features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Train models
        self.congestion_model.fit(scaled_features, congestion_target)
        self.wait_time_model.fit(scaled_features, wait_time_target)
        
        self.trained = True
        return True
    
    def predict(self, data):
        """
        Make predictions based on current traffic data.
        
        Args:
            data (DataFrame): Current traffic data
        
        Returns:
            DataFrame: Predictions for future traffic conditions
        """
        if data.empty:
            return pd.DataFrame()
        
        # Accumulate historical data
        self.historical_data = pd.concat([self.historical_data, data])
        
        # Keep only recent data (last hour)
        if 'timestamp' in self.historical_data.columns:
            one_hour_ago = datetime.now() - timedelta(hours=1)
            self.historical_data = self.historical_data[
                self.historical_data['timestamp'] > pd.Timestamp(one_hour_ago)
            ]
        
        # Train models if not trained or periodically
        if not self.trained or len(self.historical_data) % 10 == 0:
            self._train_models()
        
        # If models aren't trained yet, return simple heuristic predictions
        if not self.trained:
            return self._heuristic_prediction(data)
        
        # Prepare features for prediction
        features = self._prepare_features(data)
        if features.empty:
            return pd.DataFrame()
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Make predictions
        congestion_pred = self.congestion_model.predict(scaled_features)
        wait_time_pred = self.wait_time_model.predict(scaled_features)
        
        # Create prediction DataFrame
        predictions = data[['intersection_id', 'lane_id', 'timestamp']].copy()
        predictions['predicted_congestion'] = congestion_pred
        predictions['predicted_wait_time'] = wait_time_pred
        
        # Add prediction timestamp
        predictions['prediction_timestamp'] = predictions['timestamp'] + timedelta(minutes=self.prediction_window)
        
        # Calculate prediction confidence (simplified)
        prediction_accuracy = min(0.8, 0.5 + (len(self.historical_data) / 1000))
        predictions['confidence'] = prediction_accuracy
        
        return predictions
    
    def _heuristic_prediction(self, data):
        """
        Generate simple heuristic predictions when ML models aren't ready.
        
        Args:
            data (DataFrame): Current traffic data
        
        Returns:
            DataFrame: Simple predictions based on current data
        """
        predictions = data[['intersection_id', 'lane_id', 'timestamp']].copy()
        
        # Simple heuristic: congestion will slightly increase, wait times follow current pattern
        predictions['predicted_congestion'] = data['congestion_level'] * 1.1
        predictions['predicted_congestion'] = predictions['predicted_congestion'].clip(0, 1)
        
        predictions['predicted_wait_time'] = data['wait_time'] * 1.05
        
        # Add prediction timestamp
        predictions['prediction_timestamp'] = predictions['timestamp'] + timedelta(minutes=self.prediction_window)
        
        # Low confidence for heuristic predictions
        predictions['confidence'] = 0.5
        
        return predictions
    
    def get_model_metrics(self):
        """
        Get model performance metrics.
        
        Returns:
            dict: Model performance metrics
        """
        if not self.trained:
            return {
                'model_status': 'Not trained yet',
                'accuracy': 0,
                'data_points': len(self.historical_data)
            }
        
        # Simple metrics for demo purposes
        num_features = len(self._prepare_features(self.historical_data).columns)
        
        return {
            'model_status': 'Trained',
            'model_type': self.model_type,
            'accuracy': min(0.8, 0.5 + (len(self.historical_data) / 1000)),  # Simplified accuracy metric
            'data_points': len(self.historical_data),
            'features_used': num_features,
            'prediction_window': f"{self.prediction_window} minutes"
        }
