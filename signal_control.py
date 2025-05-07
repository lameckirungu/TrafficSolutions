import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SignalController:
    """
    Controls traffic signals using both traditional methods and ML-enhanced adaptive control.
    """
    
    def __init__(self):
        """Initialize the signal controller with default settings."""
        self.cycle_time = 60  # seconds
        self.min_green_time = 10  # seconds
        self.max_green_time = 45  # seconds
        self.yellow_time = 3  # seconds
        self.current_phase = {}  # {intersection_id: current_phase}
        self.phase_start_time = {}  # {intersection_id: start_time}
        self.last_update_time = datetime.now()
        self.emergency_override = {}  # {intersection_id: {active: bool, lane_id: str}}
        self.signal_history = {}  # {intersection_id: [historical_states]}
    
    def configure(self, scenario_config):
        """
        Configure the controller based on scenario settings.
        
        Args:
            scenario_config (dict): Configuration parameters for the scenario
        """
        signal_config = scenario_config.get('signal_config', {})
        self.cycle_time = signal_config.get('cycle_time', 60)
        self.min_green_time = signal_config.get('min_green_time', 10)
        self.max_green_time = signal_config.get('max_green_time', 45)
        self.yellow_time = signal_config.get('yellow_time', 3)
        
        # Reset controller state
        self.current_phase = {}
        self.phase_start_time = {}
        self.last_update_time = datetime.now()
        self.emergency_override = {}
        self.signal_history = {}
        
        # Initialize phases for each intersection
        for intersection_id in scenario_config.get('intersections', {}):
            self.current_phase[intersection_id] = 0
            self.phase_start_time[intersection_id] = datetime.now()
            self.emergency_override[intersection_id] = {'active': False, 'lane_id': None}
            self.signal_history[intersection_id] = []
    
    def adjust_signals(self, traffic_data, predictions):
        """
        Adjust traffic signals based on current traffic data and ML predictions.
        
        Args:
            traffic_data (DataFrame): Current traffic data
            predictions (DataFrame): ML-generated traffic predictions
        
        Returns:
            dict: Updated signal states for each intersection
        """
        if traffic_data.empty:
            return {}
        
        current_time = datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time
        
        # Process each intersection
        signal_states = {}
        
        # Group data by intersection
        intersection_groups = traffic_data.groupby('intersection_id')
        
        for intersection_id, group in intersection_groups:
            # Get current phase and elapsed time
            current_phase = self.current_phase.get(intersection_id, 0)
            phase_start = self.phase_start_time.get(intersection_id, current_time)
            elapsed_time = (current_time - phase_start).total_seconds()
            
            # Check for emergency vehicles
            has_emergency = group['has_emergency'].any()
            if has_emergency:
                # Find lane with emergency vehicle
                emergency_lanes = group[group['has_emergency']]['lane_id'].tolist()
                if emergency_lanes:
                    self.emergency_override[intersection_id] = {
                        'active': True,
                        'lane_id': emergency_lanes[0]
                    }
            
            # Get prediction for this intersection if available
            intersection_predictions = predictions[predictions['intersection_id'] == intersection_id] if not predictions.empty else pd.DataFrame()
            
            # Determine new phase and timing
            new_phase, phase_duration = self._calculate_optimal_phase(
                intersection_id,
                group,
                intersection_predictions,
                current_phase,
                elapsed_time
            )
            
            # Update phase if changed
            if new_phase != current_phase or elapsed_time >= phase_duration:
                self.current_phase[intersection_id] = new_phase
                self.phase_start_time[intersection_id] = current_time
                elapsed_time = 0
            
            # Determine signal states for each lane
            lanes = group['lane_id'].unique()
            lane_phases = {lane: i % len(lanes) for i, lane in enumerate(lanes)}
            
            lane_signals = {}
            for lane in lanes:
                # Determine if lane has green signal in current phase
                if self.emergency_override[intersection_id]['active'] and lane == self.emergency_override[intersection_id]['lane_id']:
                    # Emergency override - give green to emergency vehicle lane
                    signal = 'green'
                elif lane_phases[lane] == new_phase:
                    # Current phase matches lane phase
                    if elapsed_time < phase_duration - self.yellow_time:
                        signal = 'green'
                    else:
                        signal = 'yellow'
                else:
                    signal = 'red'
                
                lane_signals[lane] = signal
            
            # Store signal state for this intersection
            signal_states[intersection_id] = {
                'current_phase': new_phase,
                'elapsed_time': elapsed_time,
                'phase_duration': phase_duration,
                'lane_signals': lane_signals,
                'emergency_override': self.emergency_override[intersection_id]['active']
            }
            
            # Record signal history
            self.signal_history[intersection_id].append({
                'timestamp': current_time,
                'phase': new_phase,
                'lane_signals': lane_signals.copy()
            })
            
            # Keep only recent history (last 100 states)
            if len(self.signal_history[intersection_id]) > 100:
                self.signal_history[intersection_id].pop(0)
            
            # Clear emergency override if it's been active for a while
            if self.emergency_override[intersection_id]['active'] and elapsed_time > 15:
                self.emergency_override[intersection_id] = {'active': False, 'lane_id': None}
        
        return signal_states
    
    def _calculate_optimal_phase(self, intersection_id, traffic_data, predictions, current_phase, elapsed_time):
        """
        Calculate the optimal signal phase using ML predictions.
        
        Args:
            intersection_id: Identifier for the intersection
            traffic_data: Current traffic data for this intersection
            predictions: ML predictions for this intersection
            current_phase: Current signal phase
            elapsed_time: Time elapsed in current phase
        
        Returns:
            tuple: (new_phase, phase_duration)
        """
        # Get the number of lanes to determine total phases
        lanes = traffic_data['lane_id'].unique()
        num_phases = len(lanes)
        
        # Ensure minimum green time before changing
        if elapsed_time < self.min_green_time:
            return current_phase, self.cycle_time / num_phases
        
        # Calculate congestion scores for each lane
        lane_scores = {}
        for lane_id, lane_data in traffic_data.groupby('lane_id'):
            # Base score on current congestion, vehicle count and wait time
            congestion = lane_data['congestion_level'].mean()
            vehicle_count = lane_data['vehicle_count'].mean()
            wait_time = lane_data['wait_time'].mean()
            
            # Weight factors
            congestion_weight = 0.4
            count_weight = 0.3
            wait_weight = 0.3
            
            # Calculate base score
            base_score = (
                congestion_weight * congestion + 
                count_weight * (vehicle_count / 20) +  # Normalize by assuming max 20 vehicles
                wait_weight * (wait_time / 60)  # Normalize by assuming max 60 seconds wait
            )
            
            # Apply prediction adjustment if available
            prediction_adjustment = 0
            if not predictions.empty:
                lane_prediction = predictions[predictions['lane_id'] == lane_id]
                if not lane_prediction.empty:
                    predicted_congestion = lane_prediction['predicted_congestion'].mean()
                    predicted_wait = lane_prediction['predicted_wait_time'].mean()
                    
                    # Calculate prediction adjustment
                    prediction_adjustment = 0.2 * (
                        0.6 * predicted_congestion + 
                        0.4 * (predicted_wait / 60)
                    )
            
            # Final score combines current state and prediction
            lane_scores[lane_id] = base_score + prediction_adjustment
        
        # Map lanes to phases
        lane_phases = {lane: i % num_phases for i, lane in enumerate(lanes)}
        
        # Calculate phase scores
        phase_scores = {}
        for phase in range(num_phases):
            phase_lanes = [lane for lane, p in lane_phases.items() if p == phase]
            if phase_lanes:
                phase_scores[phase] = max(lane_scores.get(lane, 0) for lane in phase_lanes)
            else:
                phase_scores[phase] = 0
        
        # Determine best phase
        best_phase = max(phase_scores, key=phase_scores.get)
        
        # If current phase has score close to best, stick with it to avoid rapid switching
        if current_phase in phase_scores and phase_scores[current_phase] > 0.8 * phase_scores[best_phase]:
            best_phase = current_phase
        
        # Calculate variable phase duration based on congestion
        if best_phase in phase_scores and phase_scores[best_phase] > 0:
            # Scale duration between min and max based on score
            score = phase_scores[best_phase]
            duration = self.min_green_time + (self.max_green_time - self.min_green_time) * score
            duration = min(max(duration, self.min_green_time), self.max_green_time)
        else:
            # Default duration
            duration = self.cycle_time / num_phases
        
        return best_phase, duration
    
    def traditional_control(self, traffic_data):
        """
        Apply traditional fixed-time signal control.
        
        Args:
            traffic_data (DataFrame): Current traffic data
        
        Returns:
            dict: Signal states for each intersection
        """
        if traffic_data.empty:
            return {}
        
        current_time = datetime.now()
        self.last_update_time = current_time
        
        # Process each intersection
        signal_states = {}
        
        # Group data by intersection
        intersection_groups = traffic_data.groupby('intersection_id')
        
        for intersection_id, group in intersection_groups:
            # Get current phase and elapsed time
            if intersection_id not in self.current_phase:
                self.current_phase[intersection_id] = 0
                self.phase_start_time[intersection_id] = current_time
            
            current_phase = self.current_phase[intersection_id]
            phase_start = self.phase_start_time[intersection_id]
            elapsed_time = (current_time - phase_start).total_seconds()
            
            # Get lanes for this intersection
            lanes = group['lane_id'].unique()
            num_phases = len(lanes)
            
            # Fixed duration for each phase
            phase_duration = self.cycle_time / num_phases
            
            # Check if it's time to change phase
            if elapsed_time >= phase_duration:
                current_phase = (current_phase + 1) % num_phases
                self.current_phase[intersection_id] = current_phase
                self.phase_start_time[intersection_id] = current_time
                elapsed_time = 0
            
            # Map lanes to phases
            lane_phases = {lane: i % num_phases for i, lane in enumerate(lanes)}
            
            # Determine signal states for each lane
            lane_signals = {}
            for lane in lanes:
                if lane_phases[lane] == current_phase:
                    if elapsed_time < phase_duration - self.yellow_time:
                        signal = 'green'
                    else:
                        signal = 'yellow'
                else:
                    signal = 'red'
                
                lane_signals[lane] = signal
            
            # Store signal state for this intersection
            signal_states[intersection_id] = {
                'current_phase': current_phase,
                'elapsed_time': elapsed_time,
                'phase_duration': phase_duration,
                'lane_signals': lane_signals,
                'emergency_override': False
            }
        
        return signal_states
    
    def get_signal_history(self, intersection_id):
        """
        Get signal history for a specific intersection.
        
        Args:
            intersection_id: Identifier for the intersection
        
        Returns:
            list: Historical signal states
        """
        return self.signal_history.get(intersection_id, [])
    
    def manual_override(self, intersection_id, new_phase):
        """
        Manually override the current phase.
        
        Args:
            intersection_id: Identifier for the intersection
            new_phase: Phase to switch to
        """
        if intersection_id in self.current_phase:
            self.current_phase[intersection_id] = new_phase
            self.phase_start_time[intersection_id] = datetime.now()
            return True
        return False
