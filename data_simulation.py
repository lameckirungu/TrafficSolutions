import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math
from utils import calculate_congestion

class TrafficSimulator:
    """
    Simulates traffic data for intersections based on configured scenarios.
    Generates realistic traffic patterns including vehicle counts, speeds, and wait times.
    """
    
    def __init__(self):
        """Initialize the traffic simulator with default settings."""
        self.intersections = {}
        self.vehicles = {}
        self.emergency_vehicles = {}
        self.current_time = datetime.now()
        self.time_step = 1  # seconds
        self.scenario_name = None
        self.base_congestion = 0.2  # Base congestion level (0-1)
        self.time_factor = 0  # Time progression factor
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def configure(self, scenario_config):
        """
        Configure the simulator based on scenario settings.
        
        Args:
            scenario_config (dict): Configuration parameters for the scenario
        """
        self.scenario_name = scenario_config.get('name', 'Default Scenario')
        self.intersections = scenario_config.get('intersections', {})
        self.current_time = datetime.now()
        self.time_step = scenario_config.get('time_step', 1)
        self.base_congestion = scenario_config.get('base_congestion', 0.2)
        self.random_seed = scenario_config.get('random_seed', 42)
        
        # Reset simulation state
        self.vehicles = {}
        self.emergency_vehicles = {}
        self.time_factor = 0
        
        # Initialize vehicles at each intersection
        for intersection_id, config in self.intersections.items():
            # Initialize 10-30 vehicles per lane based on intersection size
            num_lanes = len(config['lanes'])
            for lane_idx in range(num_lanes):
                self._initialize_vehicles(intersection_id, lane_idx, config['lanes'][lane_idx])
        
        # Initialize emergency vehicles if scenario has them
        if 'emergency_vehicles' in scenario_config:
            self._initialize_emergency_vehicles(scenario_config['emergency_vehicles'])
            
        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def _initialize_vehicles(self, intersection_id, lane_idx, lane_config):
        """Initialize vehicles for a specific lane."""
        lane_id = f"{intersection_id}_lane_{lane_idx}"
        
        # Determine number of vehicles based on lane type and congestion
        lane_type = lane_config.get('type', 'regular')
        base_vehicle_count = {
            'main': np.random.randint(5, 20),
            'regular': np.random.randint(3, 15),
            'side': np.random.randint(1, 10)
        }.get(lane_type, 5)
        
        # Apply time-of-day and congestion factors
        vehicle_count = int(base_vehicle_count * (1 + self.base_congestion))
        
        # Initialize vehicles with random properties
        self.vehicles[lane_id] = []
        for _ in range(vehicle_count):
            vehicle = {
                'id': f"v_{lane_id}_{random.randint(1000, 9999)}",
                'type': np.random.choice(['car', 'truck', 'bus', 'motorcycle'], p=[0.8, 0.1, 0.05, 0.05]),
                'speed': np.random.normal(lane_config.get('speed_limit', 50) * 0.7, 10),
                'position': np.random.uniform(0, 100),
                'wait_time': 0,
                'direction': lane_config.get('direction', 'N')
            }
            self.vehicles[lane_id].append(vehicle)
    
    def _initialize_emergency_vehicles(self, emergency_config):
        """Initialize emergency vehicles if present in scenario."""
        if not emergency_config:
            return
            
        num_emergency = emergency_config.get('count', 0)
        for i in range(num_emergency):
            # Randomly select an intersection and lane
            intersection_id = random.choice(list(self.intersections.keys()))
            lane_idx = random.randint(0, len(self.intersections[intersection_id]['lanes']) - 1)
            lane_id = f"{intersection_id}_lane_{lane_idx}"
            
            # Create emergency vehicle
            emergency_vehicle = {
                'id': f"ev_{i+1}",
                'type': np.random.choice(['ambulance', 'fire_truck', 'police'], p=[0.4, 0.3, 0.3]),
                'speed': np.random.normal(80, 10),  # Emergency vehicles move faster
                'position': np.random.uniform(0, 100),
                'wait_time': 0,
                'direction': self.intersections[intersection_id]['lanes'][lane_idx].get('direction', 'N'),
                'lane_id': lane_id,
                'intersection_id': intersection_id,
                'arrival_time': self.current_time + timedelta(seconds=np.random.randint(5, 30))
            }
            
            # Add to emergency vehicles dict
            self.emergency_vehicles[emergency_vehicle['id']] = emergency_vehicle
    
    def generate_data(self):
        """
        Generate simulated traffic data for the current time step.
        
        Returns:
            pandas.DataFrame: Simulated traffic data
        """
        # Advance simulation time
        self.current_time += timedelta(seconds=self.time_step)
        self.time_factor += 1
        
        # List to store all data points
        data_points = []
        
        # Add time-varying congestion factor (simulating rush hours, etc.)
        time_congestion = 0.2 * math.sin(self.time_factor / 100) + 0.1 * math.sin(self.time_factor / 50)
        
        # Process each intersection
        for intersection_id, config in self.intersections.items():
            # Process each lane in the intersection
            for lane_idx, lane_config in enumerate(config['lanes']):
                lane_id = f"{intersection_id}_lane_{lane_idx}"
                lane_vehicles = self.vehicles.get(lane_id, [])
                
                # Skip if no vehicles in lane
                if not lane_vehicles:
                    continue
                
                # Determine if signal is green for this lane (simplified)
                # In a real system, this would come from the signal controller
                is_green = (self.time_factor % 4 == lane_idx % 4)
                
                # Calculate congestion level for this lane
                vehicle_count = len(lane_vehicles)
                lane_capacity = lane_config.get('capacity', 20)
                congestion_level = calculate_congestion(vehicle_count, lane_capacity, is_green)
                
                # Adjust for time-based congestion
                congestion_level += time_congestion
                congestion_level = max(0, min(congestion_level, 1))  # Bound to [0,1]
                
                # Calculate average speed and wait time
                speeds = [v['speed'] for v in lane_vehicles]
                avg_speed = np.mean(speeds) if speeds else 0
                
                # Adjust speed based on congestion
                avg_speed *= (1 - 0.7 * congestion_level)  # Speed reduces with congestion
                
                # Calculate wait time based on signal and congestion
                wait_time = 0
                if not is_green:
                    wait_time = np.random.exponential(5 * (1 + congestion_level))
                
                # Update vehicle wait times
                for vehicle in lane_vehicles:
                    if not is_green:
                        vehicle['wait_time'] += self.time_step
                    else:
                        vehicle['wait_time'] = max(0, vehicle['wait_time'] - self.time_step / 2)
                
                # Calculate average wait time
                avg_wait_time = np.mean([v['wait_time'] for v in lane_vehicles])
                
                # Create data point for this lane
                data_point = {
                    'timestamp': self.current_time,
                    'intersection_id': intersection_id,
                    'lane_id': lane_id,
                    'direction': lane_config.get('direction', 'N'),
                    'vehicle_count': vehicle_count,
                    'avg_speed': avg_speed,
                    'congestion_level': congestion_level,
                    'wait_time': avg_wait_time,
                    'is_green': is_green,
                    'has_emergency': False
                }
                
                # Check for emergency vehicles in this lane
                for ev_id, ev in list(self.emergency_vehicles.items()):
                    if ev['lane_id'] == lane_id and ev['arrival_time'] <= self.current_time:
                        data_point['has_emergency'] = True
                        # Process emergency vehicle movement
                        if is_green:
                            # Remove emergency vehicle if it passed through intersection
                            del self.emergency_vehicles[ev_id]
                
                data_points.append(data_point)
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        
        # For empty result, return minimal dataframe with required columns
        if df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'intersection_id', 'lane_id', 'direction',
                'vehicle_count', 'avg_speed', 'congestion_level', 'wait_time',
                'is_green', 'has_emergency'
            ])
        
        return df
