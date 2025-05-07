import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_available_scenarios():
    """
    Get list of available demonstration scenarios.
    
    Returns:
        dict: Dictionary of scenario names and descriptions
    """
    return {
        "normal_traffic": {
            "name": "Normal Traffic Conditions",
            "description": "Simulates regular traffic flow with balanced vehicle distribution across lanes. Ideal for demonstrating baseline system behavior."
        },
        "rush_hour": {
            "name": "Rush Hour Congestion",
            "description": "Heavy traffic simulating morning/evening rush hour. Shows how the system handles high congestion periods and adjusts signal timing."
        },
        "emergency_vehicle": {
            "name": "Emergency Vehicle Priority",
            "description": "Demonstrates how the system detects and prioritizes emergency vehicles, adjusting signals to provide right-of-way."
        },
        "unbalanced_flow": {
            "name": "Unbalanced Traffic Flow",
            "description": "Traffic concentrated in certain directions, showing how adaptive signals balance uneven distribution better than fixed timing."
        },
        "incident_response": {
            "name": "Traffic Incident Response",
            "description": "Simulates a traffic incident causing congestion. Shows how the system responds to sudden changes in traffic patterns."
        }
    }

def load_scenario(scenario_name):
    """
    Load configuration for specified scenario.
    
    Args:
        scenario_name (str): Name of the scenario to load
    
    Returns:
        dict: Scenario configuration
    """
    scenarios = {
        "normal_traffic": _normal_traffic_scenario(),
        "rush_hour": _rush_hour_scenario(),
        "emergency_vehicle": _emergency_vehicle_scenario(),
        "unbalanced_flow": _unbalanced_flow_scenario(),
        "incident_response": _incident_response_scenario()
    }
    
    return scenarios.get(scenario_name, _normal_traffic_scenario())

def _normal_traffic_scenario():
    """
    Normal traffic scenario configuration.
    
    Returns:
        dict: Scenario configuration
    """
    return {
        "name": "Normal Traffic Conditions",
        "description": "Regular traffic flow with balanced distribution",
        "time_step": 1,
        "base_congestion": 0.2,
        "random_seed": 42,
        "intersections": {
            "I1": {
                "name": "Main & First",
                "location": (0, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "regular"}
                ]
            },
            "I2": {
                "name": "Main & Second",
                "location": (500, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "regular"}
                ]
            },
            "I3": {
                "name": "Center & Avenue",
                "location": (250, 250),
                "lanes": [
                    {"direction": "N", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "E", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "S", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "W", "speed_limit": 40, "capacity": 15, "type": "regular"}
                ]
            }
        },
        "ml_model": {
            "type": "random_forest",
            "n_estimators": 50,
            "max_depth": 10,
            "prediction_window": 5
        },
        "signal_config": {
            "cycle_time": 60,
            "min_green_time": 10,
            "max_green_time": 30,
            "yellow_time": 3
        }
    }

def _rush_hour_scenario():
    """
    Rush hour scenario configuration.
    
    Returns:
        dict: Scenario configuration
    """
    return {
        "name": "Rush Hour Congestion",
        "description": "Heavy traffic simulating morning/evening rush hour",
        "time_step": 1,
        "base_congestion": 0.6,  # Higher base congestion
        "random_seed": 42,
        "intersections": {
            "I1": {
                "name": "Main & First",
                "location": (0, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "main"}
                ]
            },
            "I2": {
                "name": "Main & Second",
                "location": (500, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "main"}
                ]
            },
            "I3": {
                "name": "Downtown Junction",
                "location": (250, 250),
                "lanes": [
                    {"direction": "N", "speed_limit": 40, "capacity": 15, "type": "main"},
                    {"direction": "E", "speed_limit": 40, "capacity": 15, "type": "main"},
                    {"direction": "S", "speed_limit": 40, "capacity": 15, "type": "main"},
                    {"direction": "W", "speed_limit": 40, "capacity": 15, "type": "main"}
                ]
            }
        },
        "ml_model": {
            "type": "random_forest",
            "n_estimators": 100,  # More complex model for rush hour
            "max_depth": 15,
            "prediction_window": 5
        },
        "signal_config": {
            "cycle_time": 90,  # Longer cycle time for rush hour
            "min_green_time": 15,
            "max_green_time": 60,
            "yellow_time": 3
        }
    }

def _emergency_vehicle_scenario():
    """
    Emergency vehicle scenario configuration.
    
    Returns:
        dict: Scenario configuration
    """
    return {
        "name": "Emergency Vehicle Priority",
        "description": "Demonstrates emergency vehicle detection and prioritization",
        "time_step": 1,
        "base_congestion": 0.4,
        "random_seed": 42,
        "intersections": {
            "I1": {
                "name": "Hospital Junction",
                "location": (0, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "regular"}
                ]
            },
            "I2": {
                "name": "Fire Station Access",
                "location": (500, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "regular"}
                ]
            }
        },
        "emergency_vehicles": {
            "count": 3,
            "types": ["ambulance", "fire_truck", "police"]
        },
        "ml_model": {
            "type": "random_forest",
            "n_estimators": 50,
            "max_depth": 10,
            "prediction_window": 5
        },
        "signal_config": {
            "cycle_time": 60,
            "min_green_time": 10,
            "max_green_time": 30,
            "yellow_time": 3,
            "emergency_override": True
        }
    }

def _unbalanced_flow_scenario():
    """
    Unbalanced traffic flow scenario configuration.
    
    Returns:
        dict: Scenario configuration
    """
    return {
        "name": "Unbalanced Traffic Flow",
        "description": "Traffic concentrated in certain directions",
        "time_step": 1,
        "base_congestion": 0.3,
        "random_seed": 42,
        "intersections": {
            "I1": {
                "name": "Highway Exit",
                "location": (0, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "side"},
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "main"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "side"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "main"}
                ]
            },
            "I2": {
                "name": "School Zone",
                "location": (500, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 30, "capacity": 15, "type": "main"},
                    {"direction": "E", "speed_limit": 30, "capacity": 15, "type": "side"},
                    {"direction": "S", "speed_limit": 30, "capacity": 15, "type": "main"},
                    {"direction": "W", "speed_limit": 30, "capacity": 15, "type": "side"}
                ]
            },
            "I3": {
                "name": "Shopping Mall",
                "location": (250, 250),
                "lanes": [
                    {"direction": "N", "speed_limit": 40, "capacity": 15, "type": "side"},
                    {"direction": "E", "speed_limit": 40, "capacity": 15, "type": "side"},
                    {"direction": "S", "speed_limit": 40, "capacity": 15, "type": "main"},
                    {"direction": "W", "speed_limit": 40, "capacity": 15, "type": "main"}
                ]
            }
        },
        "ml_model": {
            "type": "random_forest",
            "n_estimators": 75,
            "max_depth": 12,
            "prediction_window": 5
        },
        "signal_config": {
            "cycle_time": 60,
            "min_green_time": 10,
            "max_green_time": 45,
            "yellow_time": 3
        }
    }

def _incident_response_scenario():
    """
    Traffic incident response scenario configuration.
    
    Returns:
        dict: Scenario configuration
    """
    return {
        "name": "Traffic Incident Response",
        "description": "Simulates a traffic incident causing congestion",
        "time_step": 1,
        "base_congestion": 0.3,
        "random_seed": 42,
        "intersections": {
            "I1": {
                "name": "Incident Location",
                "location": (0, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 10, "type": "regular"},  # Reduced capacity due to incident
                    {"direction": "E", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 20, "type": "regular"}
                ]
            },
            "I2": {
                "name": "Detour Junction",
                "location": (500, 0),
                "lanes": [
                    {"direction": "N", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "E", "speed_limit": 50, "capacity": 15, "type": "regular"},
                    {"direction": "S", "speed_limit": 50, "capacity": 20, "type": "regular"},
                    {"direction": "W", "speed_limit": 50, "capacity": 15, "type": "regular"}
                ]
            },
            "I3": {
                "name": "Alternative Route",
                "location": (250, 250),
                "lanes": [
                    {"direction": "N", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "E", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "S", "speed_limit": 40, "capacity": 15, "type": "regular"},
                    {"direction": "W", "speed_limit": 40, "capacity": 15, "type": "regular"}
                ]
            }
        },
        "emergency_vehicles": {
            "count": 1,
            "types": ["police"]
        },
        "ml_model": {
            "type": "random_forest",
            "n_estimators": 50,
            "max_depth": 10,
            "prediction_window": 5
        },
        "signal_config": {
            "cycle_time": 60,
            "min_green_time": 10,
            "max_green_time": 30,
            "yellow_time": 3,
            "emergency_override": True
        }
    }
