import streamlit as st
import time
import pandas as pd
import numpy as np
import threading
import sqlite3
from datetime import datetime

# Import custom modules
from data_simulation import TrafficSimulator
from ml_prediction import TrafficPredictor
from signal_control import SignalController
from visualization import (
    render_traffic_map,
    render_intersection_status,
    render_metrics_dashboard,
    render_signal_controls
)
from scenarios import load_scenario, get_available_scenarios
from database import init_db, save_simulation_data, get_historical_data
from utils import calculate_performance_metrics

# Set page configuration
st.set_page_config(
    page_title="Intelligent Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Initialize session state variables if they don't exist
if 'simulator' not in st.session_state:
    st.session_state.simulator = TrafficSimulator()

if 'predictor' not in st.session_state:
    st.session_state.predictor = TrafficPredictor()

if 'controller' not in st.session_state:
    st.session_state.controller = SignalController()

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None

if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = pd.DataFrame()

if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()

if 'signal_states' not in st.session_state:
    st.session_state.signal_states = {}

if 'simulation_metrics' not in st.session_state:
    st.session_state.simulation_metrics = {
        'avg_wait_time': 0,
        'throughput': 0,
        'congestion_level': 0,
        'emergency_response_time': 0
    }

if 'comparison_metrics' not in st.session_state:
    st.session_state.comparison_metrics = {
        'traditional': {
            'avg_wait_time': 0,
            'throughput': 0,
            'congestion_level': 0
        },
        'ml_based': {
            'avg_wait_time': 0,
            'throughput': 0,
            'congestion_level': 0
        }
    }

# Header
st.title("🚦 Intelligent Traffic Management System")
st.markdown("""
    This prototype demonstrates how Machine Learning and IoT can be leveraged 
    for smart traffic management. Use the sidebar to configure the system and
    select demonstration scenarios.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Control Panel")
    
    # Scenario selection
    st.subheader("Scenarios")
    scenarios = get_available_scenarios()
    selected_scenario = st.selectbox(
        "Select Demonstration Scenario",
        options=scenarios.keys(),
        index=0
    )
    
    scenario_desc = st.expander("Scenario Description")
    with scenario_desc:
        st.write(scenarios[selected_scenario]['description'])
    
    # Simulation controls
    st.subheader("Simulation Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Simulation", disabled=st.session_state.simulation_running):
            # Load the selected scenario
            scenario_config = load_scenario(selected_scenario)
            st.session_state.current_scenario = selected_scenario
            
            # Configure simulator, predictor, and controller with scenario data
            st.session_state.simulator.configure(scenario_config)
            st.session_state.predictor.load_model(scenario_config['ml_model'])
            st.session_state.controller.configure(scenario_config)
            
            # Start simulation
            st.session_state.simulation_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop Simulation", disabled=not st.session_state.simulation_running):
            st.session_state.simulation_running = False
            st.rerun()
    
    # Simulation speed
    simulation_speed = st.slider(
        "Simulation Speed",
        min_value=1,
        max_value=10,
        value=5,
        disabled=not st.session_state.simulation_running
    )
    
    # Configuration options
    st.subheader("Configuration")
    
    visualization_mode = st.radio(
        "Visualization Mode",
        options=["Map View", "Grid View"],
        index=0
    )
    
    show_predictions = st.checkbox("Show Predictions", value=True)
    show_metrics = st.checkbox("Show Performance Metrics", value=True)
    enable_ml = st.checkbox("Enable ML-based Control", value=True)
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        Intelligent Traffic Management System Prototype 
        using Machine Learning and IoT
    """)

# Main content layout
if st.session_state.simulation_running:
    # Update simulation
    update_interval = 11 - simulation_speed  # Convert slider to seconds (1-10 becomes 10-1)
    
    # Get new traffic data
    new_traffic_data = st.session_state.simulator.generate_data()
    
    # Save to database
    save_simulation_data(new_traffic_data)
    
    # Update session state
    st.session_state.traffic_data = new_traffic_data
    
    # Run predictions if enabled
    if enable_ml:
        st.session_state.predictions = st.session_state.predictor.predict(new_traffic_data)
        # Use predictions to adjust signals
        st.session_state.signal_states = st.session_state.controller.adjust_signals(
            new_traffic_data, 
            st.session_state.predictions
        )
    else:
        # Traditional signal control
        st.session_state.signal_states = st.session_state.controller.traditional_control(new_traffic_data)
    
    # Calculate metrics
    st.session_state.simulation_metrics = calculate_performance_metrics(
        new_traffic_data, 
        st.session_state.signal_states,
        enable_ml
    )
    
    # Update comparison metrics
    if enable_ml:
        st.session_state.comparison_metrics['ml_based'] = {
            'avg_wait_time': st.session_state.simulation_metrics['avg_wait_time'],
            'throughput': st.session_state.simulation_metrics['throughput'],
            'congestion_level': st.session_state.simulation_metrics['congestion_level']
        }
    else:
        st.session_state.comparison_metrics['traditional'] = {
            'avg_wait_time': st.session_state.simulation_metrics['avg_wait_time'],
            'throughput': st.session_state.simulation_metrics['throughput'],
            'congestion_level': st.session_state.simulation_metrics['congestion_level']
        }

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Traffic Overview", "Intersection Control", "Analytics"])

with tab1:
    # Traffic Overview Tab
    st.header("Traffic Overview")
    
    # Traffic Map/Grid Visualization
    render_traffic_map(
        st.session_state.traffic_data, 
        st.session_state.signal_states,
        visualization_mode
    )
    
    # Current status and metrics
    if show_metrics and not st.session_state.traffic_data.empty:
        st.subheader("Current Traffic Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric(
                "Avg. Wait Time", 
                f"{st.session_state.simulation_metrics['avg_wait_time']:.1f}s"
            )
        
        with metrics_col2:
            st.metric(
                "Throughput", 
                f"{st.session_state.simulation_metrics['throughput']} veh/min"
            )
        
        with metrics_col3:
            congestion_level = st.session_state.simulation_metrics['congestion_level']
            st.metric(
                "Congestion Level", 
                f"{congestion_level:.1f}%"
            )
        
        with metrics_col4:
            if 'emergency_response_time' in st.session_state.simulation_metrics:
                st.metric(
                    "Emergency Response", 
                    f"{st.session_state.simulation_metrics['emergency_response_time']:.1f}s"
                )

with tab2:
    # Intersection Control Tab
    st.header("Intersection Control")
    
    # Select intersection
    if not st.session_state.traffic_data.empty:
        # Get unique intersections from traffic data
        intersections = st.session_state.traffic_data['intersection_id'].unique()
        
        selected_intersection = st.selectbox(
            "Select Intersection",
            options=intersections,
            index=0
        )
        
        # Display intersection status
        intersection_data = st.session_state.traffic_data[
            st.session_state.traffic_data['intersection_id'] == selected_intersection
        ]
        
        signal_state = None
        if selected_intersection in st.session_state.signal_states:
            signal_state = st.session_state.signal_states[selected_intersection]
        
        render_intersection_status(intersection_data, signal_state)
        
        # Signal control panel
        st.subheader("Signal Control")
        render_signal_controls(selected_intersection, signal_state, st.session_state.controller)
    else:
        st.info("Start the simulation to see intersection control options.")

with tab3:
    # Analytics Tab
    st.header("Traffic Analytics")
    
    if show_predictions and not st.session_state.traffic_data.empty and not st.session_state.predictions.empty:
        st.subheader("Traffic Predictions")
        
        # Show the current predictions
        render_metrics_dashboard(
            st.session_state.traffic_data,
            st.session_state.predictions,
            st.session_state.comparison_metrics
        )
        
        # Historical data
        st.subheader("Historical Data")
        historical_data = get_historical_data()
        
        if not historical_data.empty:
            # Filter options
            time_range = st.selectbox(
                "Time Range",
                options=["Last 5 minutes", "Last 15 minutes", "Last hour", "All data"],
                index=0
            )
            
            # Filter data based on selection
            if time_range == "Last 5 minutes":
                cutoff_time = pd.Timestamp.now() - pd.Timedelta(minutes=5)
            elif time_range == "Last 15 minutes":
                cutoff_time = pd.Timestamp.now() - pd.Timedelta(minutes=15)
            elif time_range == "Last hour":
                cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=1)
            else:
                cutoff_time = pd.Timestamp.min
            
            filtered_data = historical_data[historical_data['timestamp'] > cutoff_time]
            
            if not filtered_data.empty:
                st.line_chart(
                    filtered_data.groupby('timestamp').agg({
                        'vehicle_count': 'sum',
                        'avg_speed': 'mean',
                        'wait_time': 'mean'
                    })
                )
            else:
                st.info(f"No data available for the selected time range: {time_range}")
        else:
            st.info("No historical data available yet. Run the simulation to collect data.")
    else:
        st.info("Start the simulation with predictions enabled to see analytics.")

# Auto-refresh for simulation updates
if st.session_state.simulation_running:
    time.sleep(update_interval)
    st.rerun()
