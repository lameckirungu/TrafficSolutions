import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import math

def render_traffic_map(traffic_data, signal_states, view_mode="Map View"):
    """
    Render traffic map visualization.
    
    Args:
        traffic_data (DataFrame): Current traffic data
        signal_states (dict): Current signal states
        view_mode (str): "Map View" or "Grid View"
    """
    if traffic_data.empty:
        st.info("No traffic data available. Start a simulation to see the traffic map.")
        return
    
    if view_mode == "Map View":
        _render_map_view(traffic_data, signal_states)
    else:
        _render_grid_view(traffic_data, signal_states)

def _render_map_view(traffic_data, signal_states):
    """Render map-based visualization of traffic."""
    # Create a simple map visualization
    st.subheader("Traffic Map")
    
    # Group data by intersection
    intersection_groups = traffic_data.groupby('intersection_id')
    
    # Create map layout
    map_fig = go.Figure()
    
    # Map boundaries
    map_bounds = {
        'min_x': 0,
        'max_x': 1000,
        'min_y': 0,
        'max_y': 1000
    }
    
    # Road layout parameters
    road_width = 20
    intersection_size = 40
    
    # Assign positions to intersections
    intersection_positions = {}
    i = 0
    for intersection_id in traffic_data['intersection_id'].unique():
        # Position intersections in a grid-like pattern
        row = i // 3
        col = i % 3
        
        # Calculate position
        pos_x = 200 + col * 300
        pos_y = 200 + row * 300
        
        intersection_positions[intersection_id] = (pos_x, pos_y)
        i += 1
    
    # Draw roads and intersections
    for intersection_id, (x, y) in intersection_positions.items():
        # Draw intersection
        map_fig.add_shape(
            type="rect",
            x0=x - intersection_size/2,
            y0=y - intersection_size/2,
            x1=x + intersection_size/2,
            y1=y + intersection_size/2,
            line=dict(color="black", width=2),
            fillcolor="lightgray"
        )
        
        # Draw roads (horizontal and vertical)
        map_fig.add_shape(
            type="rect",
            x0=map_bounds['min_x'],
            y0=y - road_width/2,
            x1=map_bounds['max_x'],
            y1=y + road_width/2,
            line=dict(color="black", width=1),
            fillcolor="gray"
        )
        
        map_fig.add_shape(
            type="rect",
            x0=x - road_width/2,
            y0=map_bounds['min_y'],
            x1=x + road_width/2,
            y1=map_bounds['max_y'],
            line=dict(color="black", width=1),
            fillcolor="gray"
        )
        
        # Add intersection label
        map_fig.add_annotation(
            x=x,
            y=y,
            text=intersection_id,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    # Draw traffic signals and vehicles
    for intersection_id, (x, y) in intersection_positions.items():
        if intersection_id in signal_states:
            # Get signal state for this intersection
            intersection_signals = signal_states[intersection_id]['lane_signals']
            
            # Draw signals for each direction
            directions = [('N', 0, 1), ('E', 1, 0), ('S', 0, -1), ('W', -1, 0)]
            for direction, dx, dy in directions:
                # Find lanes for this direction
                direction_lanes = [
                    lane for lane in intersection_signals.keys()
                    if lane.endswith(direction)
                ]
                
                if direction_lanes:
                    # Use the first lane for the direction
                    lane = direction_lanes[0]
                    signal_color = intersection_signals[lane]
                    
                    # Signal colors
                    color_map = {
                        'red': 'red',
                        'yellow': 'gold',
                        'green': 'green'
                    }
                    
                    # Calculate signal position
                    signal_x = x + dx * (intersection_size/2 + 10)
                    signal_y = y + dy * (intersection_size/2 + 10)
                    
                    # Draw signal
                    map_fig.add_shape(
                        type="circle",
                        x0=signal_x - 5,
                        y0=signal_y - 5,
                        x1=signal_x + 5,
                        y1=signal_y + 5,
                        line=dict(color="black", width=1),
                        fillcolor=color_map.get(signal_color, 'gray')
                    )
        
        # Get traffic data for this intersection
        intersection_data = traffic_data[traffic_data['intersection_id'] == intersection_id]
        
        # Visualize congestion for each direction
        for lane_id, lane_data in intersection_data.groupby('lane_id'):
            # Extract direction from lane_id
            direction = lane_data['direction'].iloc[0] if not lane_data.empty else 'N'
            
            # Map direction to dx, dy
            direction_map = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
            dx, dy = direction_map.get(direction, (0, 0))
            
            # Calculate position
            pos_x = x + dx * 100
            pos_y = y + dy * 100
            
            # Get congestion and vehicle count
            congestion = lane_data['congestion_level'].mean() if not lane_data.empty else 0
            vehicle_count = int(lane_data['vehicle_count'].mean()) if not lane_data.empty else 0
            
            # Color based on congestion
            congestion_color = _get_congestion_color(congestion)
            
            # Draw congestion indicator
            map_fig.add_shape(
                type="rect",
                x0=pos_x - 15,
                y0=pos_y - 15,
                x1=pos_x + 15,
                y1=pos_y + 15,
                line=dict(color="black", width=1),
                fillcolor=congestion_color
            )
            
            # Add vehicle count
            map_fig.add_annotation(
                x=pos_x,
                y=pos_y,
                text=str(vehicle_count),
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0)"
            )
            
            # Check for emergency vehicle
            has_emergency = lane_data['has_emergency'].any() if not lane_data.empty else False
            if has_emergency:
                map_fig.add_shape(
                    type="circle",
                    x0=pos_x - 20,
                    y0=pos_y - 20,
                    x1=pos_x + 20,
                    y1=pos_y + 20,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(0,0,0,0)"
                )
                
                map_fig.add_annotation(
                    x=pos_x,
                    y=pos_y + 30,
                    text="EMERGENCY",
                    showarrow=False,
                    font=dict(size=10, color="red", family="Arial Black"),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
    
    # Configure map layout
    map_fig.update_layout(
        showlegend=False,
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='lightblue',
        xaxis=dict(
            range=[map_bounds['min_x'], map_bounds['max_x']],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[map_bounds['min_y'], map_bounds['max_y']],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    # Display the map
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Map legend
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    
    with legend_col1:
        st.markdown("**Congestion Levels:**")
        st.markdown("🟢 Low (0-30%)")
        st.markdown("🟡 Medium (30-60%)")
        st.markdown("🔴 High (60-100%)")
    
    with legend_col2:
        st.markdown("**Traffic Signals:**")
        st.markdown("🟢 Green")
        st.markdown("🟡 Yellow")
        st.markdown("🔴 Red")
    
    with legend_col3:
        st.markdown("**Special Indicators:**")
        st.markdown("🚨 Emergency Vehicle")
        st.markdown("🔢 Vehicle Count")

def _render_grid_view(traffic_data, signal_states):
    """Render grid-based visualization of traffic."""
    st.subheader("Traffic Grid")
    
    # Group data by intersection
    intersection_groups = traffic_data.groupby('intersection_id')
    
    # Create a layout of intersections
    cols = st.columns(min(3, len(intersection_groups)))
    
    # Display each intersection
    for i, (intersection_id, group) in enumerate(intersection_groups):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            st.markdown(f"**Intersection {intersection_id}**")
            
            # Create a simple grid visualization
            grid_fig = go.Figure()
            
            # Draw intersection grid
            grid_size = 100
            center_x = grid_size / 2
            center_y = grid_size / 2
            road_width = 20
            
            # Draw roads
            grid_fig.add_shape(
                type="rect",
                x0=0,
                y0=center_y - road_width/2,
                x1=grid_size,
                y1=center_y + road_width/2,
                line=dict(color="black", width=1),
                fillcolor="gray"
            )
            
            grid_fig.add_shape(
                type="rect",
                x0=center_x - road_width/2,
                y0=0,
                x1=center_x + road_width/2,
                y1=grid_size,
                line=dict(color="black", width=1),
                fillcolor="gray"
            )
            
            # Center intersection
            grid_fig.add_shape(
                type="rect",
                x0=center_x - road_width/2,
                y0=center_y - road_width/2,
                x1=center_x + road_width/2,
                y1=center_y + road_width/2,
                line=dict(color="black", width=1),
                fillcolor="lightgray"
            )
            
            # Draw signals and traffic for each direction
            directions = [
                ('N', center_x, grid_size - 10, 'top'),
                ('E', grid_size - 10, center_y, 'right'),
                ('S', center_x, 10, 'bottom'),
                ('W', 10, center_y, 'left')
            ]
            
            # Get signal state for this intersection
            intersection_signals = signal_states.get(intersection_id, {}).get('lane_signals', {})
            
            for direction, x, y, position in directions:
                # Find lanes for this direction
                direction_lanes = group[group['direction'] == direction]['lane_id'].unique()
                
                if len(direction_lanes) > 0:
                    lane_id = direction_lanes[0]
                    lane_data = group[group['lane_id'] == lane_id]
                    
                    # Get congestion and vehicle count
                    congestion = lane_data['congestion_level'].mean() if not lane_data.empty else 0
                    vehicle_count = int(lane_data['vehicle_count'].mean()) if not lane_data.empty else 0
                    
                    # Get signal state
                    signal_color = intersection_signals.get(lane_id, 'gray')
                    
                    # Color based on congestion
                    congestion_color = _get_congestion_color(congestion)
                    
                    # Draw traffic indicator
                    grid_fig.add_shape(
                        type="circle",
                        x0=x - 8,
                        y0=y - 8,
                        x1=x + 8,
                        y1=y + 8,
                        line=dict(color="black", width=1),
                        fillcolor=congestion_color
                    )
                    
                    # Add vehicle count
                    grid_fig.add_annotation(
                        x=x,
                        y=y,
                        text=str(vehicle_count),
                        showarrow=False,
                        font=dict(size=10, color="white"),
                        bgcolor="rgba(0,0,0,0)"
                    )
                    
                    # Draw signal
                    signal_positions = {
                        'top': (center_x, center_y + road_width/2 + 5),
                        'right': (center_x + road_width/2 + 5, center_y),
                        'bottom': (center_x, center_y - road_width/2 - 5),
                        'left': (center_x - road_width/2 - 5, center_y)
                    }
                    
                    signal_x, signal_y = signal_positions[position]
                    
                    # Signal colors
                    color_map = {
                        'red': 'red',
                        'yellow': 'gold',
                        'green': 'green',
                        'gray': 'gray'
                    }
                    
                    grid_fig.add_shape(
                        type="circle",
                        x0=signal_x - 3,
                        y0=signal_y - 3,
                        x1=signal_x + 3,
                        y1=signal_y + 3,
                        line=dict(color="black", width=1),
                        fillcolor=color_map.get(signal_color, 'gray')
                    )
                    
                    # Check for emergency vehicle
                    has_emergency = lane_data['has_emergency'].any() if not lane_data.empty else False
                    if has_emergency:
                        grid_fig.add_shape(
                            type="circle",
                            x0=x - 12,
                            y0=y - 12,
                            x1=x + 12,
                            y1=y + 12,
                            line=dict(color="red", width=2),
                            fillcolor="rgba(0,0,0,0)"
                        )
            
            # Configure grid layout
            grid_fig.update_layout(
                showlegend=False,
                width=250,
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='lightblue',
                xaxis=dict(
                    range=[0, grid_size],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    range=[0, grid_size],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1
                )
            )
            
            # Display the grid
            st.plotly_chart(grid_fig, use_container_width=True)
            
            # Show additional information
            if intersection_id in signal_states:
                phase = signal_states[intersection_id]['current_phase']
                duration = signal_states[intersection_id]['phase_duration']
                elapsed = signal_states[intersection_id]['elapsed_time']
                
                st.progress(min(1.0, elapsed / duration))
                st.text(f"Phase: {phase+1}, Time: {int(elapsed)}/{int(duration)}s")
                
                if signal_states[intersection_id]['emergency_override']:
                    st.error("⚠️ Emergency vehicle priority active")

def render_intersection_status(intersection_data, signal_state):
    """
    Render detailed status for a specific intersection.
    
    Args:
        intersection_data (DataFrame): Data for the selected intersection
        signal_state (dict): Signal state for the intersection
    """
    if intersection_data.empty:
        st.info("No data available for this intersection.")
        return
    
    st.subheader("Intersection Status")
    
    # Lane data table
    lane_summary = intersection_data.groupby('lane_id').agg({
        'vehicle_count': 'mean',
        'avg_speed': 'mean',
        'wait_time': 'mean',
        'congestion_level': 'mean',
        'has_emergency': 'any'
    }).reset_index()
    
    # Add signal status
    if signal_state and 'lane_signals' in signal_state:
        lane_summary['signal'] = lane_summary['lane_id'].map(
            lambda x: signal_state['lane_signals'].get(x, 'Unknown')
        )
    else:
        lane_summary['signal'] = 'Unknown'
    
    # Format table
    formatted_table = lane_summary.copy()
    formatted_table['vehicle_count'] = formatted_table['vehicle_count'].round(0).astype(int)
    formatted_table['avg_speed'] = formatted_table['avg_speed'].round(1)
    formatted_table['wait_time'] = formatted_table['wait_time'].round(1)
    formatted_table['congestion_level'] = (formatted_table['congestion_level'] * 100).round(1).astype(str) + '%'
    
    # Rename columns for display
    formatted_table.columns = ['Lane', 'Vehicles', 'Avg Speed (km/h)', 'Wait Time (s)', 'Congestion', 'Emergency', 'Signal']
    
    # Colorize signal column
    def highlight_signal(val):
        color_map = {
            'red': 'background-color: red; color: white',
            'yellow': 'background-color: gold; color: black',
            'green': 'background-color: green; color: white',
            'Unknown': 'background-color: gray; color: white'
        }
        return color_map.get(val, '')
    
    # Colorize congestion column
    def highlight_congestion(val):
        try:
            level = float(val.strip('%'))
            if level < 30:
                return 'background-color: rgba(0, 255, 0, 0.3)'
            elif level < 60:
                return 'background-color: rgba(255, 255, 0, 0.3)'
            else:
                return 'background-color: rgba(255, 0, 0, 0.3)'
        except:
            return ''
    
    # Apply styles
    styled_table = formatted_table.style.applymap(
        highlight_signal, subset=['Signal']
    ).applymap(
        highlight_congestion, subset=['Congestion']
    )
    
    # Display table
    st.dataframe(styled_table, use_container_width=True)
    
    # Show current phase and timing
    if signal_state:
        phase = signal_state.get('current_phase', 0)
        duration = signal_state.get('phase_duration', 0)
        elapsed = signal_state.get('elapsed_time', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Phase", f"Phase {phase+1}")
            st.progress(min(1.0, elapsed / duration))
            st.text(f"Time: {int(elapsed)}/{int(duration)} seconds")
        
        with col2:
            # Display emergency status
            if signal_state.get('emergency_override', False):
                st.error("⚠️ Emergency Vehicle Priority Active")
            else:
                st.success("✓ Normal Operation")
            
            # Count vehicles by type
            if 'vehicle_type' in intersection_data.columns:
                vehicle_types = intersection_data['vehicle_type'].value_counts()
                st.write("Vehicle Types:")
                for vtype, count in vehicle_types.items():
                    st.text(f"- {vtype.capitalize()}: {count}")

def render_signal_controls(intersection_id, signal_state, controller):
    """
    Render signal control panel.
    
    Args:
        intersection_id (str): Intersection identifier
        signal_state (dict): Current signal state
        controller (SignalController): Signal controller object
    """
    st.subheader("Signal Control Panel")
    
    if not signal_state:
        st.info("No signal data available. Start the simulation first.")
        return
    
    current_phase = signal_state.get('current_phase', 0)
    num_phases = 4  # Assume 4 phases for simplicity
    
    # Manual override
    st.write("Manual Override:")
    cols = st.columns(num_phases)
    
    for i in range(num_phases):
        with cols[i]:
            phase_active = current_phase == i
            button_type = "primary" if phase_active else "secondary"
            if st.button(f"Phase {i+1}", type=button_type, key=f"phase_{i}"):
                controller.manual_override(intersection_id, i)
                st.rerun()
    
    # Signal timing configuration
    st.write("Signal Timing:")
    timing_cols = st.columns(2)
    
    with timing_cols[0]:
        min_green = st.slider(
            "Minimum Green Time (s)",
            min_value=5,
            max_value=30,
            value=controller.min_green_time,
            step=5,
            key="min_green"
        )
    
    with timing_cols[1]:
        max_green = st.slider(
            "Maximum Green Time (s)",
            min_value=30,
            max_value=90,
            value=controller.max_green_time,
            step=5,
            key="max_green"
        )
    
    # Update controller if values changed
    if min_green != controller.min_green_time or max_green != controller.max_green_time:
        controller.min_green_time = min_green
        controller.max_green_time = max_green
    
    # Emergency button
    st.write("Emergency Controls:")
    if st.button("🚨 Trigger Emergency Vehicle", type="primary"):
        # This is a demo feature - in a real system it would be triggered by actual detection
        st.session_state.simulator.emergency_vehicles[f"ev_trigger_{intersection_id}"] = {
            'id': f"ev_trigger_{intersection_id}",
            'type': 'ambulance',
            'speed': 80,
            'position': 0,
            'wait_time': 0,
            'direction': 'N',  # Default direction
            'lane_id': f"{intersection_id}_lane_0",  # Default lane
            'intersection_id': intersection_id,
            'arrival_time': datetime.now() + timedelta(seconds=2)
        }
        st.success("Emergency vehicle dispatched!")

def render_metrics_dashboard(traffic_data, predictions, comparison_metrics):
    """
    Render metrics dashboard with prediction visualizations.
    
    Args:
        traffic_data (DataFrame): Current traffic data
        predictions (DataFrame): ML-generated predictions
        comparison_metrics (dict): Metrics for comparison
    """
    # Prediction charts
    if not traffic_data.empty and not predictions.empty:
        # Create congestion prediction chart
        st.subheader("Congestion Predictions")
        
        # Merge current data with predictions
        prediction_data = pd.merge(
            traffic_data[['intersection_id', 'lane_id', 'congestion_level']],
            predictions[['intersection_id', 'lane_id', 'predicted_congestion']],
            on=['intersection_id', 'lane_id']
        )
        
        # Calculate change
        prediction_data['change'] = prediction_data['predicted_congestion'] - prediction_data['congestion_level']
        
        # Create a dataframe for plotting
        plot_data = []
        for _, row in prediction_data.iterrows():
            # Current value
            plot_data.append({
                'intersection_lane': f"{row['intersection_id']} - {row['lane_id']}",
                'value': row['congestion_level'] * 100,
                'type': 'Current'
            })
            
            # Predicted value
            plot_data.append({
                'intersection_lane': f"{row['intersection_id']} - {row['lane_id']}",
                'value': row['predicted_congestion'] * 100,
                'type': 'Predicted'
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create a bar chart
        fig = px.bar(
            plot_df,
            x='intersection_lane',
            y='value',
            color='type',
            barmode='group',
            title='Current vs Predicted Congestion (%)',
            color_discrete_map={'Current': 'blue', 'Predicted': 'red'},
            labels={'value': 'Congestion (%)', 'intersection_lane': 'Location'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_tickangle=-45,
            legend_title_text='',
            height=400
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction confidence
        if 'confidence' in predictions.columns:
            confidence = predictions['confidence'].mean() * 100
            st.metric(
                "Prediction Confidence", 
                f"{confidence:.1f}%",
                delta=None
            )
            
            # Show prediction window
            window_minutes = 5  # default value
            if hasattr(st.session_state, 'predictor'):
                window_minutes = st.session_state.predictor.prediction_window
                
            st.info(f"⏱️ Predictions are for {window_minutes} minutes ahead")
    
    # Performance comparison
    st.subheader("Performance Comparison")
    
    # Create comparison chart
    if (comparison_metrics['traditional']['avg_wait_time'] > 0 or 
        comparison_metrics['ml_based']['avg_wait_time'] > 0):
        
        # Create a dataframe for plotting
        metrics_to_compare = ['avg_wait_time', 'throughput', 'congestion_level']
        labels = ['Average Wait Time (s)', 'Throughput (veh/min)', 'Congestion Level (%)']
        
        comparison_data = []
        for metric, label in zip(metrics_to_compare, labels):
            trad_value = comparison_metrics['traditional'].get(metric, 0)
            ml_value = comparison_metrics['ml_based'].get(metric, 0)
            
            # Scale congestion to percentage
            if metric == 'congestion_level':
                trad_value *= 100
                ml_value *= 100
                
            comparison_data.append({
                'metric': label,
                'value': trad_value,
                'type': 'Traditional'
            })
            
            comparison_data.append({
                'metric': label,
                'value': ml_value,
                'type': 'ML-based'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create a bar chart
        fig = px.bar(
            comparison_df,
            x='metric',
            y='value',
            color='type',
            barmode='group',
            title='Traditional vs ML-based Control',
            color_discrete_map={'Traditional': 'gray', 'ML-based': 'green'},
            labels={'value': 'Value', 'metric': 'Metric'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_tickangle=0,
            legend_title_text='',
            height=400
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate improvement
        if comparison_metrics['traditional']['avg_wait_time'] > 0 and comparison_metrics['ml_based']['avg_wait_time'] > 0:
            wait_improvement = ((comparison_metrics['traditional']['avg_wait_time'] - 
                               comparison_metrics['ml_based']['avg_wait_time']) / 
                              comparison_metrics['traditional']['avg_wait_time'] * 100)
            
            throughput_improvement = ((comparison_metrics['ml_based']['throughput'] - 
                                    comparison_metrics['traditional']['throughput']) / 
                                   comparison_metrics['traditional']['throughput'] * 100)
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(
                    "Wait Time Reduction", 
                    f"{abs(wait_improvement):.1f}%",
                    delta=f"{-wait_improvement:.1f}%" if wait_improvement > 0 else f"{wait_improvement:.1f}%",
                    delta_color="normal"
                )
            
            with metric_col2:
                st.metric(
                    "Throughput Improvement", 
                    f"{abs(throughput_improvement):.1f}%",
                    delta=f"{throughput_improvement:.1f}%" if throughput_improvement > 0 else f"{-throughput_improvement:.1f}%",
                    delta_color="normal"
                )
    else:
        st.info("Run simulations with both traditional and ML-based control to see comparison.")

def _get_congestion_color(congestion_level):
    """Convert congestion level to color."""
    if congestion_level < 0.3:
        return "green"
    elif congestion_level < 0.6:
        return "gold"
    else:
        return "red"
