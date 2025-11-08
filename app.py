# dashboard.py - Fixed method names
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
import os
warnings.filterwarnings('ignore')

class TurbineDashboard:
    def __init__(self):
        self.turbine_monitor = None
        self.anomaly_detector = None
        self.load_ml_models()
        
        self.historical_data = []
        self.max_history = 100
    
    def load_ml_models(self):
        """Try to load ML models with multiple path options"""
        try:
            # Try different possible import approaches
            try:
                from turbine_monitor import TurbineMonitor
                from anomaly_detector import AnomalyDetector
                
                self.turbine_monitor = TurbineMonitor()
                self.anomaly_detector = AnomalyDetector()
                st.success(" ML models loaded successfully!")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load ML models: {e}")
                st.info("🔄 Using simulation mode with calculated values")
                self.turbine_monitor = None
                self.anomaly_detector = None
                
        except Exception as e:
            st.error(f"❌ Critical error: {e}")
            self.turbine_monitor = None
            self.anomaly_detector = None
    
    def calculate_theoretical_power(self, wind_speed, rotor_radius=50, efficiency=0.45):
    
    
    # Validate wind speed range first
        if wind_speed < 0:
            return 0
        elif wind_speed < 3.0:  # Below cut-in
            return 0
        elif wind_speed > 25.0:  # Above cut-out (typical for most turbines)
            return 0
        elif wind_speed > 30.0:  # Physically impossible - sensor error
            return 0
    
        try:
            air_density = 1.225
            swept_area = 3.14159 * (rotor_radius ** 2)
        
        # Realistic power curve modeling
            if wind_speed <= 12.0:  # Below rated wind speed
                power_watts = 0.5 * air_density * swept_area * (wind_speed ** 3) * efficiency
            else:  # Above rated wind speed - power should plateau
            # Power should not increase indefinitely with wind speed
                rated_wind_speed = 12.0
                power_watts = 0.5 * air_density * swept_area * (rated_wind_speed ** 3) * efficiency
        
            power_kw = power_watts / 1000
        
        # Cap at realistic maximum for a 2MW turbine
            max_realistic_power = 2200  # kW (10% over rated)
            return min(power_kw, max_realistic_power)
        
        except Exception as e:
            print(f"Theoretical power calculation error: {e}")
            return 0
    
    def predict_power(self, wind_speed, wind_direction, theoretical_power, timestamp=None):
        """Use ML model if available, otherwise use simulation"""
        if self.turbine_monitor is not None:
            try:
                # Use actual ML model
                return self.turbine_monitor.predict(
                    wind_speed=wind_speed,
                    wind_direction=wind_direction,
                    theoretical_power=theoretical_power,
                    timestamp=timestamp
                )
            except Exception as e:
                st.warning(f"ML prediction failed: {e}, using simulation")
        
        # Fallback simulation
        return self.simulate_power_prediction(wind_speed, theoretical_power)
    
    def simulate_power_prediction(self, wind_speed, theoretical_power):
        """Simulate power prediction when ML models aren't available"""
        if wind_speed < 3.0:
            efficiency = 0.0
        elif wind_speed < 6.0:
            efficiency = 0.3 + (wind_speed - 3.0) * 0.1
        elif wind_speed < 12.0:
            efficiency = 0.6 + (wind_speed - 6.0) * 0.05
        else:
            efficiency = 0.85
            
        variation = np.random.normal(0, 0.02)
        efficiency = max(0.1, min(0.95, efficiency + variation))
        
        return theoretical_power * efficiency
    
    def detect_anomalies(self, wind_speed, theoretical_power, predicted_power, timestamp=None):
        """Use ML anomaly detection if available, otherwise use simulation"""
        if self.anomaly_detector is not None:
            try:
                current_data = {
                    'wind_speed': wind_speed,
                    'theoretical_power': theoretical_power,
                    'power': predicted_power,
                    'timestamp': timestamp or datetime.now()
                }
                return self.anomaly_detector.comprehensive_detection(current_data)
            except Exception as e:
                st.warning(f"ML anomaly detection failed: {e}, using simulation")
        
        # Fallback simulation
        return self.simulate_anomaly_detection(wind_speed, theoretical_power, predicted_power)
    
    def simulate_anomaly_detection(self, wind_speed, theoretical_power, predicted_power):
        """Simulate anomaly detection when ML models aren't available"""
        anomalies = {
            'operational': {},
            'vibration': {},
            'anomaly_score': 0,
            'severity': 'normal'
        }
        
        efficiency = predicted_power / theoretical_power if theoretical_power > 0 else 0
        
        # Your existing anomaly logic
        if wind_speed > 4.0 and predicted_power < 50:
            anomalies['operational']['cut_in_failure'] = True
            anomalies['anomaly_score'] += 2
            
        if wind_speed > 12.0 and efficiency < 0.8:
            anomalies['operational']['rated_power_failure'] = True
            anomalies['anomaly_score'] += 1.5
            
        if wind_speed > 6.0 and efficiency < 0.3:
            anomalies['operational']['low_efficiency'] = True
            anomalies['anomaly_score'] += 1
            
        if wind_speed > 8.0 and np.random.random() < 0.1:
            anomalies['vibration']['high_vibration'] = True
            anomalies['anomaly_score'] += 1.5
        
        if anomalies['anomaly_score'] >= 3:
            anomalies['severity'] = 'critical'
        elif anomalies['anomaly_score'] >= 1:
            anomalies['severity'] = 'warning'
            
        return anomalies

    def calculate_efficiency_metrics(self, theoretical_power, predicted_power, wind_speed, rotor_radius=50):

        rated_power = 2000  # kW
    
    # SIMPLIFIED Power Coefficient - use theoretical power as reference
        if theoretical_power > 0:
            power_coefficient = predicted_power / theoretical_power
        else:
            power_coefficient = 0
    
    # Capacity factor
        if wind_speed < 3.0:
            capacity_factor = 0
        elif wind_speed > 12.0:
            capacity_factor = min(45, (predicted_power / rated_power) * 100)
        else:
            capacity_factor = min(35, (wind_speed - 3.0) / (12.0 - 3.0) * 35)
    
    # Operational Efficiency (same as power coefficient in this approach)
        operational_efficiency = power_coefficient * 100
    
    # Betz Limit Comparison - compare to practical max of 45%
        betz_efficiency = (power_coefficient / 0.45) * 100
    
        return {
            'power_coefficient': min(100, power_coefficient * 100),
            'capacity_factor': min(100, capacity_factor),
            'operational_efficiency': min(120, operational_efficiency),
            'betz_efficiency': min(100, betz_efficiency)
        }
    
    
    
    
    # Component health alerts
        if component_health['blades'] < 70:
            alerts.append("🚨 BLADES: Health critical - Inspect for damage/icing")
        elif component_health['blades'] < 80:
            alerts.append("⚠️ BLADES: Health deteriorating - Schedule inspection")
    
        if component_health['gearbox'] < 65:
            alerts.append("🚨 GEARBOX: Health critical - Check for vibration/issues")
        elif component_health['gearbox'] < 75:
            alerts.append("⚠️ GEARBOX: Health low - Monitor vibration levels")
    
        if component_health['generator'] < 70:
            alerts.append("🚨 GENERATOR: Health critical - Check bearings/temperature")
        elif component_health['generator'] < 80:
            alerts.append("⚠️ GENERATOR: Health deteriorating - Schedule maintenance")
    
        if component_health['control_system'] < 75:
            alerts.append("🚨 CONTROL SYSTEM: Health critical - Check sensors/software")
        elif component_health['control_system'] < 85:
            alerts.append("⚠️ CONTROL SYSTEM: Health low - Verify system operation")
    
    # Anomaly-based alerts
        operational_anomalies = anomalies.get('operational', {})
        if operational_anomalies.get('cut_in_failure'):
            alerts.append("🚨 CUT-IN FAILURE: Turbine not starting - Check brakes/sensors")
    
        if operational_anomalies.get('rated_power_failure'):
            alerts.append("🚨 RATED POWER FAILURE: Not producing at high winds - Check pitch system")
    
        if operational_anomalies.get('low_efficiency'):
            alerts.append("⚠️ LOW EFFICIENCY: Performance degraded - Inspect blades/system")
    
        vibration_anomalies = anomalies.get('vibration', {})
        if vibration_anomalies.get('high_vibration'):
            alerts.append("🚨 HIGH VIBRATION: Mechanical issue - Check bearings/alignment")
    
    # Overall health alert
        if component_health['overall'] < 60:
            alerts.append("🚨 CRITICAL: Overall turbine health critical - Immediate inspection required!")
        elif component_health['overall'] < 75:
            alerts.append("⚠️ WARNING: Turbine health deteriorating - Schedule maintenance")
    
        return alerts

    def get_urgent_alerts(self, component_health, anomalies):
        """Get only critical alerts that need immediate attention"""
        alerts = self.check_component_alerts(component_health, anomalies)
        return [alert for alert in alerts if "🚨" in alert]
    
    def process_turbine_data(self, wind_speed, wind_direction):
        """Process incoming turbine data using your actual ML models"""
        timestamp = datetime.now()
        
        # Calculate theoretical power
        theoretical_power = self.calculate_theoretical_power(wind_speed)
        
        # Get predicted power from your ML model - FIXED METHOD NAME
        predicted_power = self.predict_power(wind_speed, wind_direction, theoretical_power, timestamp)
        
        # Calculate efficiency metrics
        efficiency_metrics = self.calculate_efficiency_metrics(theoretical_power, predicted_power, wind_speed)
        
        # Detect anomalies using your ML model - FIXED METHOD NAME
        anomalies = self.detect_anomalies(wind_speed, theoretical_power, predicted_power, timestamp)
        
        # Prepare result
        result = {
            'timestamp': timestamp,
            'power_performance': {
                'wind_speed': wind_speed,
                'theoretical_power': theoretical_power,
                'predicted_power': predicted_power,
                'power_ratio': (predicted_power / theoretical_power * 100) if theoretical_power > 0 else 0
            },
            'efficiency_analysis': efficiency_metrics,
            'turbine_health': {
                'anomalies_detected': len(anomalies.get('operational', {})) + len(anomalies.get('vibration', {})),
                'anomaly_details': anomalies,
                
            'model_status': 'ML Models Active' if self.turbine_monitor else 'Fallback Mode'
        }}
        
        # Store historical data
        self.historical_data.append(result)
        if len(self.historical_data) > self.max_history:
            self.historical_data.pop(0)
            
        return result

# Initialize dashboard
dashboard = TurbineDashboard()

def main():
    st.set_page_config(
        page_title="Wind Turbine Monitoring Dashboard",
        page_icon="",
        layout="wide"
    )
    
    st.title(" Wind Turbine Monitoring Dashboard")
    
    # Display model status
    if dashboard.turbine_monitor:
        st.success(" ML Models Integrated - Using Real Predictions")
    else:
        st.warning("⚠️ Fallback Mode - Using Simplified Calculations")
    
    st.markdown("---")
    
    # Sidebar for manual input
    st.sidebar.header("Turbine Input Parameters")
    
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 35.0, 8.0, 0.1)
    wind_direction = st.sidebar.slider("Wind Direction (°)", 0, 360, 245, 1)
    
    if st.sidebar.button("Process Data"):
        with st.spinner("Processing turbine data with ML models..."):
            result = dashboard.process_turbine_data(wind_speed, wind_direction)
            display_dashboard(result)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)")
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Display historical data if available
    if dashboard.historical_data:
        display_historical_analysis()

def display_dashboard(result):
    """Display the main dashboard with current metrics"""
    
    # Model status indicator
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.info(f"**Status:** {result.get('model_status', 'Unknown')}")
    
    # Top Row - Summary Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=" Predicted Power",
            value=f"{result['power_performance']['predicted_power']:.0f} kW",
            delta=None
        )
    
    with col2:
        st.metric(
            label=" Power Coefficient",
            value=f"{result['efficiency_analysis']['power_coefficient']:.1f}%",
            delta=None
        )
    
    with col3:
        anomalies_count = result['turbine_health']['anomalies_detected']
        anomaly_status = "Normal" if anomalies_count == 0 else "Warning" if anomalies_count <= 2 else "Critical"
        status_color = "green" if anomaly_status == "Normal" else "orange" if anomaly_status == "Warning" else "red"
        st.metric(
            label=" Anomaly Status",
            value=anomaly_status,
            delta=f"{anomalies_count} issues" if anomalies_count > 0 else None
        )
    
    st.markdown("---")
    
    
    # Middle Row - Charts
    col1, col2 = st.columns(2)
    
    with col1:
        display_power_curve_chart(result)
    
    with col2:
        display_efficiency_chart(result)
    
    st.markdown("---")
    
    # Bottom Row - Detailed Panels
    col1, col2 = st.columns(2)
    
    with col1:
        display_anomaly_details(result)
    

def display_power_curve_chart(result):
    """Display power curve visualization"""
    st.subheader(" Power Performance")
    
    # Create sample power curve data
    wind_speeds = np.linspace(0, 25, 50)
    theoretical_powers = [dashboard.calculate_theoretical_power(ws) for ws in wind_speeds]
    
    fig = go.Figure()
    
    # Theoretical power curve
    fig.add_trace(go.Scatter(
        x=wind_speeds, 
        y=theoretical_powers,
        mode='lines',
        name='Theoretical Power Curve',
        line=dict(color='blue', dash='dash')
    ))
    
    # Current operating point
    current_ws = result['power_performance']['wind_speed']
    current_power = result['power_performance']['predicted_power']
    
    fig.add_trace(go.Scatter(
        x=[current_ws], 
        y=[current_power],
        mode='markers',
        name='Current Operation',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig.update_layout(
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Power (kW)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Power metrics table
    power_data = {
        'Metric': ['Wind Speed', 'Theoretical Power', 'Predicted Power', 'Power Ratio'],
        'Value': [
            f"{current_ws:.1f} m/s",
            f"{result['power_performance']['theoretical_power']:.0f} kW",
            f"{current_power:.0f} kW",
            f"{result['power_performance']['power_ratio']:.1f}%"
        ]
    }
    st.dataframe(power_data, use_container_width=True, hide_index=True)

def display_efficiency_chart(result):
    """Display efficiency metrics"""
    st.subheader(" Efficiency Analysis")
    
    efficiency_data = result['efficiency_analysis']
    metrics = ['Power Coefficient', 'Capacity Factor', 'Betz Efficiency']
    values = [
        efficiency_data['power_coefficient'],
        efficiency_data['capacity_factor'], 
        efficiency_data['betz_efficiency']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
    ])
    
    fig.update_layout(
        yaxis_title="Efficiency (%)",
        yaxis_range=[0, 100],
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency metrics table
    eff_data = {
        'Efficiency Metric': ['Power Coefficient (Cp)', 'Capacity Factor', 'Operational Efficiency', 'Betz Limit Achievement'],
        'Value': [
            f"{efficiency_data['power_coefficient']:.1f}%",
            f"{efficiency_data['capacity_factor']:.1f}%", 
            f"{efficiency_data['operational_efficiency']:.1f}%",
            f"{efficiency_data['betz_efficiency']:.1f}%"
        ],
        'Status': [
            ' Good' if efficiency_data['power_coefficient'] > 35 else ' Low',
            ' Good' if efficiency_data['capacity_factor'] > 25 else ' Low',
            ' Good' if efficiency_data['operational_efficiency'] > 70 else ' Low', 
            ' Good' if efficiency_data['betz_efficiency'] > 60 else ' Low'
        ]
    }
    st.dataframe(eff_data, use_container_width=True, hide_index=True)

    
def display_anomaly_details(result):
    """Display anomaly details and alerts"""
    st.subheader(" Anomaly Detection")
    
    anomalies = result['turbine_health']['anomaly_details']
    anomaly_count = result['turbine_health']['anomalies_detected']
    
    if anomaly_count == 0:
        st.success(" No anomalies detected - System operating normally")
    else:
        st.warning(f" {anomaly_count} anomaly(s) detected")
        
        # Display wind speed anomalies first (most critical)
        wind_anomalies = anomalies.get('wind_speed', {})
        if wind_anomalies:
            st.write("**Wind Speed Issues:**")
            for anomaly, status in wind_anomalies.items():
                if status:
                    st.error(f"• {anomaly.replace('_', ' ').title()}")
        
        # Display operational anomalies
        operational_anomalies = anomalies.get('operational', {})
        if operational_anomalies:
            st.write("**Operational Issues:**")
            for anomaly, status in operational_anomalies.items():
                if status:
                    st.error(f"• {anomaly.replace('_', ' ').title()}")
        
        # ... rest of your display code

def display_historical_analysis():
    """Display historical trends if data is available"""
    if len(dashboard.historical_data) < 2:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Historical Analysis")
    
    if st.sidebar.button("Show Trends"):
        st.markdown("---")
        st.subheader("📊 Historical Trends")
        
        # Prepare historical data
        timestamps = [data['timestamp'] for data in dashboard.historical_data]
        powers = [data['power_performance']['predicted_power'] for data in dashboard.historical_data]
        efficiencies = [data['efficiency_analysis']['power_coefficient'] for data in dashboard.historical_data]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health trend
            fig_health = go.Figure()
            fig_health.add_trace(go.Scatter(
                x=timestamps, y=health_scores, mode='lines+markers', name='Health Score'
            ))
            fig_health.update_layout(title="Health Score Trend", height=300)
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col2:
            # Power trend
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(
                x=timestamps, y=powers, mode='lines+markers', name='Predicted Power'
            ))
            fig_power.update_layout(title="Power Output Trend", height=300)
            st.plotly_chart(fig_power, use_container_width=True)

if __name__ == "__main__":
    main()