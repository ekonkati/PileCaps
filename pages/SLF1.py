# To run the Streamlit app, execute this command in your terminal:
# streamlit run your_script_name.py

# Or run it directly from Jupyter using:
import subprocess
import sys

# Save the streamlit code to a file first
streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import json

# Set page config
st.set_page_config(
    page_title="Landfill Design & Stability App",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏗️ Landfill Design, Stability, Visualization & Submission App")
st.markdown("### CPCB/EPA Compliant Landfill Design Tool")

# Sidebar for navigation
st.sidebar.title("Navigation")
module = st.sidebar.selectbox(
    "Select Module",
    [
        "Project Setup",
        "Waste Type & Site Data", 
        "Geometry Builder",
        "Liner System Design",
        "Slope Stability Analysis",
        "BOQ & Costing",
        "Fill Sequencing",
        "Reports & Export"
    ]
)

# Project Setup Module
if module == "Project Setup":
    st.header("📋 Project Setup & Site Data")
    
    # Admin settings
    col1, col2 = st.columns(2)
    with col1:
        regulatory_standard = st.selectbox(
            "Regulatory Standard",
            ["CPCB (India)", "EPA (USA)"],
            help="Select the regulatory framework for compliance"
        )
    
    with col2:
        unit_system = st.selectbox(
            "Unit System",
            ["SI (Metric)", "Imperial"],
            help="Choose measurement units for the project"
        )
    
    st.divider()
    
    # Project Information
    st.subheader("Project Information")
    col1, col2 = st.columns(2)
    
    with col1:
        project_name = st.text_input(
            "Project Name *",
            placeholder="Enter project name"
        )
        
        project_location = st.text_input(
            "Project Location",
            placeholder="City, State/Province, Country"
        )
        
        coordinate_system = st.selectbox(
            "Coordinate System",
            ["WGS84", "UTM Zone 43N", "UTM Zone 44N", "Custom"],
            help="Select the coordinate reference system"
        )
    
    with col2:
        latitude = st.number_input(
            "Latitude (°) *",
            min_value=-90.0,
            max_value=90.0,
            format="%.6f",
            help="Site latitude in decimal degrees"
        )
        
        longitude = st.number_input(
            "Longitude (°) *",
            min_value=-180.0,
            max_value=180.0,
            format="%.6f",
            help="Site longitude in decimal degrees"
        )
        
        ground_level = st.number_input(
            "Average Ground Level (m)",
            min_value=0.0,
            value=100.0,
            help="Average ground elevation above sea level"
        )
    
    # Site Data
    st.subheader("Site Data")
    col1, col2 = st.columns(2)
    
    with col1:
        water_table_depth = st.number_input(
            "Water Table Depth (m) *",
            min_value=0.1,
            value=5.0,
            help="Depth to groundwater table from surface"
        )
        
        # Warning for shallow water table
        if water_table_depth < 2.0:
            st.warning("⚠️ Water table depth < 2m. Consider additional protective measures.")
    
    with col2:
        seismic_zone = st.selectbox(
            "Seismic Zone",
            ["Zone I (Low)", "Zone II (Moderate)", "Zone III (Moderate)", "Zone IV (High)", "Zone V (Very High)"],
            help="Seismic zone classification for the site"
        )
        
        pga_value = st.number_input(
            "Peak Ground Acceleration (g)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            format="%.3f",
            help="Expected peak ground acceleration for seismic analysis"
        )
    
    # Validation
    st.divider()
    if st.button("Validate Project Setup", type="primary"):
        errors = []
        if not project_name:
            errors.append("Project name is required")
        if latitude == 0.0:
            errors.append("Latitude must be specified")
        if longitude == 0.0:
            errors.append("Longitude must be specified")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            st.success("✅ Project setup validated successfully!")
            
            # Store in session state
            st.session_state.project_data = {
                'name': project_name,
                'location': project_location,
                'latitude': latitude,
                'longitude': longitude,
                'coordinate_system': coordinate_system,
                'ground_level': ground_level,
                'water_table_depth': water_table_depth,
                'seismic_zone': seismic_zone,
                'pga': pga_value,
                'regulatory_standard': regulatory_standard,
                'unit_system': unit_system
            }

# Waste Type & Site Data Module
if module == "Waste Type & Site Data":
    st.header("🗑️ Waste Type & Site Data")
    
    # Waste Type Selection
    st.subheader("Waste Classification")
    waste_type = st.selectbox(
        "Waste Type *",
        ["MSW (Municipal Solid Waste)", "Hazardous Waste"],
        help="Select the primary waste type for the landfill"
    )
    
    # Display different requirements based on waste type
    if waste_type == "Hazardous Waste":
        st.info("ℹ️ Hazardous waste requires enhanced liner systems and monitoring")
        hazard_class = st.multiselect(
            "Hazard Classification",
            ["Class I - Ignitable", "Class II - Corrosive", "Class III - Reactive", "Class IV - Toxic"],
            help="Select applicable hazard classifications"
        )
    else:
        st.info("ℹ️ MSW landfill with standard liner and monitoring requirements")
    
    st.divider()
    
    # Waste Characteristics
    st.subheader("Waste Characteristics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        waste_inflow_tpd = st.number_input(
            "Waste Inflow (TPD) *",
            min_value=1.0,
            value=100.0,
            help="Tonnes per day of waste input"
        )
        
        waste_density = st.number_input(
            "Waste Density (t/m³) *",
            min_value=0.1,
            max_value=2.0,
            value=0.6 if waste_type == "MSW (Municipal Solid Waste)" else 1.2,
            format="%.2f",
            help="Compacted waste density"
        )
    
    with col2:
        compaction_factor = st.number_input(
            "Compaction Factor",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            format="%.1f",
            help="Ratio of loose to compacted waste volume"
        )
        
        moisture_content = st.number_input(
            "Moisture Content (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            format="%.1f",
            help="Waste moisture content percentage"
        )
    
    with col3:
        organic_content = st.number_input(
            "Organic Content (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0 if waste_type == "MSW (Municipal Solid Waste)" else 20.0,
            format="%.1f",
            help="Biodegradable organic matter percentage"
        )
        
        calorific_value = st.number_input(
            "Calorific Value (kJ/kg)",
            min_value=0.0,
            value=8000.0 if waste_type == "MSW (Municipal Solid Waste)" else 15000.0,
            help="Energy content of waste"
        )
    
    st.divider()
    
    # Lifespan Calculation
    st.subheader("Landfill Capacity & Lifespan")
    col1, col2 = st.columns(2)
    
    with col1:
        calculation_method = st.radio(
            "Calculation Method",
            ["Specify Target Lifespan", "Calculate from Total Capacity"],
            help="Choose how to determine landfill lifespan"
        )
        
        if calculation_method == "Specify Target Lifespan":
            target_lifespan_years = st.number_input(
                "Target Lifespan (years) *",
                min_value=1,
                max_value=50,
                value=20,
                help="Desired operational lifespan"
            )
            
            # Calculate total capacity needed
            total_waste_tonnes = waste_inflow_tpd * 365 * target_lifespan_years
            total_volume_m3 = total_waste_tonnes / waste_density
            
            st.metric("Total Waste Capacity Required", f"{total_waste_tonnes:,.0f} tonnes")
            st.metric("Total Volume Required", f"{total_volume_m3:,.0f} m³")
        
        else:
            total_capacity_m3 = st.number_input(
                "Total Landfill Capacity (m³) *",
                min_value=1000.0,
                value=500000.0,
                help="Available landfill volume"
            )
            
            # Calculate lifespan
            daily_volume_m3 = waste_inflow_tpd / waste_density
            lifespan_days = total_capacity_m3 / daily_volume_m3
            lifespan_years = lifespan_days / 365
            
            st.metric("Calculated Lifespan", f"{lifespan_years:.1f} years")
            st.metric("Daily Volume Requirement", f"{daily_volume_m3:.0f} m³/day")
    
    with col2:
        # Settlement and gas generation estimates
        st.subheader("Design Considerations")
        
        settlement_factor = st.number_input(
            "Settlement Factor (%)",
            min_value=0.0,
            max_value=50.0,
            value=15.0 if waste_type == "MSW (Municipal Solid Waste)" else 5.0,
            format="%.1f",
            help="Expected waste settlement over time"
        )
        
        gas_generation_rate = st.number_input(
            "Gas Generation Rate (m³/t/year)",
            min_value=0.0,
            value=50.0 if waste_type == "MSW (Municipal Solid Waste)" else 10.0,
            format="%.1f",
            help="Expected landfill gas generation"
        )
        
        leachate_generation = st.number_input(
            "Leachate Generation (L/t/year)",
            min_value=0.0,
            value=200.0 if waste_type == "MSW (Municipal Solid Waste)" else 100.0,
            format="%.1f",
            help="Expected leachate production"
        )
    
    # Validation and storage
    st.divider()
    if st.button("Validate Waste Data", type="primary"):
        errors = []
        if waste_inflow_tpd <= 0:
            errors.append("Waste inflow must be greater than 0")
        if waste_density <= 0:
            errors.append("Waste density must be greater than 0")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            st.success("✅ Waste data validated successfully!")
            
            # Store in session state
            waste_data = {
                'waste_type': waste_type,
                'waste_inflow_tpd': waste_inflow_tpd,
                'waste_density': waste_density,
                'compaction_factor': compaction_factor,
                'moisture_content': moisture_content,
                'organic_content': organic_content,
                'calorific_value': calorific_value,
                'settlement_factor': settlement_factor,
                'gas_generation_rate': gas_generation_rate,
                'leachate_generation': leachate_generation
            }
            
            if waste_type == "Hazardous Waste":
                waste_data['hazard_class'] = hazard_class
            
            if calculation_method == "Specify Target Lifespan":
                waste_data['target_lifespan_years'] = target_lifespan_years
                waste_data['total_capacity_m3'] = total_volume_m3
            else:
                waste_data['total_capacity_m3'] = total_capacity_m3
                waste_data['calculated_lifespan_years'] = lifespan_years
            
            st.session_state.waste_data = waste_data

# Geometry Builder Module
if module == "Geometry Builder":
    st.header("📐 Geometry Builder")
    st.info("Define the landfill's physical dimensions and visualize the cross-section.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Design Inputs")
        footprint_length = st.number_input("Footprint Length (m)", min_value=10.0, value=200.0)
        footprint_width = st.number_input("Footprint Width (m)", min_value=10.0, value=150.0)
        depth_below_ground = st.number_input("Depth Below Ground (m)", min_value=0.0, value=5.0)
        total_height_above_ground = st.number_input("Total Height Above Ground (m)", min_value=5.0, value=30.0)
        inside_slope = st.slider("Inside Slope (H:V)", 1.0, 5.0, 3.0, 0.1)
        outside_slope = st.slider("Outside Slope (H:V)", 1.0, 5.0, 3.0, 0.1)
        berm_width = st.number_input("Berm Width (m)", min_value=0.0, value=5.0)
        berm_interval_height = st.number_input("Berm Vertical Interval (m)", min_value=1.0, value=10.0)

    # Calculations for visualization and volume
    base_area = footprint_length * footprint_width
    top_width = footprint_width - 2 * (total_height_above_ground / outside_slope)
    top_length = footprint_length - 2 * (total_height_above_ground / outside_slope)
    top_area = top_length * top_width if top_length > 0 and top_width > 0 else 0
    
    # Approximate volume using prismoidal formula
    mid_area = ((footprint_length + top_length) / 2) * ((footprint_width + top_width) / 2)
    volume_above_ground = (total_height_above_ground / 6) * (base_area + 4 * mid_area + top_area)
    volume_below_ground = base_area * depth_below_ground
    total_volume = volume_above_ground + volume_below_ground

    with col2:
        st.subheader("2D Cross-Section (Width)")
        fig = go.Figure()
        
        # Ground level
        fig.add_shape(type="line", x0=-footprint_width, y0=0, x1=footprint_width, y1=0, line=dict(color="Green", width=2, dash="dash"), name="Ground")

        # Excavation
        x_excavation = [-footprint_width/2, -footprint_width/2, footprint_width/2, footprint_width/2]
        y_excavation = [0, -depth_below_ground, -depth_below_ground, 0]
        fig.add_trace(go.Scatter(x=x_excavation, y=y_excavation, fill="toself", fillcolor="rgba(139, 69, 19, 0.3)", line_color="saddlebrown", name="Excavation"))

        # Above ground structure with berms
        num_berms = math.floor(total_height_above_ground / berm_interval_height)
        x_coords_left = [-footprint_width/2]
        y_coords_left = [0]
        current_height = 0
        current_width = -footprint_width/2

        while current_height < total_height_above_ground:
            step_height = min(berm_interval_height, total_height_above_ground - current_height)
            current_width += step_height * outside_slope # Corrected from division to multiplication
            current_height += step_height
            x_coords_left.append(current_width)
            y_coords_left.append(current_height)
            if current_height < total_height_above_ground and berm_width > 0:
                current_width += berm_width
                x_coords_left.append(current_width)
                y_coords_left.append(current_height)

        # Mirror for the right side
        x_coords_right = [-x for x in reversed(x_coords_left)]
        y_coords_right = list(reversed(y_coords_left))
        
        x_fill = x_coords_left + x_coords_right
        y_fill = y_coords_left + y_coords_right

        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself", fillcolor="rgba(128, 128, 128, 0.5)", line_color="gray", name="Landfill Body"))
        
        fig.update_layout(
            title="Schematic Landfill Cross-Section",
            xaxis_title="Width (m)",
            yaxis_title="Height (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            legend_title="Components"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    st.subheader("Calculated Geometry")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Base Area", f"{base_area:,.0f} m²")
    col2.metric("Top Area", f"{top_area:,.0f} m²")
    col3.metric("Estimated Total Volume", f"{total_volume:,.0f} m³")

    if st.button("Save Geometry", type="primary"):
        st.session_state.geometry_data = {
            'footprint_length': footprint_length,
            'footprint_width': footprint_width,
            'depth_below_ground': depth_below_ground,
            'total_height_above_ground': total_height_above_ground,
            'outside_slope': outside_slope,
            'total_volume': total_volume
        }
        st.success("✅ Geometry data saved!")

# Liner System Design Module
if module == "Liner System Design":
    st.header("🛡️ Liner System Design")
    st.info("Design the liner system based on waste type and regulatory requirements.")

    # Check for previous data
    if 'project_data' not in st.session_state or 'waste_data' not in st.session_state:
        st.warning("Please complete the 'Project Setup' and 'Waste Type & Site Data' modules first.")
    else:
        project_data = st.session_state.project_data
        waste_data = st.session_state.waste_data

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Liner Configuration")
            
            # Standard configurations
            liner_configs = {
                "Custom": {},
                "CPCB (MSW)": {
                    "Compacted Clay Liner (m)": 1.0,
                    "HDPE Geomembrane (mm)": 1.5,
                    "Geotextile Filter (g/m²)": 300,
                    "Drainage Layer (m)": 0.3
                },
                "EPA Subtitle D (MSW)": {
                    "Compacted Clay Liner (m)": 0.6,
                    "HDPE Geomembrane (mm)": 1.5,
                    "Drainage Layer (m)": 0.3
                },
                "EPA Subtitle C (Hazardous)": {
                    "Top HDPE Geomembrane (mm)": 1.5,
                    "Primary Drainage Layer (m)": 0.3,
                    "Bottom HDPE Geomembrane (mm)": 1.5,
                    "Geosynthetic Clay Liner (GCL)": 1,
                    "Compacted Clay Liner (m)": 0.9
                }
            }
            
            config_choice = st.selectbox("Choose a standard configuration or build a custom one", list(liner_configs.keys()))
            
            st.divider()
            st.subheader("Liner Components")
            
            # Use chosen config or allow custom input
            current_config = liner_configs[config_choice]
            
            layers = {}
            layers['Compacted Clay Liner (m)'] = st.number_input("Compacted Clay Liner (m)", 0.0, 5.0, current_config.get("Compacted Clay Liner (m)", 0.0), 0.1)
            layers['HDPE Geomembrane (mm)'] = st.number_input("HDPE Geomembrane (mm)", 0.0, 5.0, current_config.get("HDPE Geomembrane (mm)", 1.5), 0.1)
            layers['Geotextile Filter (g/m²)'] = st.number_input("Geotextile Filter (g/m²)", 0, 1000, current_config.get("Geotextile Filter (g/m²)", 0), 50)
            layers['Drainage Layer (m)'] = st.number_input("Drainage Layer (m)", 0.0, 2.0, current_config.get("Drainage Layer (m)", 0.3), 0.1)

            if st.button("Save Liner System", type="primary"):
                st.session_state.liner_data = layers
                st.success("✅ Liner system data saved!")

        with col2:
            st.subheader("Liner System Schematic")
            
            # Filter layers with thickness > 0 for visualization
            vis_layers = {k: v for k, v in layers.items() if v > 0}
            if not vis_layers:
                st.info("Add layers to see the schematic.")
            else:
                layer_names = list(vis_layers.keys())
                layer_thicknesses = list(vis_layers.values())
                
                fig = go.Figure()
                
                current_y = 0
                colors = px.colors.qualitative.Plotly
                for i, (name, thickness) in enumerate(vis_layers.items()):
                    # Normalize thickness for better visualization
                    vis_thickness = np.log1p(thickness * 10) # Log scale for better display
                    fig.add_trace(go.Bar(y=['Liner'], x=[vis_thickness], name=f"{name} ({thickness} units)", orientation='h', marker_color=colors[i % len(colors)]))
                
                fig.update_layout(
                    title_text='Liner System Layers (Not to Scale)',
                    barmode='stack',
                    xaxis=dict(showticklabels=False, title=""),
                    yaxis=dict(showticklabels=False, title=""),
                    showlegend=True,
                    legend_title="Layers (from bottom up)"
                )
                st.plotly_chart(fig, use_container_width=True)

# Slope Stability Analysis Module
if module == "Slope Stability Analysis":
    st.header("⚖️ Slope Stability Analysis")
    st.info("Enter geotechnical parameters to calculate the Factor of Safety (FoS) for the landfill slopes.")

    if 'geometry_data' not in st.session_state:
        st.warning("Please complete and save the 'Geometry Builder' module first.")
    else:
        geometry = st.session_state.geometry_data
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Geotechnical Parameters")
            
            with st.expander("Waste Mass Properties"):
                waste_unit_weight = st.number_input("Unit Weight (kN/m³)", 10.0, 25.0, 18.0, 0.5)
                waste_cohesion = st.number_input("Cohesion (kPa)", 0.0, 50.0, 5.0, 1.0)
                waste_friction_angle = st.slider("Friction Angle (°)", 0, 45, 30)

            with st.expander("Foundation Soil Properties"):
                foundation_unit_weight = st.number_input("Foundation Unit Weight (kN/m³)", 12.0, 28.0, 20.0, 0.5)
                foundation_cohesion = st.number_input("Foundation Cohesion (kPa)", 0.0, 100.0, 20.0, 2.0)
                foundation_friction_angle = st.slider("Foundation Friction Angle (°)", 0, 50, 35)

            analysis_method = st.selectbox("Analysis Method", ["Bishop's Simplified Method (Circular)"])
            run_analysis = st.button("Calculate Factor of Safety", type="primary")

        with col2:
            st.subheader("Analysis Results")
            if run_analysis:
                # Simplified FoS Calculation (Illustrative)
                slope_angle_rad = np.arctan(1 / geometry['outside_slope'])
                slope_angle_deg = np.degrees(slope_angle_rad)
                
                # Simplified calculation assuming failure plane through waste mass only
                c = waste_cohesion
                phi = np.radians(waste_friction_angle)
                gamma = waste_unit_weight
                H = geometry['total_height_above_ground']
                
                # Driving Force (approx)
                driving_force = gamma * H * np.sin(slope_angle_rad)
                # Resisting Force (approx)
                resisting_force = c + (gamma * H * np.cos(slope_angle_rad) * np.tan(phi))
                
                fos = resisting_force / driving_force if driving_force > 0 else 999

                st.metric("Factor of Safety (FoS)", f"{fos:.2f}")
                if fos >= 1.5:
                    st.success("✅ Slope is considered stable under static conditions.")
                elif 1.2 <= fos < 1.5:
                    st.warning("⚠️ Slope is marginally stable. Further analysis is recommended.")
                else:
                    st.error("❌ Slope may be unstable. Design revisions are required.")
                
                # Visualization with slip circle
                # (This is a schematic representation)
                fig = go.Figure()
                # Re-draw geometry
                x_fill = [-geometry['footprint_width']/2, geometry['footprint_width']/2 - geometry['total_height_above_ground']*geometry['outside_slope'], geometry['footprint_width']/2, -geometry['footprint_width']/2]
                y_fill = [0, geometry['total_height_above_ground'], 0, 0]
                fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself", fillcolor="rgba(128, 128, 128, 0.5)", line_color="gray", name="Landfill Body"))
                
                # Add slip circle
                center_x = (geometry['footprint_width']/2 - geometry['total_height_above_ground']*geometry['outside_slope']) / 2
                center_y = geometry['total_height_above_ground'] * 1.5
                radius = center_y
                fig.add_shape(type="circle", xref="x", yref="y", x0=center_x-radius, y0=center_y-radius, x1=center_x+radius, y1=center_y+radius, line_color="Red", line_width=2, line_dash="dash")

                fig.update_layout(title="Schematic with Potential Slip Surface", yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enter parameters and click 'Calculate' to see results.")

# BOQ & Costing Module
if module == "BOQ & Costing":
    st.header("💰 BOQ & Costing")
    st.info("Generate a preliminary Bill of Quantities (BOQ) and cost estimate.")

    if 'geometry_data' not in st.session_state or 'liner_data' not in st.session_state:
        st.warning("Please complete and save the 'Geometry Builder' and 'Liner System Design' modules first.")
    else:
        geometry = st.session_state.geometry_data
        liner = st.session_state.liner_data
        
        st.subheader("Unit Costs")
        col1, col2, col3, col4 = st.columns(4)
        cost_excavation = col1.number_input("Excavation Cost ($/m³)", 1.0, 100.0, 10.0)
        cost_clay = col2.number_input("Clay Liner Cost ($/m³)", 10.0, 200.0, 50.0)
        cost_geomembrane = col3.number_input("Geomembrane Cost ($/m²)", 5.0, 50.0, 15.0)
        cost_drainage = col4.number_input("Drainage Layer Cost ($/m³)", 20.0, 250.0, 75.0)
        
        st.divider()
        
        # Calculations
        excavation_vol = geometry['footprint_length'] * geometry['footprint_width'] * geometry['depth_below_ground']
        
        # Approximate liner area (base + sloped sides)
        base_area = geometry['footprint_length'] * geometry['footprint_width']
        side_slope_length = np.sqrt(geometry['total_height_above_ground']**2 + (geometry['total_height_above_ground'] * geometry['outside_slope'])**2)
        perimeter = 2 * (geometry['footprint_length'] + geometry['footprint_width'])
        side_area = perimeter * side_slope_length # Approximation
        total_liner_area = base_area + side_area

        boq_data = []
        boq_data.append({"Item": "Site Excavation", "Quantity": excavation_vol, "Unit": "m³", "Unit Cost": cost_excavation, "Total Cost": excavation_vol * cost_excavation})
        
        if liner.get('Compacted Clay Liner (m)', 0) > 0:
            clay_vol = total_liner_area * liner['Compacted Clay Liner (m)']
            boq_data.append({"Item": "Compacted Clay Liner", "Quantity": clay_vol, "Unit": "m³", "Unit Cost": cost_clay, "Total Cost": clay_vol * cost_clay})

        if liner.get('HDPE Geomembrane (mm)', 0) > 0:
            boq_data.append({"Item": "HDPE Geomembrane", "Quantity": total_liner_area, "Unit": "m²", "Unit Cost": cost_geomembrane, "Total Cost": total_liner_area * cost_geomembrane})

        if liner.get('Drainage Layer (m)', 0) > 0:
            drainage_vol = total_liner_area * liner['Drainage Layer (m)']
            boq_data.append({"Item": "Drainage Layer", "Quantity": drainage_vol, "Unit": "m³", "Unit Cost": cost_drainage, "Total Cost": drainage_vol * cost_drainage})
            
        boq_df = pd.DataFrame(boq_data)
        boq_df["Quantity"] = boq_df["Quantity"].map('{:,.2f}'.format)
        boq_df["Unit Cost"] = boq_df["Unit Cost"].map('${:,.2f}'.format)
        boq_df["Total Cost"] = boq_df["Total Cost"].map('${:,.2f}'.format)
        
        st.subheader("Bill of Quantities")
        st.dataframe(boq_df, use_container_width=True)
        
        total_cost = sum(item['Total Cost'] for item in boq_data)
        st.metric("Estimated Total Project Cost", f"${total_cost:,.2f}")

# Fill Sequencing Module
if module == "Fill Sequencing":
    st.header("🎬 Fill Sequencing")
    st.info("Visualize the landfill filling process over time.")

    if 'geometry_data' not in st.session_state or 'waste_data' not in st.session_state:
        st.warning("Please complete 'Geometry Builder' and 'Waste Type & Site Data' modules first.")
    else:
        geometry = st.session_state.geometry_data
        waste = st.session_state.waste_data
        
        lifespan = waste.get('target_lifespan_years', waste.get('calculated_lifespan_years', 20))
        
        selected_year = st.slider("Select Year", 1, int(lifespan), 1)
        
        fill_percentage = selected_year / lifespan
        current_fill_height = geometry['total_height_above_ground'] * fill_percentage
        
        st.subheader(f"Fill Status at Year {selected_year}")
        
        fig = go.Figure()
        
        # Draw the full landfill geometry as a background
        x_full = [-geometry['footprint_width']/2, geometry['footprint_width']/2 - geometry['total_height_above_ground']*geometry['outside_slope'], geometry['footprint_width']/2, -geometry['footprint_width']/2]
        y_full = [0, geometry['total_height_above_ground'], 0, 0]
        fig.add_trace(go.Scatter(x=x_full, y=y_full, fill="toself", fillcolor="rgba(211, 211, 211, 0.5)", line_color="gray", name="Full Capacity"))

        # Draw the current fill level
        x_fill_current = [-geometry['footprint_width']/2, geometry['footprint_width']/2 - current_fill_height*geometry['outside_slope'], geometry['footprint_width']/2, -geometry['footprint_width']/2]
        y_fill_current = [0, current_fill_height, 0, 0]
        fig.add_trace(go.Scatter(x=x_fill_current, y=y_fill_current, fill="toself", fillcolor="rgba(139, 69, 19, 0.7)", line_color="saddlebrown", name=f"Fill at Year {selected_year}"))
        
        fig.update_layout(
            title="Fill Sequencing Visualization",
            xaxis_title="Width (m)",
            yaxis_title="Height (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1, range=[-geometry['depth_below_ground']-5, geometry['total_height_above_ground']+5]),
            legend_title="Components"
        )
        st.plotly_chart(fig, use_container_width=True)

# Reports & Export Module
if module == "Reports & Export":
    st.header("📄 Reports & Export")
    st.info("Generate a summary report and export data.")

    required_modules = ['project_data', 'waste_data', 'geometry_data', 'liner_data']
    if not all(module in st.session_state for module in required_modules):
        st.warning("Please complete all previous modules to generate a full report.")
    else:
        st.subheader("Project Summary")
        
        # Combine all data
        full_report_data = {
            "Project Information": st.session_state.project_data,
            "Waste Characteristics": st.session_state.waste_data,
            "Landfill Geometry": st.session_state.geometry_data,
            "Liner System Design": st.session_state.liner_data
        }
        
        st.json(full_report_data)

        # Create downloadable files
        report_text = json.dumps(full_report_data, indent=4)
        
        # Re-calculate BOQ for export
        geometry = st.session_state.geometry_data
        liner = st.session_state.liner_data
        excavation_vol = geometry['footprint_length'] * geometry['footprint_width'] * geometry['depth_below_ground']
        base_area = geometry['footprint_length'] * geometry['footprint_width']
        side_slope_length = np.sqrt(geometry['total_height_above_ground']**2 + (geometry['total_height_above_ground'] * geometry['outside_slope'])**2)
        perimeter = 2 * (geometry['footprint_length'] + geometry['footprint_width'])
        side_area = perimeter * side_slope_length
        total_liner_area = base_area + side_area

        boq_data = []
        boq_data.append({"Item": "Site Excavation", "Quantity": excavation_vol, "Unit": "m³"})
        if liner.get('Compacted Clay Liner (m)', 0) > 0:
            clay_vol = total_liner_area * liner['Compacted Clay Liner (m)']
            boq_data.append({"Item": "Compacted Clay Liner", "Quantity": clay_vol, "Unit": "m³"})
        if liner.get('HDPE Geomembrane (mm)', 0) > 0:
            boq_data.append({"Item": "HDPE Geomembrane", "Quantity": total_liner_area, "Unit": "m²"})
        if liner.get('Drainage Layer (m)', 0) > 0:
            drainage_vol = total_liner_area * liner['Drainage Layer (m)']
            boq_data.append({"Item": "Drainage Layer", "Quantity": drainage_vol, "Unit": "m³"})
        boq_df = pd.DataFrame(boq_data)

        st.divider()
        st.subheader("Downloads")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Full Report (.txt)",
                data=report_text,
                file_name=f"{st.session_state.project_data.get('name', 'landfill')}_report.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="📥 Download BOQ (.csv)",
                data=boq_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{st.session_state.project_data.get('name', 'landfill')}_boq.csv",
                mime="text/csv"
            )
'''

# Write the streamlit code to a file with UTF-8 encoding
with open('landfill_app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_code)

print("Streamlit app saved as 'landfill_app.py'")
print("\nTo run the app, execute one of these commands in your terminal:")
print("1. streamlit run landfill_app.py")
print("2. python -m streamlit run landfill_app.py")
