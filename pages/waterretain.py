import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import io

# 1. Configuration and Setup
st.set_page_config(layout="wide", page_title="Liquid Retaining Structures Design")

# 2. Core Calculation Function (MUST BE REPLACED WITH YOUR EXCEL/VBA LOGIC)
def calculate_design_parameters(H, W_wall, t_wall, fck, fyk, soil_density, is_earthquake):
    """
    Function to re-implement the calculations from 'Liquid retaining structures.xlsm'.
    
    ⚠️ WARNING: The calculations below are SIMPLIFIED DUMMY VALUES for illustration.
    You MUST replace the logic here with the exact engineering formulas 
    and checks (e.g., for bending moment, shear, reinforcement, stability) 
    found in your Excel sheets and VBA macros.
    """
    
    # --- DUMMY CALCULATIONS (REPLACE THIS BLOCK) ---
    gamma_w = 9.81  # Water density (kN/m^3)
    max_pressure = gamma_w * H  # Maximum pressure at the base
    
    # Bending Moment (Simplified, assuming a Cantilever Wall)
    # Your Excel might use complex coefficients for tank design (e.g., for 'Sheet1' calculations)
    B_max = (max_pressure * H**2) / 6 
    
    # Effective depth (assuming average concrete cover)
    d = t_wall - 0.05
    B_u = 1.0 # Width of the section (1m strip)
    
    # Check for K_limit and calculate required steel
    if d <= 0:
         # Prevents division by zero if inputs are bad
        A_st_mm2_per_m = 0 
    else:
        # K-factor check (requires converting kN.m to N.mm)
        K = (B_max * 10**6) / (B_u * 1000 * d**2 * fck) 
        # Simplified steel area calculation (assuming under-reinforced section)
        if K < 0.15:
            A_st_m2 = 0.5 * (fck / fyk) * (1 - np.sqrt(1 - 4.79 * K)) * B_u * d
            A_st_mm2_per_m = A_st_m2 * 10**6
        else:
            A_st_mm2_per_m = 9999 # Over-reinforced or requires compression steel
    
    # Stability Check (Simplified Factor of Safety against Sliding)
    resisting_force = W_wall * soil_density * 0.5
    overturning_force = (0.5 * max_pressure * H)
    SF_sliding = resisting_force / overturning_force if overturning_force else 0
    
    # Adjustments for Seismic Load (Replicating potential VBA logic)
    seismic_factor = 1.0
    if is_earthquake:
        seismic_factor = 1.2
        B_max *= seismic_factor
        A_st_mm2_per_m *= seismic_factor
        SF_sliding /= 1.1 # Stability factor reduced under seismic conditions

    results = {
        'Max Pressure (kPa)': max_pressure,
        'Design Bending Moment (kN.m/m)': B_max,
        'Required Steel Area (mm²/m)': A_st_mm2_per_m,
        'SF Against Sliding': SF_sliding,
        'Status': 'PASS' if SF_sliding >= 1.5 else 'FAIL'
    }
    return results

# 3. Utility Function for Download
def to_excel(df):
    """Saves DataFrame to an in-memory Excel file and provides a download button."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Design_Results', index=False)
    
    # Create the download link
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode('utf-8')
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Liquid_Structure_Design.xlsx">📥 Download Detailed Results (Excel)</a>'
    return href

# 4. Streamlit UI Layout
st.title("💧 Design of Liquid Retaining Structures")
st.markdown("This application mimics the input and output flow of **`Liquid retaining structures.xlsm`**. Ensure the core calculation function uses your exact Excel/VBA engineering logic.")

st.sidebar.header("Design Inputs (Replicating Excel Input Cells)")

# --- Geometry and Loads (Likely from Sheet 1) ---
st.sidebar.subheader("Geometric & Load Parameters")
structure_type = st.sidebar.selectbox("Structure Type (Select Box from Excel Controls):", 
                                      ["Rectangular Tank", "Circular Tank", "Cantilever Retaining Wall"])
H_fluid = st.sidebar.number_input("Design Fluid Height, $H$ (m):", min_value=0.1, value=3.5, step=0.1, key='H')
W_wall = st.sidebar.number_input("Wall Thickness, $t$ (m):", min_value=0.1, value=0.4, step=0.01, key='W_wall')
t_wall = st.sidebar.number_input("Base Slab Length (or Wall Width), $L$ (m):", min_value=0.1, value=2.5, step=0.1, key='t_wall')

# --- Material Properties (Likely from Sheet 2) ---
st.sidebar.subheader("Material Properties")
fck = st.sidebar.selectbox("Concrete Characteristic Strength, $f_{ck}$ (N/mm²):", [25, 30, 35, 40], index=1)
fyk = st.sidebar.selectbox("Steel Yield Strength, $f_{yk}$ (N/mm²):", [415, 500], index=1)
soil_density = st.sidebar.number_input("Soil Density, $\\gamma_{soil}$ (kN/m³):", min_value=10.0, value=18.0, step=0.5)

# --- Custom Controls (Replicating Excel Form Controls/VBA Triggers) ---
st.sidebar.subheader("Design Options & Checks")
is_earthquake = st.sidebar.checkbox("Consider Seismic Load (Linked to VBA Logic)?")
st.sidebar.markdown("---")

# Button to run calculation (Replicating a macro-linked button)
calculate_button = st.sidebar.button("▶️ RUN DESIGN CALCULATION (Excel Macro Replicate)")

# --- Main Content Area for Results ---

if calculate_button:
    
    # 3. Run Calculation
    results = calculate_design_parameters(H_fluid, W_wall, t_wall, fck, fyk, soil_density, is_earthquake)

    st.header(f"Results Summary for {structure_type}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Design Pressure", f"{results['Max Pressure (kPa)']:.2f} kPa")
    with col2:
        st.metric("Design Bending Moment", f"{results['Design Bending Moment (kN.m/m)']:.2f} kN.m/m")
    with col3:
        st.metric("Required Steel Area", f"{results['Required Steel Area (mm²/m)']:.0f} mm²/m")
    with col4:
        # Status check replicating an 'OK' or 'FAIL' message in a cell
        if results['Status'] == 'PASS':
            st.success(f"Stability: {results['SF Against Sliding']:.2f} (PASS)")
        else:
            st.error(f"Stability: {results['SF Against Sliding']:.2f} (FAIL)")
    
    st.markdown("---")
    
    ## 📊 Visualization (Replicating Excel Charts/Drawings - Sheet 4)
    st.subheader("Bending Moment and Pressure Diagram")
    
    # Prepare data for plotting
    num_points = 50
    heights = np.linspace(0, H_fluid, num_points)
    # Pressure is linear (max at base, zero at top)
    pressure_values = results['Max Pressure (kPa)'] * (1 - heights / H_fluid)
    # Bending moment is quadratic (or follows a more complex function in the real design)
    moment_values = results['Design Bending Moment (kN.m/m)'] * (heights / H_fluid)**2

    plot_data = pd.DataFrame({
        'Height from Base (m)': heights, 
        'Pressure (kPa)': pressure_values,
        'Moment (kN.m/m)': moment_values
    })
    
    fig = px.line(plot_data, x='Pressure (kPa)', y='Height from Base (m)', title='Hydrostatic Pressure Diagram')
    fig.update_yaxes(autorange="reversed") # Base at bottom
    st.plotly_chart(fig, use_container_width=True)

    ## 📝 Detailed Reinforcement Schedule (Likely from Sheet 3/4)
    st.subheader("Detailed Design Output (Rebar Schedule)")
    
    # Dummy Rebar Schedule mimicking a common output table
    output_df = pd.DataFrame({
        'Location': ['Inner Face (Base)', 'Outer Face (Wall Base)', 'Wall Mid-Height (Horizontal)'],
        'Required As (mm²/m)': [f"{results['Required Steel Area (mm²/m)']:.0f}", 
                                f"{results['Required Steel Area (mm²/m)']*0.8:.0f}", 
                                f"{results['Required Steel Area (mm²/m)']*0.3:.0f}"],
        'Selected Bar/Spacing': ['T20 @ 150mm', 'T16 @ 125mm', 'T10 @ 200mm'],
        'Actual As Provided (mm²/m)': [2094, 1608, 785] # Dummy values
    })
    
    st.dataframe(output_df, hide_index=True)

    # --- Download Functionality (Replicating 'Save Report' Macro) ---
    st.markdown(to_excel(output_df), unsafe_allow_html=True)
    
else:
    st.info("👈 Enter design parameters in the sidebar and click the 'RUN DESIGN CALCULATION' button to generate the results. Remember to update the Python calculation logic!")

st.markdown("---")
st.caption("This app is an emulation of the Excel file structure. The engineering results are **not guaranteed to be accurate** until the custom Python function (`calculate_design_parameters`) is precisely updated with the formulas from the original VBA/Excel calculations.")
