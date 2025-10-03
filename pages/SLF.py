import streamlit as st
import pandas as pd
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="SLF Design Calculator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions for Calculations ---

def calculate_bottom_dimensions(top_length, top_width, depth, side_slope_ratio):
    """Calculates the bottom dimensions of the landfill cell."""
    if side_slope_ratio is None or side_slope_ratio == 0:
        return top_length, top_width
    
    # The reduction on each side is depth * horizontal_ratio
    reduction = 2 * depth * side_slope_ratio
    bottom_length = top_length - reduction
    bottom_width = top_width - reduction
    
    # Dimensions cannot be negative
    return max(0, bottom_length), max(0, bottom_width)

def calculate_areas(top_length, top_width, bottom_length, bottom_width, depth, side_slope_ratio):
    """Calculates the various areas of the landfill cell."""
    top_area = top_length * top_width
    bottom_area = bottom_length * bottom_width
    
    if side_slope_ratio is None or side_slope_ratio == 0:
        side_area_long = 0
        side_area_short = 0
    else:
        # Slant height = sqrt(vertical^2 + horizontal^2) = sqrt(depth^2 + (depth*ratio)^2)
        slant_height = math.sqrt(depth**2 + (depth * side_slope_ratio)**2)
        
        # Area of trapezoidal sides
        side_area_long = ((top_length + bottom_length) / 2) * slant_height
        side_area_short = ((top_width + bottom_width) / 2) * slant_height

    total_side_area = 2 * side_area_long + 2 * side_area_short
    total_liner_area = bottom_area + total_side_area
    
    return {
        "Top Area (sq.m)": top_area,
        "Bottom Area (sq.m)": bottom_area,
        "Side Area (Long Sides, each) (sq.m)": side_area_long,
        "Side Area (Short Sides, each) (sq.m)": side_area_short,
        "Total Side Area (sq.m)": total_side_area,
        "Total Liner Area Required (sq.m)": total_liner_area
    }

def calculate_volume(top_area, bottom_area, depth):
    """Calculates the volume of the landfill cell using the prismoidal formula approximation."""
    # Using average end area method for simplicity, as in many such sheets
    average_area = (top_area + bottom_area) / 2
    volume = average_area * depth
    return volume

# --- Streamlit UI ---

# --- Header ---
st.title("🏗️ SLF Area & Volume Calculator")
st.markdown("""
This application replicates the calculations from an SLF (Solid Landfill Facility) design spreadsheet. 
Use the sidebar to input the dimensions of a landfill cell, and the main panel will display the calculated results for areas, liner requirements, and volume.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Enter the dimensions of the landfill cell below.")

# Input widgets
top_length = st.sidebar.number_input("Top Length (m)", min_value=1.0, value=150.0, step=1.0)
top_width = st.sidebar.number_input("Top Width (m)", min_value=1.0, value=100.0, step=1.0)
depth = st.sidebar.number_input("Depth (m)", min_value=1.0, value=10.0, step=0.5)

# Side slope input
# A dictionary to map user-friendly strings to ratio values
slope_options = {
    "1:1 (45°)": 1.0,
    "2:1 (26.6°)": 2.0,
    "3:1 (18.4°)": 3.0,
    "4:1 (14.0°)": 4.0,
    "Vertical (Not Recommended)": None
}
selected_slope = st.sidebar.selectbox(
    "Side Slope (Horizontal:Vertical)",
    options=list(slope_options.keys()),
    index=2  # Default to 3:1
)
side_slope_ratio = slope_options[selected_slope]

# --- Main Panel for Displaying Results ---

# Perform calculations based on inputs
bottom_length, bottom_width = calculate_bottom_dimensions(top_length, top_width, depth, side_slope_ratio)
areas = calculate_areas(top_length, top_width, bottom_length, bottom_width, depth, side_slope_ratio)
volume = calculate_volume(areas["Top Area (sq.m)"], areas["Bottom Area (sq.m)"], depth)

st.markdown("---")

# --- Display Summary Metrics ---
st.header("📊 Key Metrics Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Volume (cubic meters)", value=f"{volume:,.2f}")
with col2:
    st.metric(label="Total Liner Area Required (sq. m)", value=f"{areas['Total Liner Area Required (sq.m)']:,.2f}")
with col3:
    st.metric(label="Top Footprint Area (sq. m)", value=f"{areas['Top Area (sq.m)']:,.2f}")

# --- Display Detailed Calculations in Tabs ---
st.header("🧮 Detailed Calculations")
tab1, tab2 = st.tabs(["Dimension & Area Calculations", "Volume Calculation Details"])

with tab1:
    st.subheader("Calculated Dimensions")
    dim_data = {
        "Dimension": ["Top Length", "Top Width", "Bottom Length", "Bottom Width"],
        "Value (m)": [top_length, top_width, f"{bottom_length:.2f}", f"{bottom_width:.2f}"]
    }
    st.table(pd.DataFrame(dim_data))

    st.subheader("Area Break-down")
    area_df = pd.DataFrame(list(areas.items()), columns=['Area Component', 'Value (sq.m)'])
    area_df['Value (sq.m)'] = area_df['Value (sq.m)'].map('{:,.2f}'.format)
    st.table(area_df)

with tab2:
    st.subheader("Volume Calculation (Average End Area Method)")
    st.markdown(f"""
    The volume is calculated by averaging the top and bottom areas and multiplying by the depth. This is a standard method for prismatic or trapezoidal excavations.
    
    - **Top Area:** `{areas["Top Area (sq.m)"]:,.2f} m²`
    - **Bottom Area:** `{areas["Bottom Area (sq.m)"]:,.2f} m²`
    - **Average Area:** `({areas["Top Area (sq.m)"]:,.2f} + {areas["Bottom Area (sq.m)"]:,.2f}) / 2 = {((areas["Top Area (sq.m)"] + areas["Bottom Area (sq.m)"]) / 2):,.2f} m²`
    - **Depth:** `{depth:,.2f} m`
    
    #### **Volume = Average Area × Depth**
    ### **Volume = {((areas["Top Area (sq.m)"] + areas["Bottom Area (sq.m)"]) / 2):,.2f} m² × {depth:,.2f} m = {volume:,.2f} m³**
    """)

# --- Footer ---
st.markdown("---")
st.info("This calculator is for estimation purposes. All designs should be verified by a qualified engineer.")
