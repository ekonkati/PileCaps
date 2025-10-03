import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ASSUMED DESIGN LOGIC (BASED ON IS 456/STANDARD PRACTICE) ---
def design_isolated_footing(Pu, Mucx, Mucz, fc, fy, SBC, gamma_c, bc, dc):
    """
    Performs simplified isolated footing design calculations.
    - Assumes square footing for simplicity in initial sizing.
    """
    try:
        # 1. Calculate Required Area
        P_service = Pu / 1.5  # Approximate service load (assuming 1.5 load factor)
        # Assume 0.5m depth for initial self-weight estimation
        D_assumed = 0.5 
        W_self = gamma_c * D_assumed # Self weight per m² (assuming 0.5m depth)
        
        # Estimate total area required
        A_req = P_service / (SBC - W_self)
        if A_req <= 0: A_req = (P_service / SBC) * 1.2 # Fallback
        
        # Assume square footing
        B_req = np.sqrt(A_req)
        B = np.ceil(B_req * 100) / 100  # Round up to nearest 0.01m
        L = B
        
        A_actual = L * B
        
        # 2. Check Soil Pressure and Eccentricity
        e_x = abs(Mucx / P_service)
        e_z = abs(Mucz / P_service)
        
        if e_x > L / 6 or e_z > B / 6:
             # Case of uplift / outside middle third (or middle sixth for biaxial)
             q_max = 9999.0 # Sentinel value for failure
             pressure_status = "FAILED (Uplift/High Eccentricity)"
        else:
             # Pressure calculation (P_service includes self weight calculation refinement)
             W_footing = gamma_c * L * B * D_assumed # Rough self weight estimate
             P_total_service = P_service + W_footing
             
             q_max = P_total_service / A_actual + (6 * Mucx) / (L * B**2) + (6 * Mucz) / (B * L**2)
             pressure_status = "OK"
        
        # Factored net upward pressure for design
        q_net_u = Pu / A_actual
        
        # 3. Approximate Depth 'D' (Initial guess based on B/3.5)
        D_trial = B / 3.5
        d = D_trial - 0.075 # effective depth (m)
        
        # 4. Design Bending Moment (Critical section at column face)
        a_x = (L - bc) / 2
        M_u_x = q_net_u * B * a_x**2 / 2 # kNm
        
        # 5. Required Steel (Ast) per meter width (B=1m)
        b_m = 1000 # 1 meter width in mm
        d_m = d * 1000 # effective depth in mm
        
        # The term inside the square root (R_term)
        R_term = (4.6 * M_u_x * 10**6) / (fc * b_m * d_m**2)

        if R_term >= 1.0 or R_term < 0:
             # Moment capacity exceeded, depth is too shallow
             Ast_x = 99999.0 # Sentinel value for insufficient depth
             steel_status = "FAILED (Depth Insufficient)"
        else:
             # Calculate Ast (mm2/m)
             # Equation for Ast from IS 456, solving quadratic equation
             Ast_req_m2 = (0.5 * fc / fy) * (1 - np.sqrt(1 - R_term)) * b_m * d_m
             
             # Ast_min (0.12% for Fe415/500)
             Ast_min_m2 = 0.0012 * b_m * d_m
             Ast_x = max(Ast_req_m2, Ast_min_m2)
             steel_status = "OK"

        # RESULTS DICTIONARY (KEYS STANDARDIZED)
        results = {
            "Footing Length (L) [m]": L,
            "Footing Width (B) [m]": B,
            "Required Area [m²]": A_req,
            "Actual Area [m²]": A_actual,
            "Max Soil Pressure (q_max)": q_max, # KEY FIXED
            "Pressure Status": pressure_status,
            "Trial Depth (D) [m]": D_trial,
            "Design Moment (Mu_x) [kNm]": M_u_x,
            "Req. Steel (Ast_x)": Ast_x, # KEY FIXED
            "Steel Status": steel_status
        }
        return results
        
    except Exception as e:
        # Return the error message explicitly
        return {"Error": f"Calculation failed during general design step. Detail: {str(e)}"}

# --- PLOTLY VISUALIZATIONS (Unchanged) ---

def plot_footing_3d(L, B, D, bc, dc):
    """Generates an interactive 3D plot of the column and footing."""
    # ... (Plotly code remains the same)
    # 1. Footing (Base)
    footing = go.Mesh3d(
        x=[0, L, L, 0, 0, L, L, 0],
        y=[0, 0, B, B, 0, 0, B, B],
        z=[-D, -D, -D, -D, 0, 0, 0, 0],
        colorbar_title='Footing',
        colorscale='Viridis',
        opacity=0.6,
        name='Footing',
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        flatshading=True,
        color='lightblue'
    )
    
    # 2. Column (Simplified as a block)
    col_x_start = L/2 - bc/2
    col_x_end = L/2 + bc/2
    col_y_start = B/2 - dc/2
    col_y_end = B/2 + dc/2
    col_height = 2*D # Extend column above footing
    
    col_x = [col_x_start, col_x_end, col_x_end, col_x_start, col_x_start, col_x_end, col_x_end, col_x_start]
    col_y = [col_y_start, col_y_start, col_y_end, col_y_end, col_y_start, col_y_start, col_y_end, col_y_end]
    col_z = [0, 0, 0, 0, col_height, col_height, col_height, col_height]
    
    column = go.Mesh3d(
        x=col_x,
        y=col_y,
        z=col_z,
        opacity=0.8,
        name='Column',
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        flatshading=True,
        color='gray'
    )
    
    layout = go.Layout(
        title='3D Isolated Footing Sketch (Plotly)',
        scene=dict(
            xaxis=dict(title='Length (m)'),
            yaxis=dict(title='Width (m)'),
            zaxis=dict(title='Depth (m)'),
            aspectmode='data'
        )
    )
    
    fig = go.Figure(data=[footing, column], layout=layout)
    return fig

def plot_footing_plan(L, B, bc, dc):
    """Generates a 2D plan view showing reinforcement areas."""
    
    fig = go.Figure()
    
    # 1. Footing Outline
    fig.add_trace(go.Scatter(
        x=[0, L, L, 0, 0], y=[0, 0, B, B, 0],
        mode='lines', name='Footing Edge', line=dict(color='black', width=3)
    ))
    
    # 2. Column
    col_x_start = L/2 - bc/2
    col_x_end = L/2 + bc/2
    col_y_start = B/2 - dc/2
    col_y_end = B/2 + dc/2
    fig.add_trace(go.Scatter(
        x=[col_x_start, col_x_end, col_x_end, col_x_start, col_x_start],
        y=[col_y_start, col_y_start, col_y_end, col_y_end, col_y_start],
        fill="toself", fillcolor='grey', mode='lines', name='Column'
    ))
    
    # 3. Reinforcement Area (approximate critical section area)
    # Area for Ast_x (L-direction)
    fig.add_shape(type="rect",
        x0=col_x_start, y0=0, x1=col_x_end, y1=B,
        line=dict(color="red", width=1, dash="dot"),
        name="Critical Bending Strip (L-Dir)"
    )
    
    # 4. Center Lines
    fig.add_vline(x=L/2, line_width=1, line_dash="dash", line_color="blue")
    fig.add_hline(y=B/2, line_width=1, line_dash="dash", line_color="blue")

    fig.update_layout(
        title='Footing Plan and Critical Sections',
        xaxis_title='L (m)',
        yaxis_title='B (m)',
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1
    )
    return fig

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")

st.title("🏗️ Isolated Foundation Design (IS Code Based)")
st.caption("Replicating Functionality from 'ISLOATED FOUNDATION DESIGN FOR ALL NODES.xlsm'")

st.warning("⚠️ **Disclaimer:** This app uses standard IS code logic as a template. To fully replicate your Excel file's functionality, you must ensure the parameters and calculation formulas are consistent with your original spreadsheet.")

st.sidebar.header("Design Inputs")

# Material Properties
st.sidebar.subheader("Material & Soil Properties")
fc = st.sidebar.selectbox("Concrete Grade (fck) [N/mm²]", options=[20, 25, 30, 35, 40], index=1)
fy = st.sidebar.selectbox("Steel Grade (fy) [N/mm²]", options=[415, 500], index=0)
SBC = st.sidebar.number_input("Safe Bearing Capacity (SBC) [kN/m²]", value=200.0, step=10.0)
gamma_c = st.sidebar.number_input("Unit Weight of Concrete [kN/m³]", value=25.0, step=1.0)

# Load Data
st.sidebar.subheader("Column Load Data (Factored)")
Pu = st.sidebar.number_input("Axial Load (Pu) [kN]", value=800.0, step=50.0)
Mucx = st.sidebar.number_input("Moment about X-axis (Mucx) [kNm]", value=20.0, step=5.0)
Mucz = st.sidebar.number_input("Moment about Z-axis (Mucz) [kNm]", value=15.0, step=5.0)

# Column Dimensions
st.sidebar.subheader("Column Dimensions")
bc = st.sidebar.number_input("Column Width (bc) [m]", value=0.40, step=0.05)
dc = st.sidebar.number_input("Column Depth (dc) [m]", value=0.40, step=0.05)


if st.sidebar.button("Run Design"):
    
    # Perform Design
    results = design_isolated_footing(Pu, Mucx, Mucz, fc, fy, SBC, gamma_c, bc, dc)
    
    # --- ERROR HANDLING BLOCK ---
    if "Error" in results:
        st.error(f"Design Calculation Failed! Check inputs or logic. Details: {results['Error']}")
        # The app stops displaying results here if there's a Python error
    else:
        # If no "Error" key, we proceed safely
        L_final = results['Footing Length (L) [m]']
        B_final = results['Footing Width (B) [m]']
        D_trial = results['Trial Depth (D) [m]']

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Design Results Summary")
            # Create a dictionary for clean display with units added back
            display_data = {
                "Footing Size (L x B)": f"{L_final:.2f}m x {B_final:.2f}m",
                "Required Area": f"{results['Required Area [m²]']:.2f} m²",
                "Actual Area": f"{results['Actual Area [m²]']:.2f} m²",
                "Trial Depth (D)": f"{D_trial:.3f} m",
                "Design Moment (Mu_x)": f"{results['Design Moment (Mu_x) [kNm]']:.2f} kNm",
                "Max Soil Pressure": f"{results['Max Soil Pressure (q_max)']:.2f} kN/m²",
                "Required Steel (Ast, X)": f"{results['Req. Steel (Ast_x)']:.2f} mm²/m"
            }
            df = pd.DataFrame(display_data.items(), columns=['Parameter', 'Value'])
            st.dataframe(df, use_container_width=True)

        with col2:
            st.header("Design Checks")
            
            # 1. Soil Pressure Check
            q_max_val = results['Max Soil Pressure (q_max)']
            pressure_status = results['Pressure Status']
            st.subheader("1. Safe Bearing Capacity (SBC) Check")
            
            if pressure_status.startswith("FAILED"):
                 st.error(f"Pressure Check: **{pressure_status}**. The foundation is undersized for the applied moment.")
            elif q_max_val < SBC * 1.1: 
                 st.success(f"Maximum Pressure: {q_max_val:.2f} kN/m² < SBC: {SBC:.2f} kN/m² **(OK)**")
            else:
                 st.error(f"Maximum Pressure: {q_max_val:.2f} kN/m² > SBC: {SBC:.2f} kN/m² **(FAIL)**. Increase Footing Size.")
                 
            # 2. Steel/Depth Check
            st.subheader("2. Depth and Steel Check")
            steel_status = results['Steel Status']
            req_steel = results['Req. Steel (Ast_x)']
            
            if steel_status.startswith("FAILED"):
                st.error(f"Steel Design: **{steel_status}**. The Trial Depth ({D_trial:.2f}m) is too shallow for the bending moment. **Increase depth (D)**.")
                req_steel_text = "N/A (Depth Fail)"
            else:
                st.success("Bending Depth and Steel calculation **(OK)**.")
                req_steel_text = f"{req_steel:.2f} mm²/m width"


            st.subheader("Key Output")
            st.metric("Footing Size (L x B)", f"{L_final:.2f}m x {B_final:.2f}m")
            st.metric("Required Steel (Ast, X-Dir)", req_steel_text)


        st.markdown("---")
        st.header("Design Sketches")
        
        # Display sketches only if design size is reasonable (not failed with 9999)
        if L_final < 100 and B_final < 100:
            sketch_col1, sketch_col2 = st.columns(2)
            
            with sketch_col1:
                 # 3D Footing Sketch (Plotly)
                 st.plotly_chart(plot_footing_3d(L_final, B_final, D_trial, bc, dc), use_container_width=True)
                 
            with sketch_col2:
                # Footing Plan Sketch (Plotly)
                st.plotly_chart(plot_footing_plan(L_final, B_final, bc, dc), use_container_width=True)
        else:
             st.error("Footing size is unrealistically large. Check inputs or design logic.")
             
else:
    st.info("Enter design parameters in the sidebar and click 'Run Design' to generate results and sketches.")
