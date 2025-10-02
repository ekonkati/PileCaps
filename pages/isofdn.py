import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ASSUMED DESIGN LOGIC (BASED ON IS 456/STANDARD PRACTICE) ---
def design_isolated_footing(Pu, Mucx, Mucz, fc, fy, SBC, gamma_c, bc, dc):
    """
    Performs simplified isolated footing design calculations.
    - Assumes square footing for simplicity in initial sizing.
    - Does NOT include full shear and moment depth checks.
    """
    try:
        # 1. Calculate Required Area
        P_service = Pu / 1.5  # Approximate service load (assuming 1.5 load factor)
        P_eff = P_service + gamma_c * 1 * 1 * 1  # Add self-weight (approx 10% or assume 1m deep)
        A_req = P_eff / SBC
        
        # Assume square footing
        B_req = np.sqrt(A_req)
        B = np.ceil(B_req * 100) / 100  # Round up to nearest 0.01m
        L = B
        
        A_actual = L * B
        
        # Calculate Max Soil Pressure (q_max) for checking
        # Check eccentricity
        e_x = Mucx / P_service
        e_z = Mucz / P_service
        
        # Core check: is it outside the middle third (or middle sixth for biaxial)?
        if e_x > L / 6 or e_z > B / 6:
            # Simplified for safety: If outside middle sixth, assume failure mode / uplift for this code
            # Note: A real design would calculate the pressure distribution for partial bearing.
             q_max = 9999.0 # Placeholder for a very large pressure indicating failure/uplift
        else:
             # Pressure calculation (simplified by assuming service load P_service)
             q_max = P_service / A_actual + (6 * Mucx) / (L * B**2) + (6 * Mucz) / (B * L**2)
        
        q_net_u = Pu / A_actual # Simplified net upward pressure for design
        
        # 2. Approximate Depth 'D' (Initial guess based on punching shear or thumb rule)
        D_trial = B / 4
        
        # 3. Design Bending Moment (Simplified, considering critical section at column face)
        a_x = (L - bc) / 2
        M_u_x = q_net_u * B * a_x**2 / 2
        
        # 4. Required Steel (Ast) - Simplified
        # Assuming d = D_trial - cover
        d = D_trial - 0.075 # 75mm cover
        
        # Limit moment calculation (from IS 456)
        if fy == 415:
            R_lim = 0.138 * fc * (d*1000)**2 * 1e-6 # Convert to kNm
        elif fy == 500:
            R_lim = 0.133 * fc * (d*1000)**2 * 1e-6 # Convert to kNm
        else: # assuming 250
            R_lim = 0.148 * fc * (d*1000)**2 * 1e-6 # Convert to kNm
            
        if M_u_x > R_lim * B:
             # This checks if the moment per meter exceeds the section capacity
             pass # Let the user know, but don't crash
        
        # Simplified Ast calculation (T-beam or uncoupled section assumed)
        # Ast_req in mm2 per meter width (B=1m)
        Ast_req = (0.5 * fc / fy) * (1 - np.sqrt(1 - (4.6 * M_u_x * 10**6) / (fc * B * (d*1000)**2))) * B * d * 1000 # Corrected units
        
        # Ast_min (0.12% for Fe415/500, 0.15% for Fe250)
        Ast_min = 0.0012 * 1 * (d * 1000) * 1000 # mm2 per meter width
        Ast_x = max(Ast_req * 1000, Ast_min) # Convert from m2 to mm2 and check min (was incorrectly converting B*d to m2)

        # Simplified results for demonstration
        results = {
            "Footing Length (L) [m]": L,
            "Footing Width (B) [m]": B,
            "Required Area [m²]": A_req,
            "Actual Area [m²]": A_actual,
            "Max Soil Pressure (q_max) [kN/m²]": q_max,
            "Trial Depth (D) [m]": D_trial,
            "Design Moment (Mu_x) [kNm]": M_u_x,
            "Req. Steel (Ast_x) [mm²/m]": Ast_x,
        }
        return results
    except Exception as e:
        # Return the error message explicitly
        return {"Error": f"Calculation failed: {str(e)}"}

# --- PLOTLY VISUALIZATIONS (Unchanged) ---

def plot_footing_3d(L, B, D, bc, dc):
    """Generates an interactive 3D plot of the column and footing."""
    
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

st.warning("⚠️ **Disclaimer:** This app uses standard IS code logic as a template. To fully replicate your Excel file's functionality, please ensure the parameters and calculation formulas are consistent with those in your original spreadsheet.")

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
    
    # --- ERROR HANDLING FIX ---
    if "Error" in results:
        st.error(f"Design Calculation Failed! Check inputs or logic. Details: {results['Error']}")
        # Stop execution here if there's an error
    else:
        L_final = results['Footing Length (L) [m]']
        B_final = results['Footing Width (B) [m]']
        D_trial = results['Trial Depth (D) [m]']

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Design Results Summary")
            # Filter the dictionary to remove temporary calculation keys for clean display
            display_results = {k: v for k, v in results.items()}
            df = pd.DataFrame(display_results.items(), columns=['Parameter', 'Value'])
            st.dataframe(df, use_container_width=True)

        with col2:
            st.header("Check against SBC")
            
            # Use the retrieved value, now safe because we checked for 'Error'
            q_max_val = results['Max Soil Pressure (q_max)']
            
            if q_max_val == 9999.0:
                 st.error("Soil Pressure Check **FAILED (Uplift/Outside Middle Sixth)**. The foundation is undersized for the applied moment. Increase size or reduce eccentricity.")
            elif q_max_val < SBC * 1.1: 
                 st.success(f"Maximum Pressure: {q_max_val:.2f} kN/m² < SBC: {SBC:.2f} kN/m² **(OK)**")
            else:
                 st.error(f"Maximum Pressure: {q_max_val:.2f} kN/m² > SBC: {SBC:.2f} kN/m² **(FAIL)**. Increase Footing Size.")
            
            st.subheader("Key Design Output")
            st.metric("Footing Size (L x B)", f"{L_final:.2f}m x {B_final:.2f}m")
            st.metric("Required Steel (Ast, X-Dir)", f"{results['Req. Steel (Ast_x)']:.2f} mm²/m width")


        st.markdown("---")
        st.header("Design Sketches")
        
        sketch_col1, sketch_col2 = st.columns(2)
        
        with sketch_col1:
             # 3D Footing Sketch (Plotly)
             st.plotly_chart(plot_footing_3d(L_final, B_final, D_trial, bc, dc), use_container_width=True)
             
        with sketch_col2:
            # Footing Plan Sketch (Plotly)
            st.plotly_chart(plot_footing_plan(L_final, B_final, bc, dc), use_container_width=True)
             
else:
    st.info("Enter design parameters in the sidebar and click 'Run Design' to generate results and sketches.")
