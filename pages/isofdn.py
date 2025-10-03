import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- IS 456:2000 DESIGN CONSTANTS ---

# Simplified lookup for Tau_c_max (Table 20, IS 456:2000)
TAU_C_MAX = {
    20: 2.8, 25: 3.1, 30: 3.5, 35: 3.7, 40: 4.0
}

# Simplified lookup for percentage steel pt (Table 19, IS 456:2000)
def get_tau_c(fck, pt):
    pt = max(0.15, min(pt, 2.0))
    if fck == 20:
        if pt < 0.25: return 0.36
        if pt < 0.50: return 0.48
        if pt < 0.75: return 0.56
        if pt < 1.00: return 0.62
        return 0.8
    elif fck == 30:
        if pt < 0.25: return 0.41
        if pt < 0.50: return 0.54
        if pt < 0.75: return 0.63
        if pt < 1.00: return 0.71
        return 0.95
    else: # Default for M25
        if pt < 0.25: return 0.37
        if pt < 0.50: return 0.50
        if pt < 0.75: return 0.57
        if pt < 1.00: return 0.64
        return 0.82

# --- CORE ENGINEERING FUNCTIONS ---

def calculate_provided_ast(width_m, phi_mm, spacing_mm):
    """Calculates total provided steel area (m²) over the given width."""
    if spacing_mm == 0: return 0.0
    
    As_bar = np.pi * (phi_mm / 1000)**2 / 4 # Area of one bar (m²)
    # Number of bars * Area of one bar (provided over the total width)
    Ast_prov_total = (width_m / (spacing_mm / 1000)) * As_bar
    Ast_prov_per_meter = Ast_prov_total / width_m # Ast per meter (m²/m)
    
    return Ast_prov_total, Ast_prov_per_meter * 1000000 # returns total m² and mm²/m

def calculate_ast_required(Mu_kNm, B_m, d_m, fc, fy):
    """Calculates required steel area (Ast) in mm² per meter width."""
    Mu_Nmm = Mu_kNm * 10**6
    B_mm = B_m * 1000
    d_mm = d_m * 1000
    
    if fy == 415:
        Mu_lim = 0.138 * fc * B_mm * d_mm**2
    else: # Fe 500
        Mu_lim = 0.133 * fc * B_mm * d_mm**2
        
    if Mu_Nmm > Mu_lim:
        return 99999.0, "FAIL_DEPTH"

    R_term = (4.6 * Mu_Nmm) / (fc * B_mm * d_mm**2)
    
    if R_term >= 1.0 or R_term < 0:
        return 99999.0, "FAIL_MATH" 

    Ast_req_m2 = (0.5 * fc / fy) * (1 - np.sqrt(1 - R_term)) * B_mm * d_mm
    
    # Ast_min (0.12% of gross area B*D is typical, but 0.12% of B*d is simpler for Ast/m)
    Ast_min_m2 = 0.0012 * B_mm * d_mm
    
    Ast_final = max(Ast_req_m2, Ast_min_m2)
    return Ast_final, "OK"

def one_way_shear_check(Pu, q_net_u, L, B, bc, d, fc, Ast_prov_total_m2):
    """Performs one-way shear check (critical section at 'd' from column face)."""
    
    a_crit = (L - bc) / 2 - d
    
    if a_crit <= 0:
        return 0, 0, 0, "OK_DEEP"

    Vu = q_net_u * B * a_crit
    tau_v = Vu * 1000 / (B * 1000 * d * 1000)
    
    # Use total provided steel for the shear resistance calculation (pt ratio)
    pt_prov = (Ast_prov_total_m2 / (B * d)) * 100 
    tau_c = get_tau_c(fc, pt_prov)
    
    result = "OK" if tau_v < tau_c else "FAIL"
    
    return Vu, tau_v, tau_c, result

def punching_shear_check(Pu, q_net_u, L, B, bc, dc, d, fc):
    """Performs punching shear check (critical section at 'd/2' from column face)."""
    
    b_1 = bc + d
    d_1 = dc + d
    bo = 2 * (b_1 + d_1)
    
    A_crit = b_1 * d_1
    
    Vu_p = Pu - q_net_u * A_crit
    
    tau_vp = Vu_p * 1000 / (bo * d * 1000)
    
    beta_c = min(bc, dc) / max(bc, dc)
    ks = min(0.5 + beta_c, 1.0)
    tau_c_prime = 0.25 * np.sqrt(fc)
    
    tau_c_p = ks * tau_c_prime
    
    tau_c_max = TAU_C_MAX.get(fc, 3.1)
    
    result = "OK"
    if tau_vp > tau_c_max:
        result = "FAIL_MAX"
    elif tau_vp > tau_c_p:
        result = "FAIL"
    
    return Vu_p, tau_vp, tau_c_p, tau_c_max, result

def design_footing_checks(L, B, D, bc, dc, fc, fy, SBC, gamma_c, Pu, P_service, Mucx, Mucz, 
                          phi_bx, s_bx, phi_bz, s_bz, phi_tx, s_tx, phi_tz, s_tz):

    # 1. Effective Depths (assuming X-bars are bottom layer)
    d_x = D - 0.075 # For X-direction (Moment Mu_x, Width B)
    d_z = d_x - (phi_bx / 1000) # For Z-direction (Moment Mu_z, Width L)
    
    if d_x <= 0 or d_z <= 0:
        return {"FINAL_STATUS": "FAIL_GEOMETRY", "Error": "Effective depth is zero or negative. Increase D."}

    # 2. Base Pressure Check
    A_actual = L * B
    W_footing = gamma_c * L * B * D
    P_total_service = P_service + W_footing
    
    e_x = abs(Mucx / P_service)
    e_z = abs(Mucz / P_service)
    
    if e_x > L / 6 or e_z > B / 6:
        q_max, q_min, pressure_status = 9999.0, 0.0, "FAIL (Uplift/High Eccentricity)"
    else:
        q_max = P_total_service / A_actual + (6 * Mucx) / (B * L**2) + (6 * Mucz) / (L * B**2)
        q_min = P_total_service / A_actual - (6 * Mucx) / (B * L**2) - (6 * Mucz) / (L * B**2)
        
        pressure_status = "OK" if q_max < SBC else "FAIL"
        if q_min < 0: pressure_status = "FAIL (Tension/Uplift)"
        
    q_net_u = Pu / A_actual 

    # 3. Bending Check (X-Direction)
    a_x = (L - bc) / 2
    M_u_x = q_net_u * B * a_x**2 / 2 
    Ast_req_x, moment_status_x = calculate_ast_required(M_u_x, B, d_x, fc, fy)
    Ast_prov_x_total, Ast_prov_x_perm = calculate_provided_ast(B, phi_bx, s_bx)
    
    # 4. Bending Check (Z-Direction)
    a_z = (B - dc) / 2
    M_u_z = q_net_u * L * a_z**2 / 2 
    Ast_req_z, moment_status_z = calculate_ast_required(M_u_z, L, d_z, fc, fy)
    Ast_prov_z_total, Ast_prov_z_perm = calculate_provided_ast(L, phi_bz, s_bz)

    # 5. Shear Checks (Use the lesser effective depth d_z for conservatism)
    d_shear = d_z
    
    # Use Ast_prov_x_total for one-way shear check
    Vu_1w, tau_v_1w, tau_c_1w, shear_1w_status = one_way_shear_check(Pu, q_net_u, L, B, bc, d_shear, fc, Ast_prov_x_total)
    
    Vu_p, tau_v_p, tau_c_p, tau_c_max, shear_p_status = punching_shear_check(Pu, q_net_u, L, B, bc, dc, d_shear, fc)
    
    # 6. Top Steel (Check against minimum steel requirement, 0.12% of gross area B*D)
    Ast_min_temp = 0.0012 * B * D * 1000000 # mm² (total area)
    
    Ast_prov_tx_total, _ = calculate_provided_ast(B, phi_tx, s_tx)
    Ast_prov_tz_total, _ = calculate_provided_ast(L, phi_tz, s_tz)
    
    # 7. Final Status Compilation
    final_status = "OK"
    if pressure_status.startswith("FAIL"): final_status = "FAIL"
    if moment_status_x.startswith("FAIL") or moment_status_z.startswith("FAIL"): final_status = "FAIL_DEPTH"
    if shear_1w_status.startswith("FAIL") or shear_p_status.startswith("FAIL"): final_status = "FAIL_SHEAR"
    if Ast_prov_x_total < Ast_req_x * (B/1000) or Ast_prov_z_total < Ast_req_z * (L/1000): final_status = "FAIL_REBAR"
    if (Ast_prov_tx_total + Ast_prov_tz_total) < Ast_min_temp: final_status = "FAIL_TOP_MIN_REBAR"


    # RESULTS DICTIONARY
    results = {
        "Footing Length (L) [m]": L, "Footing Width (B) [m]": B, "Trial Depth (D) [m]": D, 
        "d_x [m]": d_x, "d_z [m]": d_z, "Max Soil Pressure [kN/m²]": q_max, 
        "Min Soil Pressure [kN/m²]": q_min, "Pressure Status": pressure_status,
        
        # X-Direction Bending
        "Mu_x [kNm]": M_u_x, "Ast_req_x [mm²/m]": Ast_req_x, "Ast_prov_x_total [m²]": Ast_prov_x_total,
        "Ast_prov_x_perm [mm²/m]": Ast_prov_x_perm, "Moment Status X": moment_status_x,
        
        # Z-Direction Bending
        "Mu_z [kNm]": M_u_z, "Ast_req_z [mm²/m]": Ast_req_z, "Ast_prov_z_total [m²]": Ast_prov_z_total,
        "Ast_prov_z_perm [mm²/m]": Ast_prov_z_perm, "Moment Status Z": moment_status_z,
        
        # Shear Checks
        "1W Shear Force (Vu) [kN]": Vu_1w, "1W Shear Stress (τv) [N/mm²]": tau_v_1w, 
        "1W Permissible (τc) [N/mm²]": tau_c_1w, "1W Shear Status": shear_1w_status,
        "Punching Shear Stress (τvp) [N/mm²]": tau_v_p, "Punching Permissible (τcp) [N/mm²]": tau_c_p, 
        "Punching Shear Status": shear_p_status,
        
        # Top Steel
        "Ast_min_temp [mm²]": Ast_min_temp, 
        "Ast_prov_top_total [mm²]": (Ast_prov_tx_total + Ast_prov_tz_total) * 1000000,
        
        "FINAL_STATUS": final_status
    }
    return results

# --- PLOTLY VISUALIZATIONS ---

def plot_footing_3d(L, B, D, bc, dc):
    """Generates an interactive 3D plot of the column and footing."""
    # 1. Footing (Base)
    footing = go.Mesh3d(
        x=[0, L, L, 0, 0, L, L, 0],
        y=[0, 0, B, B, 0, 0, B, B],
        z=[-D, -D, -D, -D, 0, 0, 0, 0], 
        opacity=0.6,
        color='lightblue',
        name='Footing',
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
    )
    
    # 2. Column (Pedestal)
    col_x_start = L/2 - bc/2
    col_x_end = L/2 + bc/2
    col_y_start = B/2 - dc/2
    col_y_end = B/2 + dc/2
    col_height = 1.5*D
    
    col_x = [col_x_start, col_x_end, col_x_end, col_x_start, col_x_start, col_x_end, col_x_end, col_x_start]
    col_y = [col_y_start, col_y_start, col_y_end, col_y_end, col_y_start, col_y_start, col_y_end, col_y_end]
    col_z = [0, 0, 0, 0, col_height, col_height, col_height, col_height]
    
    column = go.Mesh3d(
        x=col_x,
        y=col_y,
        z=col_z,
        opacity=0.8,
        color='gray',
        name='Pedestal',
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Length (m)'),
            yaxis=dict(title='Width (m)'),
            zaxis=dict(title='Depth (m)'),
            aspectmode='data'
        )
    )
    
    fig = go.Figure(data=[footing, column], layout=layout)
    return fig

def plot_base_pressure_diagram(L, B, P_total_service, Mucx, Mucz, SBC):
    """Creates a 3D surface plot of the soil pressure distribution."""
    x = np.linspace(0, L, 50)
    y = np.linspace(0, B, 50)
    X, Y = np.meshgrid(x, y)
    
    q = (P_total_service / (L * B)) + \
        (Mucx * (X - L/2)) / (L * B**3 / 12) + \
        (Mucz * (Y - B/2)) / (B * L**3 / 12)
        
    q[q < 0] = 0

    fig = go.Figure(data=[
        go.Surface(z=q, x=X, y=Y, colorscale='RdYlGn_r', cmin=0, cmax=SBC * 1.5)
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Footing Length (m)',
            yaxis_title='Footing Width (m)',
            zaxis_title='Pressure (kN/m²)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        coloraxis_colorbar=dict(title="Pressure (q)")
    )
    return fig

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")
st.title("🏗️ Isolated Foundation Design | IS 456:2000")
st.caption("Enhanced tool with detailed two-way reinforcement, structural checks, and load case management.")

# Initialize or load load cases dataframe
default_data = {
    'Case ID': ['DL+LL', '1.5(DL+LL)'],
    'P_Service (kN)': [600.0, 900.0],
    'Mu_x (kNm)': [20.0, 30.0],
    'Mu_z (kNm)': [15.0, 22.5],
    'Factor': [1.0, 1.5]
}
default_df = pd.DataFrame(default_data)

st.sidebar.header("1️⃣ Project Inputs")
st.sidebar.markdown("---")

# Material Properties
st.sidebar.subheader("Material & Soil")
fc = st.sidebar.selectbox("Concrete Grade (fck) [N/mm²]", options=[20, 25, 30, 35, 40], index=1)
fy = st.sidebar.selectbox("Steel Grade (fy) [N/mm²]", options=[415, 500], index=0)
SBC = st.sidebar.number_input("Safe Bearing Capacity (SBC) [kN/m²]", value=200.0, step=10.0, min_value=10.0)
gamma_c = st.sidebar.number_input("Unit Weight of Concrete [kN/m³]", value=25.0, step=1.0)
st.sidebar.markdown("---")

# Column Dimensions
st.sidebar.subheader("Column Dimensions")
bc = st.sidebar.number_input("Column Width (bc) [m]", value=0.40, step=0.05)
dc = st.sidebar.number_input("Column Depth (dc) [m]", value=0.40, step=0.05)
st.sidebar.markdown("---")

# Foundation Geometry (User Sizing Control)
st.sidebar.subheader("2️⃣ Footing Sizing")
L_input = st.sidebar.number_input("Footing Length (L) [m]", value=2.2, step=0.1, min_value=bc)
B_input = st.sidebar.number_input("Footing Width (B) [m]", value=2.2, step=0.1, min_value=dc)
D_input = st.sidebar.number_input("Overall Depth (D) [m]", value=0.50, step=0.05, min_value=0.2)
st.sidebar.markdown("---")

# --- REINFORCEMENT INPUTS (DF equivalent setup) ---
st.sidebar.subheader("3️⃣ Reinforcement Design")

# BOTTOM REBAR
with st.sidebar.expander("⬇️ Bottom Rebar (Moment Steel)", expanded=True):
    st.markdown("**X-direction (Along L, Width B)**")
    phi_bx = st.selectbox("Bar Diameter, $\\phi_x$ [mm]", options=[10, 12, 16, 20], index=1, key="phi_bx")
    s_bx = st.number_input("Spacing, $s_x$ [mm]", value=150.0, step=10.0, min_value=50.0, max_value=300.0, key="s_bx")
    
    st.markdown("**Z-direction (Along B, Width L)**")
    phi_bz = st.selectbox("Bar Diameter, $\\phi_z$ [mm]", options=[10, 12, 16, 20], index=1, key="phi_bz")
    s_bz = st.number_input("Spacing, $s_z$ [mm]", value=150.0, step=10.0, min_value=50.0, max_value=300.0, key="s_bz")

# TOP REBAR
with st.sidebar.expander("⬆️ Top Rebar (Minimum/Temp. Steel)", expanded=False):
    st.markdown("*Note: Top steel provides minimum area required for temperature and shrinkage.*")
    st.markdown("**X-direction (Along L, Width B)**")
    phi_tx = st.selectbox("Bar Diameter, $\\phi_{tx}$ [mm]", options=[8, 10, 12], index=1, key="phi_tx")
    s_tx = st.number_input("Spacing, $s_{tx}$ [mm]", value=250.0, step=10.0, min_value=50.0, max_value=300.0, key="s_tx")

    st.markdown("**Z-direction (Along B, Width L)**")
    phi_tz = st.selectbox("Bar Diameter, $\\phi_{tz}$ [mm]", options=[8, 10, 12], index=1, key="phi_tz")
    s_tz = st.number_input("Spacing, $s_{tz}$ [mm]", value=250.0, step=10.0, min_value=50.0, max_value=300.0, key="s_tz")

# --- LOAD CASE ENTRY ---
st.subheader("Load Case Entry (Copy/Paste or Direct Edit) 📋")
st.info("Input **Service Loads** for SBC check and **Factored Loads** for Strength/Shear checks.")

load_cases_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    column_config={
        'P_Service (kN)': st.column_config.NumberColumn(format="%.1f", help="Unfactored (Service) Axial Load"),
        'Mu_x (kNm)': st.column_config.NumberColumn(format="%.2f", help="Unfactored Moment about X-axis"),
        'Mu_z (kNm)': st.column_config.NumberColumn(format="%.2f", help="Unfactored Moment about Z-axis"),
        'Factor': st.column_config.NumberColumn(format="%.2f", help="Load Factor (e.g., 1.5 for ultimate checks)"),
    },
    key="load_cases_editor"
)

if load_cases_df.empty:
    st.error("Please enter at least one load case.")
else:
    # Identify the critical load cases
    load_cases_df['Pu_Factored'] = load_cases_df['P_Service (kN)'] * load_cases_df['Factor']
    
    critical_service_row = load_cases_df.loc[load_cases_df['P_Service (kN)'].idxmax()]
    P_service_crit = critical_service_row['P_Service (kN)']
    Mucx_service_crit = critical_service_row['Mu_x (kNm)']
    Mucz_service_crit = critical_service_row['Mu_z (kNm)']
    
    critical_ultimate_row = load_cases_df.loc[load_cases_df['Pu_Factored'].idxmax()]
    Pu_crit = critical_ultimate_row['Pu_Factored']
    
    # --- RUN DESIGN AND CHECKS ---
    st.markdown("---")
    st.header("Design Check Results")
    
    try:
        # Run design checks
        design_results = design_footing_checks(
            L=L_input, B=B_input, D=D_input, bc=bc, dc=dc, fc=fc, fy=fy, 
            SBC=SBC, gamma_c=gamma_c, Pu=Pu_crit, P_service=P_service_crit, 
            Mucx=Mucx_service_crit, Mucz=Mucz_service_crit, 
            phi_bx=phi_bx, s_bx=s_bx, phi_bz=phi_bz, s_bz=s_bz, 
            phi_tx=phi_tx, s_tx=s_tx, phi_tz=phi_tz, s_tz=s_tz
        )
        
        # --- DISPLAY CORE RESULTS ---
        colA, colB, colC = st.columns(3)
        
        with colA:
            st.subheader("Footing Geometry")
            st.metric("Footing Size", f"{L_input:.2f} m x {B_input:.2f} m")
            st.metric("Overall Depth (D)", f"{D_input:.2f} m")
            st.metric("Effective Depth ($d_x/d_z$)", f"{design_results['d_x [m]']:.3f} m / {design_results['d_z [m]']:.3f} m")
            
        with colB:
            st.subheader("Bottom Steel (X-Direction)")
            st.metric("Design Moment ($M_u$, x)", f"{design_results['Mu_x [kNm]']:.2f} kNm")
            st.metric("Required $A_{st, bx}$", f"{design_results['Ast_req_x [mm²/m]']:.2f} mm²/m")
            st.metric("Provided $A_{st, bx}$", f"{design_results['Ast_prov_x_perm [mm²/m]']:.2f} mm²/m (T{phi_bx} @ {s_bx} mm)")

        with colC:
            st.subheader("Bottom Steel (Z-Direction)")
            st.metric("Design Moment ($M_u$, z)", f"{design_results['Mu_z [kNm]']:.2f} kNm")
            st.metric("Required $A_{st, bz}$", f"{design_results['Ast_req_z [mm²/m]']:.2f} mm²/m")
            st.metric("Provided $A_{st, bz}$", f"{design_results['Ast_prov_z_perm [mm²/m]']:.2f} mm²/m (T{phi_bz} @ {s_bz} mm)")
        
        
        # --- DESIGN VISUALIZATIONS AND CHECKS ---
        st.markdown("---")
        st.header("Design Visualizations and Final Checks")
        
        colD, colE, colF = st.columns([1, 1, 1])
        
        # 3D Sketch
        with colD:
            st.subheader("3D Footing & Pedestal Sketch")
            st.plotly_chart(plot_footing_3d(L_input, B_input, D_input, bc, dc), use_container_width=True)
            
            final_status = design_results['FINAL_STATUS']
            if final_status == "OK":
                st.success(f"✅ Final Design Status: **PASSED ALL CHECKS**")
            elif "FAIL" in final_status:
                st.error(f"❌ Final Design Status: **FAILED** ({final_status.replace('FAIL_', '')})")
            
        # Base Pressure Diagram
        with colE:
            st.subheader("Soil Base Pressure Diagram")
            st.plotly_chart(plot_base_pressure_diagram(L_input, B_input, P_service_crit, Mucx_service_crit, Mucz_service_crit, SBC), use_container_width=True)
            
            st.markdown("##### SBC Check")
            if design_results['Pressure Status'].startswith("FAIL"):
                st.error(f"FAIL: Max Pressure: {design_results['Max Soil Pressure [kN/m²]']:.2f} kN/m² > SBC: {SBC:.2f} kN/m².")
            else:
                st.success(f"OK: Max Pressure: {design_results['Max Soil Pressure [kN/m²]']:.2f} kN/m² (SBC: {SBC:.2f} kN/m²).")
                
        # Detailed Structural Checks
        with colF:
            st.subheader("Detailed Structural Checks")
            
            st.markdown("##### 1. Bending Capacity (Depth Check)")
            if design_results['Moment Status X'].startswith("FAIL") or design_results['Moment Status Z'].startswith("FAIL"):
                 st.error(f"FAIL: Bending depth is insufficient for moment $M_u$. **Increase D**.")
            else:
                 st.success("Bending Moment Depth Check **(OK)**")
                 
            st.markdown("##### 2. Shear Capacity (Punching/One-Way)")
            if design_results['Punching Shear Status'].startswith("FAIL") or design_results['1W Shear Status'].startswith("FAIL"):
                st.error(f"FAIL: Shear stresses exceeded (Punching $\\tau_{{vp}}$: {design_results['Punching Shear Stress (τvp) [N/mm²]']:.3f}). **Increase D**.")
            else:
                st.success(f"Shear Checks **(OK)**. $\\tau_{{vp}}$: {design_results['Punching Shear Stress (τvp) [N/mm²]']:.3f} N/mm².")

            st.markdown("##### 3. Reinforcement Checks")
            if final_status == "FAIL_REBAR":
                st.error(f"FAIL: Provided bottom steel is less than required for strength.")
            elif final_status == "FAIL_TOP_MIN_REBAR":
                 st.warning(f"FAIL: Provided top steel ({(design_results['Ast_prov_top_total [mm²]'] / 100):.2f} cm²) is less than minimum required ({(design_results['Ast_min_temp [mm²]'] / 100):.2f} cm²).")
            else:
                st.success("Reinforcement Areas **(OK)**.")

    except Exception as e:
        if "FAIL_GEOMETRY" in str(design_results):
            st.error(f"FATAL ERROR: {design_results['Error']}")
        else:
            st.error(f"An unexpected error occurred during design calculation. Error detail: {e}")
