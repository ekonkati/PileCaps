import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import fsolve

# --- IS 456:2000 DESIGN CONSTANTS ---

# Simplified lookup for Tau_c_max (Table 20, IS 456:2000)
TAU_C_MAX = {
    20: 2.8, 25: 3.1, 30: 3.5, 35: 3.7, 40: 4.0
}

# Simplified lookup for percentage steel pt (Table 19, IS 456:2000)
# This is a highly simplified interpolation for illustrative purposes
def get_tau_c(fck, pt):
    pt = max(0.15, min(pt, 2.0)) # Limit pt between 0.15% and 2.0%
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

def calculate_ast_required(Mu_kNm, B_m, d_m, fc, fy):
    """Calculates required steel area (Ast) in mm² per meter width."""
    # Convert Mu to N-mm
    Mu_Nmm = Mu_kNm * 10**6
    B_mm = B_m * 1000
    d_mm = d_m * 1000
    
    # Check if moment exceeds section capacity (Mu_limit)
    if fy == 415:
        Mu_lim = 0.138 * fc * B_mm * d_mm**2
    else: # Fe 500
        Mu_lim = 0.133 * fc * B_mm * d_mm**2
        
    if Mu_Nmm > Mu_lim:
        return 99999.0, "FAIL_DEPTH" # Sentinel value for depth failure

    # Calculate R_term (term inside sqrt)
    R_term = (4.6 * Mu_Nmm) / (fc * B_mm * d_mm**2)
    
    if R_term >= 1.0:
        return 99999.0, "FAIL_MATH" # Math error/impossible design

    # Ast calculation (mm² per meter)
    Ast_req_m2 = (0.5 * fc / fy) * (1 - np.sqrt(1 - R_term)) * B_mm * d_mm
    
    # Ast_min (0.12% for Fe415/500)
    Ast_min_m2 = 0.0012 * B_mm * d_mm
    
    Ast_final = max(Ast_req_m2, Ast_min_m2)
    return Ast_final, "OK"

def one_way_shear_check(Pu, q_net_u, L, B, bc, d, fc, Ast_req, Ast_prov):
    """Performs one-way shear check (critical section at 'd' from column face)."""
    
    # Critical section from column face
    a_crit = (L - bc) / 2 - d
    
    if a_crit <= 0: # Column is wider than footing overhang + d
        return 0, 0, 0, "OK_DEEP"

    # Shear Force (Vu)
    Vu = q_net_u * B * a_crit
    
    # Nominal Shear Stress (tau_v)
    tau_v = Vu * 1000 / (B * 1000 * d * 1000) # (N) / (mm * mm) = N/mm2
    
    # Permissible Shear Stress (tau_c)
    # Use provided steel ratio
    pt_prov = (Ast_prov / (B * 1000 * d * 1000)) * 100 
    tau_c = get_tau_c(fc, pt_prov)
    
    result = "OK" if tau_v < tau_c else "FAIL"
    
    return Vu, tau_v, tau_c, result

def punching_shear_check(Pu, q_net_u, L, B, bc, dc, d, fc):
    """Performs punching shear check (critical section at 'd/2' from column face)."""
    
    # Critical perimeter dimensions
    b_1 = bc + d
    d_1 = dc + d
    bo = 2 * (b_1 + d_1)
    
    # Area inside the critical perimeter
    A_crit = b_1 * d_1
    
    # Punching Shear Force (Vu_p)
    Vu_p = Pu - q_net_u * A_crit
    
    # Nominal Punching Shear Stress (tau_vp)
    tau_vp = Vu_p * 1000 / (bo * d * 1000) # (N) / (mm * mm) = N/mm2
    
    # Permissible Punching Shear Stress (tau_c_p)
    beta_c = min(bc, dc) / max(bc, dc)
    ks = min(0.5 + beta_c, 1.0)
    tau_c_prime = 0.25 * np.sqrt(fc)
    
    tau_c_p = ks * tau_c_prime
    
    # Check 1: Tau_vp vs Tau_c_p
    result_c = "OK" if tau_vp < tau_c_p else "FAIL"
    
    # Check 2: Tau_vp vs Tau_c_max
    tau_c_max = TAU_C_MAX.get(fc, 3.1) # Default M25
    result_max = "OK" if tau_vp < tau_c_max else "FAIL_MAX"
    
    result = "FAIL" if result_c == "FAIL" else result_max if result_max == "FAIL_MAX" else "OK"
    
    return Vu_p, tau_vp, tau_c_p, tau_c_max, result

def design_footing_checks(L, B, D, bc, dc, fc, fy, SBC, gamma_c, Pu, P_service, Mucx, Mucz, phi, spacing):
    """Runs all checks and returns a comprehensive dictionary of results."""

    d = D - 0.075 # Assume 75mm effective cover (0.075m)
    
    # 1. Base Pressure Check
    A_actual = L * B
    W_footing = gamma_c * L * B * D
    P_total_service = P_service + W_footing
    
    e_x = abs(Mucx / P_service)
    e_z = abs(Mucz / P_service)
    
    if e_x > L / 6 or e_z > B / 6:
        q_max = 9999.0
        pressure_status = "FAIL (Uplift/High Eccentricity)"
    else:
        q_max = P_total_service / A_actual + (6 * Mucx) / (B * L**2) + (6 * Mucz) / (L * B**2)
        q_min = P_total_service / A_actual - (6 * Mucx) / (B * L**2) - (6 * Mucz) / (L * B**2)
        
        pressure_status = "OK" if q_max < SBC else "FAIL"
        if q_min < 0:
             pressure_status = "FAIL (Tension/Uplift)"
        
    # 2. Bending Check (Design Moment)
    q_net_u = Pu / A_actual # Factored net upward pressure
    
    # Moment in X-direction (M_u_x, governs Ast_x)
    a_x = (L - bc) / 2
    M_u_x = q_net_u * B * a_x**2 / 2 # kNm
    
    Ast_req_x, moment_status = calculate_ast_required(M_u_x, B, d, fc, fy)
    
    # 3. Provided Steel Calculation
    As_bar = np.pi * (phi/1000)**2 / 4 # Area of one bar (m²)
    Ast_prov_x = (B * As_bar) / (spacing/1000) # Provided steel in m² over length L
    
    # 4. Shear Checks
    Vu_1w, tau_v_1w, tau_c_1w, shear_1w_status = one_way_shear_check(Pu, q_net_u, L, B, bc, d, fc, Ast_req_x, Ast_prov_x)
    
    Vu_p, tau_v_p, tau_c_p, tau_c_max, shear_p_status = punching_shear_check(Pu, q_net_u, L, B, bc, dc, d, fc)
    
    # 5. Final Status
    final_status = "OK"
    if pressure_status.startswith("FAIL"): final_status = "FAIL"
    if moment_status.startswith("FAIL"): final_status = "FAIL"
    if shear_1w_status.startswith("FAIL"): final_status = "FAIL"
    if shear_p_status.startswith("FAIL"): final_status = "FAIL"

    # RESULTS DICTIONARY
    results = {
        "Footing Length (L) [m]": L,
        "Footing Width (B) [m]": B,
        "Trial Depth (D) [m]": D,
        "Effective Depth (d) [m]": d,
        "Max Soil Pressure [kN/m²]": q_max,
        "Pressure Status": pressure_status,
        "Design Moment (Mu_x) [kNm]": M_u_x,
        "Req. Steel (Ast_x) [mm²/m]": Ast_req_x if isinstance(Ast_req_x, (int, float)) and Ast_req_x < 99999 else 0,
        "Moment Status": moment_status,
        "1W Shear Force (Vu) [kN]": Vu_1w,
        "1W Shear Stress (τv) [N/mm²]": tau_v_1w,
        "1W Permissible (τc) [N/mm²]": tau_c_1w,
        "1W Shear Status": shear_1w_status,
        "Punching Shear Force (Vup) [kN]": Vu_p,
        "Punching Shear Stress (τvp) [N/mm²]": tau_v_p,
        "Punching Permissible (τcp) [N/mm²]": tau_c_p,
        "Punching Shear Status": shear_p_status,
        "FINAL_STATUS": final_status
    }
    return results

# --- PLOTLY VISUALIZATIONS ---

def plot_base_pressure_diagram(L, B, P_total_service, Mucx, Mucz, SBC):
    """Creates a 3D surface plot of the soil pressure distribution."""
    x = np.linspace(0, L, 50)
    y = np.linspace(0, B, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate pressure q(x,y)
    # Origin (0,0) is assumed at the corner of the footing. Center is (L/2, B/2)
    # x' = x - L/2, y' = y - B/2 (coordinates relative to centroid)
    
    q = (P_total_service / (L * B)) + \
        (Mucx * (X - L/2)) / (L * B**3 / 12) + \
        (Mucz * (Y - B/2)) / (B * L**3 / 12)
        
    # Clip pressure to zero for tension zones (simple approximation)
    q[q < 0] = 0

    fig = go.Figure(data=[
        go.Surface(z=q, x=X, y=Y, colorscale='RdYlGn_r', cmin=0, cmax=SBC * 1.5)
    ])
    
    fig.update_layout(
        title='Soil Base Pressure Diagram (kN/m²)',
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
st.caption("Enhanced design tool with detailed structural checks, rebar selection, and load case management.")

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

# Reinforcement Selection
st.sidebar.subheader("3️⃣ Reinforcement Design")
rebar_phi = st.sidebar.selectbox("Bar Diameter (φ) [mm]", options=[10, 12, 16, 20], index=1)
rebar_spacing_mm = st.sidebar.number_input("Spacing (s) [mm]", value=150.0, step=10.0, min_value=50.0, max_value=300.0)

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
    # 1. Critical for Area/SBC (Max P_Service)
    critical_service_row = load_cases_df.loc[load_cases_df['P_Service (kN)'].idxmax()]
    P_service_crit = critical_service_row['P_Service (kN)']
    Mucx_service_crit = critical_service_row['Mu_x (kNm)']
    Mucz_service_crit = critical_service_row['Mu_z (kNm)']
    
    # 2. Critical for Strength (Max Factored Load Pu)
    load_cases_df['Pu_Factored'] = load_cases_df['P_Service (kN)'] * load_cases_df['Factor']
    load_cases_df['Mux_Factored'] = load_cases_df['Mu_x (kNm)'] * load_cases_df['Factor']
    load_cases_df['Muz_Factored'] = load_cases_df['Mu_z (kNm)'] * load_cases_df['Factor']
    
    critical_ultimate_row = load_cases_df.loc[load_cases_df['Pu_Factored'].idxmax()]
    Pu_crit = critical_ultimate_row['Pu_Factored']
    Mucx_crit = critical_ultimate_row['Mux_Factored']
    Mucz_crit = critical_ultimate_row['Muz_Factored']
    
    # --- RUN DESIGN AND CHECKS ---
    st.markdown("---")
    st.header("Design Check Results")
    
    # Run design checks using the critical factored loads for strength and service loads for pressure
    design_results = design_footing_checks(
        L=L_input, B=B_input, D=D_input, bc=bc, dc=dc, fc=fc, fy=fy, 
        SBC=SBC, gamma_c=gamma_c, Pu=Pu_crit, P_service=P_service_crit, 
        Mucx=Mucx_service_crit, Mucz=Mucz_service_crit, 
        phi=rebar_phi, spacing=rebar_spacing_mm
    )

    # --- DISPLAY CORE RESULTS ---
    colA, colB, colC = st.columns(3)
    
    with colA:
        if design_results['FINAL_STATUS'] == "OK":
            st.success(f"✅ Design Status: {design_results['FINAL_STATUS']}")
        else:
            st.error(f"❌ Design Status: {design_results['FINAL_STATUS']}")
        
        st.metric("Critical Factored Load (Pu)", f"{Pu_crit:.2f} kN")
        st.metric("Max Base Pressure (q_max)", f"{design_results['Max Soil Pressure [kN/m²]']:.2f} kN/m²")
        st.metric("Req. Steel (Ast, X-Dir)", f"{design_results['Req. Steel (Ast_x) [mm²/m]']:.2f} mm²/m")

    with colB:
        st.subheader("Footing Geometry")
        st.metric("Footing Size", f"{L_input:.2f} m x {B_input:.2f} m")
        st.metric("Overall Depth", f"{D_input:.2f} m")
        st.metric("Effective Depth (d)", f"{design_results['Effective Depth (d) [m]']:.3f} m")
        
    with colC:
        st.subheader("Provided Reinforcement")
        Ast_req_x_m2 = design_results['Req. Steel (Ast_x) [mm²/m]'] / 1000000 # convert to m²/m
        
        # Calculate number of bars
        As_bar = np.pi * (rebar_phi/1000)**2 / 4 # Area of one bar (m²)
        N_bars = np.ceil(L_input / (rebar_spacing_mm / 1000))
        
        st.metric("Bar Diameter (φ)", f"T{rebar_phi} mm")
        st.metric("Spacing (s)", f"{rebar_spacing_mm:.0f} mm")
        st.metric("Provided Bars (Nos)", f"{int(N_bars)} Nos. (over {B_input}m width)")
        
        # Check Ast Provided vs Required
        Ast_prov_x_m2 = (B_input * As_bar) / (rebar_spacing_mm / 1000) # Provided steel in m² over width B
        Ast_req_total = Ast_req_x_m2 * B_input # Total required steel in m² over width B
        
        if Ast_prov_x_m2 > Ast_req_total:
             st.success(f"Ast Provided ({Ast_prov_x_m2*10000:.2f} cm²/m) > Required ({(Ast_req_total/B_input)*10000:.2f} cm²/m) **(OK)**")
        else:
             st.error(f"Ast Provided ({Ast_prov_x_m2*10000:.2f} cm²/m) < Required ({(Ast_req_total/B_input)*10000:.2f} cm²/m) **(FAIL)**")


    # --- BASE PRESSURE DIAGRAM ---
    st.markdown("---")
    st.header("Base Pressure & Shear Check Diagrams")
    colD, colE = st.columns([1, 1])

    with colD:
        st.plotly_chart(plot_base_pressure_diagram(L_input, B_input, P_service_crit, Mucx_service_crit, Mucz_service_crit, SBC), use_container_width=True)
    
    with colE:
        st.subheader("Detailed Structural Checks")
        
        st.markdown("##### 1. SBC and Moment Checks")
        if design_results['Pressure Status'].startswith("FAIL"):
            st.error(f"SBC Check: **{design_results['Pressure Status']}**. Required Area is larger.")
        elif design_results['Moment Status'].startswith("FAIL"):
             st.error(f"Moment Check: **{design_results['Moment Status']}**. D is too shallow.")
        else:
             st.success("SBC and Bending Moment Checks **(OK)**")

        st.markdown("##### 2. One-Way Shear Check (Beam Shear)")
        if design_results['1W Shear Status'].startswith("FAIL"):
            st.error(f"FAIL: τv ({design_results['1W Shear Stress (τv) [N/mm²]']:.3f}) > τc ({design_results['1W Permissible (τc) [N/mm²]']:.3f})")
        else:
            st.success(f"OK: τv ({design_results['1W Shear Stress (τv) [N/mm²]']:.3f}) < τc ({design_results['1W Permissible (τc) [N/mm²]']:.3f})")
            st.caption(f"Shear Force $V_u$: {design_results['1W Shear Force (Vu) [kN]']:.2f} kN")

        st.markdown("##### 3. Punching Shear Check")
        if design_results['Punching Shear Status'].startswith("FAIL"):
            st.error(f"FAIL: τvp ({design_results['Punching Shear Stress (τvp) [N/mm²]']:.3f}) > τcp ({design_results['Punching Permissible (τcp) [N/mm²]']:.3f})")
        else:
            st.success(f"OK: τvp ({design_results['Punching Shear Stress (τvp) [N/mm²]']:.3f}) < τcp ({design_results['Punching Permissible (τcp) [N/mm²]']:.3f})")
            st.caption(f"Shear Force $V_{{up}}$: {design_results['Punching Shear Force (Vup) [kN]']:.2f} kN")
