import streamlit as st
import numpy as np
import pandas as pd
import math

# --- 1. CORE ENGINEERING FUNCTIONS (BS 8110 IMPLEMENTATION) ---

# Global constants for BS 8110
GAMMA_C = 1.5  # Partial safety factor for concrete
GAMMA_S = 1.05 # Partial safety factor for steel

def get_vc_bs8110(fcu, rho, d):
    """
    Calculates the design concrete shear stress capacity (vc) based on simplified BS 8110.
    This is an approximation of the complex tabular look-up and interpolation.
    
    Args:
        fcu (float): Concrete characteristic strength (N/mm²)
        rho (float): Reinforcement ratio (As_prov / (b*d))
        d (float): Effective depth (mm)
        
    Returns:
        float: vc (N/mm²)
    """
    # Cl. 3.4.5.4: Base design shear stress vc, in N/mm²
    # Max rho is 3.0%
    rho = min(rho * 100, 3.0) 
    
    # fcu term (max fcu is 40 N/mm²)
    fcu_term = (fcu / 25) ** (1/3)
    fcu_term = min(fcu_term, (40 / 25) ** (1/3))
    
    # d term (max d is 400 mm for the d term in vc)
    d_term = (400 / d) ** (1/4)
    d_term = min(d_term, 1.0)
    
    # Simplified ACI/BS approximation for vc (BS 8110 Table 3.9 is complex)
    # vc = 0.79 * fcu_term * d_term * (100 * rho)**(1/3) / GAMMA_C 
    # Using the standard approximate form for 100*As/bd
    vc_base = 0.79 * d_term * fcu_term * (100 * rho) ** (1/3) / GAMMA_C
    
    # Minimum vc for 0.15% steel
    vc_min = 0.35 # Simplified min for grade 35 concrete (0.35 to 0.4 N/mm2)

    return max(vc_base, vc_min)

def design_bs8110_flexure(Mu_kNm, fcu, fy, b_mm, d_mm):
    """
    Calculates required steel area (As) for a singly reinforced section (BS 8110).
    Mu is in kNm, all dimensions in mm, stresses in N/mm².
    """
    Mu = Mu_kNm * 10**6  # Convert kNm to Nmm
    
    # Step 1: Calculate K
    K = Mu / (fcu * b_mm * d_mm**2)
    
    # K' (Limit for singly reinforced section - Cl. 3.4.4.4)
    K_prime = 0.156
    
    if K > K_prime:
        return 0, 0, "FAIL (Section too small or over-reinforced)"

    # Step 2: Calculate lever arm ratio (z)
    z = d_mm * (0.5 + math.sqrt(0.25 - K / 0.9))
    z = min(z, 0.95 * d_mm) # Limit z <= 0.95d
    
    # Step 3: Calculate required steel area (As_req)
    As_req = Mu / ((fy / GAMMA_S) * z)
    
    return As_req, z, "OK"

def calculate_pile_reactions(P_total, Mx, My, pile_coords_df):
    """
    Calculates the axial load (V_i) on each pile using the rigid cap theory.
    P_total (kN), Mx (kNm), My (kNm)
    """
    pile_count = len(pile_coords_df)
    if pile_count == 0:
        return None, 0, 0
    
    df = pile_coords_df.copy()
    
    # 1. Calculate moment of inertia for the group (m^2)
    df['x_i^2'] = df['x_i (m)']**2
    df['y_i^2'] = df['y_i (m)']**2
    
    Ix_group = df['y_i^2'].sum()
    Iy_group = df['x_i^2'].sum()
    
    # 2. Calculate the pile reactions
    results = {}
    max_reaction = 0.0
    
    for index, row in df.iterrows():
        x_i = row['x_i (m)']
        y_i = row['y_i (m)']
        pile_name = row['Pile']
        
        # Axial Load (P/N)
        V_N = P_total / pile_count
        
        # Moment Contribution (Mx causes stress change in Y direction, My in X direction)
        V_Mx = (Mx * y_i / Ix_group) if Ix_group != 0 else 0
        V_My = (My * x_i / Iy_group) if Iy_group != 0 else 0
        
        # The critical unfactored reaction V_i for capacity check (max compression)
        V_i = V_N + V_Mx + V_My
        
        max_reaction = max(max_reaction, V_i)
        
        results[pile_name] = {'V_N': V_N, 'V_Mx': V_Mx, 'V_My': V_My, 'V_i': V_i}
        
    reactions_df = pd.DataFrame.from_dict(results, orient='index')
    reactions_df.index.name = "Pile"
    
    return reactions_df, Ix_group, Iy_group, max_reaction

# --- 2. STREAMLIT UI LAYOUT & DATA INPUT ---

st.set_page_config(layout="wide", page_title="CID Pile Cap Design (BS 8110)")

st.title("🏗️ CID Pile Cap Design Calculator (BS 8110)")
st.markdown("---")

# Initialize state for pile coordinates if not present
if 'pile_coords' not in st.session_state:
    st.session_state.pile_coords = []

# --- SIDEBAR: Materials & Geometry ---
with st.sidebar:
    st.header("1. Material Properties")
    fcu = st.number_input("Concrete Grade ($f_{cu}$, N/mm²)", value=35.0, min_value=20.0, step=5.0)
    fy = st.number_input("Reinforcement Yield ($f_{y}$, N/mm²)", value=460.0, min_value=250.0, step=100.0)
    pile_capacity = st.number_input("Allowable Pile Capacity (kN)", value=420.0, min_value=50.0, step=50.0)
    
    st.header("2. Cap Geometry & Reinforcement")
    cap_L = st.number_input("Cap Length (L, m)", value=1.5, min_value=0.5, step=0.1)
    cap_B = st.number_input("Cap Width (B, m)", value=1.0, min_value=0.5, step=0.1)
    cap_H = st.number_input("Cap Overall Depth (H, mm)", value=750, min_value=300, step=50)
    column_A = st.number_input("Column Short Dim (A, mm)", value=350, min_value=150, step=50)
    column_C = st.number_input("Column Long Dim (C, mm)", value=400, min_value=150, step=50)
    pile_dia = st.number_input("Pile Diameter ($\emptyset$, mm)", value=500, min_value=200, step=50)
    cover = st.number_input("Nominal Cover (mm)", value=75, min_value=50, step=5)
    bar_dia = st.number_input("Main Bar Diameter ($\emptyset_{bar}$, mm)", value=16, min_value=10, step=2)
    
    # Calculate effective depth 'd'
    # Assuming two layers of steel, with L direction being the first (d1) and B direction second (d2)
    d1 = cap_H - cover - bar_dia / 2
    d2 = cap_H - cover - bar_dia - bar_dia / 2 # Assuming same bar diameter for both layers
    st.metric("Effective Depth 1 ($d_1$)", f"{d1:.0f} mm")
    st.metric("Effective Depth 2 ($d_2$)", f"{d2:.0f} mm")
    
# --- MAIN PANEL: Loads and Layout ---
col_L, col_layout = st.columns([2, 1])

with col_L:
    st.header("3. Applied Column Actions (Characteristic)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dead ($G_k$)")
        P_Gk = st.number_input("Axial (kN)", key="Gk_P", value=218.0)
        Mx_Gk = st.number_input("Moment $M_x$ (kNm)", key="Gk_Mx", value=23.2)
        My_Gk = st.number_input("Moment $M_y$ (kNm)", key="Gk_My", value=0.0)
    with col2:
        st.subheader("Imposed ($Q_k$)")
        P_Qk = st.number_input("Axial (kN)", key="Qk_P", value=104.2)
        Mx_Qk = st.number_input("Moment $M_x$ (kNm)", key="Qk_Mx", value=10.4)
        My_Qk = st.number_input("Moment $M_y$ (kNm)", key="Qk_My", value=0.0)
    with col3:
        st.subheader("Wind ($W_k$)")
        P_Wk = st.number_input("Axial (kN)", key="Wk_P", value=27.5)
        Mx_Wk = st.number_input("Moment $M_x$ (kNm)", key="Wk_Mx", value=2.7)
        My_Wk = st.number_input("Moment $M_y$ (kNm)", key="Wk_My", value=0.0)

with col_layout:
    st.header("4. Pile Layout ($\pm x, \pm y$)")
    num_piles = st.number_input("Number of Piles ($N$)", value=4, min_value=1, max_value=25, step=1)
    
    s_default = 0.9 # Default spacing for common cap sizes
    
    layout_presets = {
        '2 Piles': [('P1', 0.0, s_default/2), ('P2', 0.0, -s_default/2)],
        '3 Piles': [('P1', 0.0, s_default/math.sqrt(3)), ('P2', s_default/2, -s_default/(2*math.sqrt(3))), ('P3', -s_default/2, -s_default/(2*math.sqrt(3)))],
        '4 Piles (2x2)': [('P1', s_default/2, s_default/2), ('P2', -s_default/2, s_default/2), ('P3', s_default/2, -s_default/2), ('P4', -s_default/2, -s_default/2)],
        '6 Piles (3x2)': [('P1', s_default, s_default/2), ('P2', 0.0, s_default/2), ('P3', -s_default, s_default/2), ('P4', s_default, -s_default/2), ('P5', 0.0, -s_default/2), ('P6', -s_default, -s_default/2)],
        'Custom (Edit Table Below)': st.session_state.pile_coords
    }
    
    preset_choice = st.selectbox("Load Pile Layout Preset", list(layout_presets.keys()))
    
    if preset_choice != 'Custom (Edit Table Below)':
        st.session_state.pile_coords = layout_presets[preset_choice][:num_piles] # Truncate if fewer piles requested
        
    # Editable DataFrame for custom coordinates
    pile_layout_df_edit = pd.DataFrame(st.session_state.pile_coords, columns=['Pile', 'x_i (m)', 'y_i (m)'])
    pile_layout_df_edit.set_index('Pile', inplace=True)
    
    st.markdown("Edit/Verify Pile Coordinates (Centered at 0,0):")
    edited_df = st.data_editor(pile_layout_df_edit, num_rows="dynamic", use_container_width=True)
    st.session_state.pile_coords = list(edited_df.reset_index().itertuples(index=False, name=None))


# --- 5. CALCULATION AND OUTPUT ---
st.markdown("---")
if st.button("Run Pile Cap Design Checks", use_container_width=True):
    
    if len(st.session_state.pile_coords) < 1:
        st.error("Please define at least one pile coordinate.")
        st.stop()
        
    # --- Load Combinations ---
    # ULS Case 1: Max gravity (1.4Gk + 1.6Qk)
    P_ULT1 = 1.4 * P_Gk + 1.6 * P_Qk
    Mx_ULT1 = 1.4 * Mx_Gk + 1.6 * Mx_Qk
    My_ULT1 = 1.4 * My_Gk + 1.6 * My_Qk
    
    # ULS Case 2: Max wind (1.2Gk + 1.2Qk + 1.2Wk)
    P_ULT2 = 1.2 * (P_Gk + P_Qk + P_Wk)
    Mx_ULT2 = 1.2 * (Mx_Gk + Mx_Qk + Mx_Wk)
    My_ULT2 = 1.2 * (My_Gk + My_Qk + My_Wk)

    # SLS Case (1.0Gk + 1.0Qk) - Used for pile capacity check (required by code)
    P_SERVICE = P_Gk + P_Qk
    Mx_SERVICE = Mx_Gk + Mx_Qk
    My_SERVICE = My_Gk + My_Qk
    
    # Critical ULS Moments for Flexural/Shear Design
    Mu_x_design = max(abs(Mx_ULT1), abs(Mx_ULT2))
    Mu_y_design = max(abs(My_ULT1), abs(My_ULT2))
    Pu_design = max(P_ULT1, P_ULT2)

    # --- TABBED OUTPUT ---
    tab1, tab2, tab3 = st.tabs(["Pile Reaction & Capacity", "Flexural Design", "Shear Checks"])

    # --- TAB 1: PILE REACTION & CAPACITY ---
    with tab1:
        st.subheader(f"Service Load Pile Reactions (P = {P_SERVICE:.1f} kN, Mx = {Mx_SERVICE:.1f} kNm, My = {My_SERVICE:.1f} kNm)")
        
        reactions_df, Ix, Iy, max_reaction = calculate_pile_reactions(P_SERVICE, Mx_SERVICE, My_SERVICE, edited_df.reset_index())
        
        if reactions_df is not None:
            reactions_df['Capacity Limit'] = pile_capacity
            reactions_df['Capacity Check'] = np.where(reactions_df['V_i'] > reactions_df['Capacity Limit'], 
                                                     '<span style="color:red;font-weight:bold;">FAIL</span>', 
                                                     '<span style="color:green;font-weight:bold;">OK</span>')
            
            reactions_df.rename(columns={'V_N': '$V_{P/N}$ (kN)', 'V_Mx': '$V_{M_x}$ (kN)', 'V_My': '$V_{M_y}$ (kN)', 'V_i': '$V_{i,max}$ (kN)'}, inplace=True)
            
            st.markdown(reactions_df.to_html(escape=False), unsafe_allow_html=True)
            
            st.markdown(f"**Group Inertia:** $I_x = {Ix:.3f}$ m$^4$, $I_y = {Iy:.3f}$ m$^4$")

            if max_reaction > pile_capacity:
                st.error(f"**CAPACITY FAILURE:** Max pile reaction ({max_reaction:.1f} kN) exceeds allowable capacity ({pile_capacity:.1f} kN). Increase $N$ or cap size.")
            else:
                st.success(f"Pile capacity check: OK (Max $V_i$ is {max_reaction:.1f} kN, Capacity is {pile_capacity:.1f} kN)")
        else:
            st.warning("No pile coordinates defined for reaction check.")

    # --- TAB 2: FLEXURAL DESIGN ---
    with tab2:
        st.subheader("Flexural Reinforcement Design (Ultimate Limit State)")
        st.markdown(f"**Design Inputs:** $f_{{cu}}={fcu:.0f}$ N/mm², $f_{{y}}={fy:.0f}$ N/mm²")
        
        # Design for M_x (Long Direction - Steel perp. to L, width B)
        b_x = cap_B * 1000
        d_x = d1 # Use d1 (largest d)
        
        # For simplicity, critical Mu is taken as the maximum ULS moment here.
        # In a real sheet, Mu is calculated from pile reactions at the column face.
        # Mu_x_crit = (Sum of factored pile loads * moment arm 'a') - M_column_face_from_loads 
        # For this template, we'll use a simplified critical moment:
        Mu_x_design = max(abs(Mx_ULT1), abs(Mx_ULT2))
        
        As_x_req, z_x, status_x = design_bs8110_flexure(Mu_x_design, fcu, fy, b_x, d_x)
        
        # Design for M_y (Short Direction - Steel perp. to B, width L)
        b_y = cap_L * 1000
        d_y = d2 # Use d2 (smallest d)
        
        Mu_y_design = max(abs(My_ULT1), abs(My_ULT2))
        
        As_y_req, z_y, status_y = design_bs8110_flexure(Mu_y_design, fcu, fy, b_y, d_y)
        
        flex_data = {
            "Direction": ["X-Dir (perp. to L)", "Y-Dir (perp. to B)"],
            "Critical Moment ($M_{u}$, kNm)": [f"{Mu_x_design:.2f}", f"{Mu_y_design:.2f}"],
            "Design Width ($b$, mm)": [f"{b_x:.0f}", f"{b_y:.0f}"],
            "Effective Depth ($d$, mm)": [f"{d_x:.0f}", f"{d_y:.0f}"],
            "Required Steel ($A_{s,req}$, mm²)": [f"{As_x_req:.0f}", f"{As_y_req:.0f}"],
            "Status": [status_x, status_y]
        }
        st.dataframe(flex_data, hide_index=True, use_container_width=True)
        
        if status_x == "FAIL" or status_y == "FAIL":
            st.error("Flexural Failure: Increase cap depth (H) or concrete grade ($f_{cu}$).")
        else:
            st.success("Flexural Design Check: OK")


    # --- TAB 3: SHEAR CHECKS ---
    with tab3:
        st.subheader("One-Way Shear (Beam Shear)")
        st.info("The actual design requires calculating $V_{u}$ at a distance $d$ from the column face, factoring in pile contributions. A full shear flow calculation is needed.")
        
        # Placeholder for One-Way Shear (based on Pu and simplified vc)
        # Assuming As_prov = 0.5% (to get a typical vc value)
        rho_assumed = 0.005 
        
        # Long direction (L) check
        vc_L = get_vc_bs8110(fcu, rho_assumed, d1)
        # Vu_L should be calculated from piles. Using a fraction of Pu for illustration.
        Vu_L_design = Pu_design / 2 # Simplified Vu
        v_L = Vu_L_design * 1000 / (b_x * d1)
        
        # Short direction (B) check
        vc_B = get_vc_bs8110(fcu, rho_assumed, d2)
        Vu_B_design = Pu_design / 2 # Simplified Vu
        v_B = Vu_B_design * 1000 / (b_y * d2)
        
        
        shear1_data = {
            "Direction": ["X-Dir (Long)", "Y-Dir (Short)"],
            "Design Shear ($V_{u}$, kN, assumed)": [f"{Vu_L_design:.1f}", f"{Vu_B_design:.1f}"],
            "Applied Shear Stress ($v$, N/mm²)": [f"{v_L:.3f}", f"{v_B:.3f}"],
            "Concrete Capacity ($v_{c}$, N/mm²)": [f"{vc_L:.3f}", f"{vc_B:.3f}"],
            "Status": ["OK" if v_L < vc_L else "FAIL", "OK" if v_B < vc_B else "FAIL"]
        }
        st.dataframe(shear1_data, hide_index=True, use_container_width=True)
        
        if v_L > vc_L or v_B > vc_B:
            st.error("One-Way Shear Failure: Cap depth (H) is likely insufficient. Increase H.")
        

        st.subheader("Two-Way Shear (Punching Shear) at Column Face")
        st.warning("Punching Shear is the critical failure mode. This check is complex and requires defining a critical perimeter ($u$) at $1.5d$ from the column face and calculating the corresponding $V_{eff}$. The result below is a placeholder.")

        # Simplified Punching Shear Check
        critical_perimeter_u = 2 * (column_A + 3 * d1) + 2 * (column_C + 3 * d1) # Perimeter at 1.5d
        
        # Maximum allowed shear stress v_max (Cl. 3.7.6.2 - v_max = 0.8 * sqrt(fcu) or 5 N/mm²)
        v_max = min(0.8 * math.sqrt(fcu), 5.0) 
        
        # Applied stress is Pu_design / (u * d)
        v_applied_punching = (Pu_design * 1000) / (critical_perimeter_u * d1)
        
        v_punch_status = "OK" if v_applied_punching < v_max else "FAIL"

        punching_data = {
            "Check": ["Max Applied Stress ($v$)", "Max Allowable Stress ($v_{max}$)"],
            "Value (N/mm²)": [f"{v_applied_punching:.3f}", f"{v_max:.3f}"],
            "Status": [v_punch_status, "N/A"]
        }
        st.dataframe(punching_data, hide_index=True, use_container_width=True)
        
        if v_punch_status == "FAIL":
            st.error("Punching Shear Failure: Increase cap depth (H).")
        else:
            st.success("Punching Shear Check (Simplified): OK")
