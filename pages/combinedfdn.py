import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import math

# --- 1. DESIGN FUNCTIONS (BASED ON IS 456:2000 FOR CONCRETE DESIGN) ---

def calculate_base_pressure(Pu, Mz, Mx, L, B):
    """Calculates the pressure at the four corners of the footing."""
    
    # Calculate section modulus
    I_y = (B * L**3) / 12  # Moment of inertia about Z-axis (for Mx/ey eccentricity)
    I_x = (L * B**3) / 12  # Moment of inertia about X-axis (for Mz/ex eccentricity)
    
    A = L * B
    
    # Max/Min pressure points are at the corners (cx = L/2, cy = B/2)
    cx = L / 2
    cy = B / 2
    
    q_avg = Pu / A
    
    # Mx causes bending along the long axis (L), creating pressure variation along B (cy)
    # Mz causes bending along the short axis (B), creating pressure variation along L (cx)
    
    # Corner 1 (Max Pressure) -> x=+L/2, y=+B/2
    q1 = q_avg + (Mx * cy / I_x) + (Mz * cx / I_y) 
    
    # Corner 2 -> x=-L/2, y=+B/2
    q2 = q_avg - (Mx * cy / I_x) + (Mz * cx / I_y)
    
    # Corner 3 -> x=-L/2, y=-B/2
    q3 = q_avg - (Mx * cy / I_x) - (Mz * cx / I_y)
    
    # Corner 4 (Min Pressure) -> x=+L/2, y=-B/2
    q4 = q_avg + (Mx * cy / I_x) - (Mz * cx / I_y)
    
    return {
        'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
        'q_max': max(q1, q2, q3, q4),
        'q_min': min(q1, q2, q3, q4),
        'e_x': Mz / Pu,
        'e_y': Mx / Pu,
    }

def get_ast_required(Mu, fck, fy, d, B):
    """Calculates required steel area (Ast) per unit width (B) using IS 456 Limit State Method."""
    
    if fy == 500:
        k = 0.133
    elif fy == 415:
        k = 0.138
    else: # Fe250
        k = 0.148
        
    # Convert all inputs to N and mm
    B_mm = B * 1000
    d_mm = d * 1000
    Mu_Nmm = Mu * 1e6
        
    Mu_lim = k * fck * B_mm * d_mm**2 / 1e6 # kNm (Used for check only)
        
    if Mu > Mu_lim:
        # Beam is over-reinforced or section depth is insufficient
        return 0 # Indicate failure
    
    # Ast formula from IS 456:2000, Annex G
    Ast_req = (0.5 * fck / fy) * B_mm * d_mm * (1 - math.sqrt(1 - (4.6 * Mu_Nmm) / (fck * B_mm * d_mm**2)))
    
    # Check minimum steel requirement (0.12% of gross area for slabs/footings)
    Ast_min = 0.0012 * B_mm * d_mm
    
    return max(Ast_req, Ast_min) # Ast in mm^2/m

# --- 2. DATA PROCESSING AND APPLICATION LOGIC ---

def process_and_design(df_staad, L, B, D, fck, fy, sbc_gross, sbc_gross_seismic, C_C_Distance, w_pedestal, l_pedestal, gamma_conc, gamma_soil, D_f, Clear_Cover):
    """Main design logic that iterates through load cases."""
    
    df_results = pd.DataFrame()
    
    # 1. Calculate effective depth (d) and self-weight/surcharge
    d_m = (D - Clear_Cover - 12 - 12/2) / 1000.0 # Assuming 12mm bar and 12mm cover adjustment, in meters
    
    # Calculate soil and concrete weight (in kN)
    footing_sw = L * B * D / 1000 * gamma_conc
    soil_surcharge = L * B * (D_f - D / 1000) * gamma_soil
    pedestal_sw_1 = w_pedestal[0] / 1000 * l_pedestal[0] / 1000 * (D_f / 1000 + 0.3) * gamma_conc # Simplified
    pedestal_sw_2 = w_pedestal[1] / 1000 * l_pedestal[1] / 1000 * (D_f / 1000 + 0.3) * gamma_conc # Simplified
    
    sw_surcharge = footing_sw + soil_surcharge + pedestal_sw_1 + pedestal_sw_2
    
    # 2. Iterate through load cases
    for index, row in df_staad.iterrows():
        lc_name = row['L/C']
        
        # Determine allowable SBC and Factored Load based on LC name (heuristic)
        if any(keyword in lc_name.upper() for keyword in ['EQ', 'WL', 'WIND', 'SEISMIC']):
            allowable_sbc = sbc_gross_seismic
            fact_f = 1.2
        else:
            allowable_sbc = sbc_gross
            fact_f = 1.5

        # --- UNFACTORED (SERVICE) LOAD FOR SBC CHECK ---
        
        P1_Fy = row['P1_Fy kN']
        P1_Mx = row['P1_Mx kNm']
        P1_Mz = row['P1_Mz kNm']
        P2_Fy = row['P2_Fy kN']
        P2_Mx = row['P2_Mx kNm']
        P2_Mz = row['P2_Mz kNm']
        
        Pu_unfactored = P1_Fy + P2_Fy + sw_surcharge
        
        # Calculate moments about the CoG of the footing (L/2, B/2)
        # Assuming pedestals are centered along the width (B/2) and spaced C_C_Distance along the length (L)
        # x-coordinate of P1/P2 from footing center: +/- C_C_Distance / 2.0
        x_off = C_C_Distance / 2.0
        
        # Total Moment about X-axis (causes pressure variation along B)
        # Mx_total = M_x(P1) + M_x(P2) + F_y(P1) * y_off + F_y(P2) * y_off (y_off = 0 if centered along width B)
        Mx_total_unfactored = P1_Mx + P2_Mx
        
        # Total Moment about Z-axis (causes pressure variation along L)
        # Mz_total = M_z(P1) + M_z(P2) + F_y(P1) * (-x_off) + F_y(P2) * (+x_off)
        Mz_total_unfactored = P1_Mz + P2_Mz + P1_Fy * (-x_off) + P2_Fy * (x_off)
        
        # Base Pressure Check
        q_pressures = calculate_base_pressure(Pu_unfactored, Mz_total_unfactored, Mx_total_unfactored, L, B)
        q_max = q_pressures['q_max']
        q_min = q_pressures['q_min']
        e_x = q_pressures['e_x']
        e_y = q_pressures['e_y']
        
        sbc_check = "SAFE" if q_max <= allowable_sbc and q_min >= 0 else "UNSAFE"
        ecc_check = "OK" if abs(e_x) <= L/6 and abs(e_y) <= B/6 else "FAIL"

        # --- FACTORED LOAD FOR STRUCTURAL DESIGN ---
        
        P1u_Fy = P1_Fy * fact_f
        P2u_Fy = P2_Fy * fact_f
        
        # Recalculate factored moments about the footing CoG (for net pressure calculation)
        Pu_factored_col = P1u_Fy + P2u_Fy
        Mx_total_factored = P1_Mx * fact_f + P2_Mx * fact_f
        Mz_total_factored = P1_Mz * fact_f + P2_Mz * fact_f + P1u_Fy * (-x_off) + P2u_Fy * (x_off)
        
        # Net Upward Pressure (factored column loads minus self-weight/surcharge)
        # Pressure used for structural design
        q_net_u_pressures = calculate_base_pressure(Pu_factored_col, Mz_total_factored, Mx_total_factored, L, B)
        q_net_u_max = q_net_u_pressures['q_max'] - (sw_surcharge / (L*B)) # Deduct average self-weight from gross factored pressure
        
        # Moment and Shear Calculation (Long Direction - Along L)
        # Max Hogging Moment (between columns)
        # Simplified as a beam supported by soil pressure and loaded by Pu1, Pu2
        Mu_hog = (P1u_Fy * x_off) - (P2u_Fy * x_off) 
        
        # Max Sagging Moment (Cantilever)
        # Critical distance for cantilever: x_cant = (L - C_C_Distance) / 2
        x_cant = (L - C_C_Distance) / 2.0
        q_cant = q_net_u_max
        Mu_sag = (q_cant * B) * (x_cant**2 / 2)
        
        Mu_max = max(abs(Mu_hog), abs(Mu_sag))
        
        # Required Steel Area (Longitudinal - Lengthwise)
        Ast_L = get_ast_required(Mu_max, fck, fy, d_m, B) # Ast in mm^2/m
        
        # Punching Shear Check
        col_w, col_l = w_pedestal[1]/1000, l_pedestal[1]/1000 # Pedestal dimensions in meters
        
        bo = 2 * ((col_w + d_m) + (col_l + d_m)) # Critical perimeter in meters
        
        Vu_punch = P2u_Fy - (q_net_u_max * (col_w + d_m) * (col_l + d_m))
        
        tau_v = (Vu_punch * 1000) / (bo * 1000 * d_m * 1000) # N/mm2
        
        # Allowable shear stress (tau_c')
        tau_c_prime = 0.25 * math.sqrt(fck) # Max allowable as per IS 456
        tau_c_prime = min(tau_c_prime, 1.25) # Max allowable limit
        
        punch_check = "SAFE" if tau_v <= tau_c_prime else "UNSAFE"
        
        
        # Store results
        df_results.loc[index, 'L/C'] = lc_name
        df_results.loc[index, 'Factored'] = fact_f
        df_results.loc[index, 'q_max (kN/m2)'] = q_max
        df_results.loc[index, 'Allow_SBC (kN/m2)'] = allowable_sbc
        df_results.loc[index, 'SBC_CHECK'] = sbc_check
        df_results.loc[index, 'ex (m)'] = e_x
        df_results.loc[index, 'ey (m)'] = e_y
        df_results.loc[index, 'ECC_CHECK'] = ecc_check
        df_results.loc[index, 'q_net_u_max (kN/m2)'] = q_net_u_max
        df_results.loc[index, 'Mu_max (kNm)'] = Mu_max
        df_results.loc[index, 'Ast_L (mm2/m)'] = Ast_L
        df_results.loc[index, 'Punch_tau_v (N/mm2)'] = tau_v
        df_results.loc[index, 'Punch_Allow (N/mm2)'] = tau_c_prime
        df_results.loc[index, 'PUNCH_CHECK'] = punch_check
        
    return df_results, sw_surcharge

# --- 3. STREAMLIT APP LAYOUT ---

def app():
    st.set_page_config(layout="wide", page_title="Combined Foundation Design App")

    st.title("🏗️ Combined Foundation Design for Eccentric Pedestals")
    
    # --- Sidebar for Design Inputs (from DESIGN.csv) ---
    st.sidebar.header("📐 Foundation Geometry & Materials")
    
    # Default values based on WORK.csv and DESIGN.csv
    L = st.sidebar.number_input("Footing Length L (m)", value=3.0, min_value=1.0, step=0.1)
    B = st.sidebar.number_input("Footing Width B (m)", value=3.0, min_value=1.0, step=0.1)
    D_mm = st.sidebar.number_input("Footing Overall Depth D (mm)", value=500, min_value=300, step=50)
    Clear_Cover = st.sidebar.number_input("Clear Cover (mm)", value=75, min_value=50, step=5)
    
    st.sidebar.subheader("Pedestal Data")
    C_C_Distance = st.sidebar.number_input("Column C-C Distance (m)", value=1.5, min_value=0.5, step=0.1)
    w_pedestal_1 = st.sidebar.number_input("Pedestal 1 Width (mm)", value=600, min_value=300, step=50)
    l_pedestal_1 = st.sidebar.number_input("Pedestal 1 Length (mm)", value=400, min_value=300, step=50)
    w_pedestal_2 = st.sidebar.number_input("Pedestal 2 Width (mm)", value=600, min_value=400, step=50)
    l_pedestal_2 = st.sidebar.number_input("Pedestal 2 Length (mm)", value=400, min_value=400, step=50)
    
    st.sidebar.subheader("Material Properties")
    fck = st.sidebar.number_input("Concrete Grade fck (N/mm²)", value=25, min_value=20, max_value=40)
    fy = st.sidebar.number_input("Steel Grade fy (N/mm²)", value=415, min_value=250, step=100)
    gamma_conc = st.sidebar.number_input("Unit weight of Concrete (kN/m³)", value=25.0, step=1.0)
    
    st.sidebar.subheader("Soil Properties")
    gamma_soil = st.sidebar.number_input("Unit Weight of Soil (kN/m³)", value=18.0, step=1.0)
    D_f = st.sidebar.number_input("Depth of foundation Df (m)", value=2.0, step=0.1)
    sbc_gross = st.sidebar.number_input("Gross SBC (DL+LL) (kN/m²)", value=286.0, step=10.0)
    sbc_gross_seismic = st.sidebar.number_input("Gross SBC (W/E) (kN/m²)", value=348.5, step=10.0)

    
    # --- Main Content: STAAD Input Upload (rewired logic) ---
    
    st.header("Upload STAAD Reactions 📥")
    uploaded_file = st.file_uploader("Upload STAAD Reactions CSV file", type="csv")
    
    df_staad = None
    
    if uploaded_file is not None:
        try:
            # --- REWIRED FILE PARSING LOGIC ---
            
            uploaded_data = uploaded_file.getvalue().decode("utf-8")
            data_io = StringIO(uploaded_data)
            
            lines = data_io.readlines()
            header_row_index = -1
            
            # Find the row containing the detailed headers
            for i, line in enumerate(lines):
                if 'L/C' in line and 'Fx kN' in line and 'Mx kNm' in line:
                    header_row_index = i
                    break
            
            if header_row_index != -1:
                data_io.seek(0)
                df_raw = pd.read_csv(data_io, skiprows=header_row_index, header=0, skipinitialspace=True)
                
                # Drop rows that are completely NaN and columns that are completely NaN
                df_raw = df_raw.dropna(axis=0, how='all').dropna(axis=1, how='all')
                
                # CORRECTED Column Selection based on inspection:
                # Column indices (0-based) for: L/C, P1_Fy, P1_Mx, P1_Mz, P2_Fy, P2_Mx, P2_Mz
                # Indices: [1, 3, 5, 7, 11, 13, 15]
                
                # Check if the dataframe has enough columns for iloc
                if df_raw.shape[1] >= 16: 
                    df_staad = df_raw.iloc[:, [1, 3, 5, 7, 11, 13, 15]].copy()
                    df_staad.columns = ['L/C', 'P1_Fy kN', 'P1_Mx kNm', 'P1_Mz kNm', 'P2_Fy kN', 'P2_Mx kNm', 'P2_Mz kNm']
                    
                    for col in df_staad.columns[1:]:
                        df_staad[col] = pd.to_numeric(df_staad[col], errors='coerce')
                        
                    df_staad = df_staad.dropna(subset=df_staad.columns[1:])
                    st.success("STAAD Reactions loaded and parsed successfully!")
                    st.dataframe(df_staad.head())
                else:
                    st.error(f"File parsing error: Expected at least 16 columns but found {df_raw.shape[1]}. Please check file structure.")
                    st.stop()
                    
            else:
                st.error("Could not reliably identify the header row in the uploaded file.")
                st.stop()

        except Exception as e:
            st.error(f"Error reading and parsing the uploaded file: {e}")
            st.stop()

    if df_staad is not None and st.button("Run Foundation Design & Checks"):
        
        # Run the main calculation logic
        w_pedestal = [w_pedestal_1, w_pedestal_2]
        l_pedestal = [l_pedestal_1, l_pedestal_2]
        
        df_results, sw_surcharge = process_and_design(
            df_staad, L, B, D_mm, fck, fy, sbc_gross, sbc_gross_seismic, C_C_Distance, 
            w_pedestal, l_pedestal, gamma_conc, gamma_soil, D_f, Clear_Cover
        )

        st.success("Design calculations complete!")

        # --- 4. Design Summary and Critical Case ---
        
        st.header("✅ Design Summary")
        
        critical_sbc = df_results[df_results['SBC_CHECK'] == 'UNSAFE']
        critical_ecc = df_results[df_results['ECC_CHECK'] == 'FAIL']
        critical_punch = df_results[df_results['PUNCH_CHECK'] == 'UNSAFE']
        
        # ... (Safety check messages as before) ...
        if not critical_sbc.empty or not critical_ecc.empty or not critical_punch.empty:
            st.warning("⚠️ Some checks failed. Review the results and adjust geometry/parameters.")
        else:
            st.success("All checks (SBC, Eccentricity, Punching Shear) are SAFE for all load cases.")

        st.subheader("Structural Design Results (Based on Critical Moment)")
        critical_Mu_row = df_results.loc[df_results['Mu_max (kNm)'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Required $A_{st,L}$ (Bottom Rebar)", f"{critical_Mu_row['Ast_L (mm2/m)']:.0f} mm²/m")
        col2.metric("Max Design Moment $M_u$", f"{critical_Mu_row['Mu_max (kNm)']:.2f} kNm (LC: {critical_Mu_row['L/C']})")
        col3.metric("Max Net Upward Pressure $q_{net,u}$", f"{df_results['q_net_u_max (kN/m2)'].max():.2f} kN/m²")
        col4.metric("Footing $L \\times B \\times D$", f"{L*1000:.0f} x {B*1000:.0f} x {D_mm} mm")
        
        st.subheader("Recommended Rebar Provision (From WORK.csv)")
        st.markdown(
            """
            * **Footing Thickness:** **500 mm**
            * **Footing Reinforcement (Lengthwise & Widthwise):** **T 12@ 200mm C/C** (Both Top and Bottom)
            """
        )
        
        # --- 5. Plotly Sketches ---
        
        st.header("📊 Design Visualizations (Plotly Sketches)")
        
        q_plot_row = df_results.loc[df_results['q_max (kN/m2)'].idxmax()]
        
        # --- Sketch 1: Footing and Pedestals Plan View ---
        st.subheader("1. Footing Plan View")
        # ... (Plotly code for Plan View as before) ...
        
        fig_plan = go.Figure()
        fig_plan.add_shape(type="rect", x0=0, y0=0, x1=L, y1=B, line=dict(color="RoyalBlue", width=2), fillcolor="lightblue", opacity=0.5)
        p1_center_x = L / 2 - C_C_Distance / 2.0
        p1_center_y = B / 2.0
        p1_x0, p1_x1 = p1_center_x - l_pedestal_1 / 2000.0, p1_center_x + l_pedestal_1 / 2000.0
        p1_y0, p1_y1 = p1_center_y - w_pedestal_1 / 2000.0, p1_center_y + w_pedestal_1 / 2000.0
        fig_plan.add_shape(type="rect", x0=p1_x0, y0=p1_y0, x1=p1_x1, y1=p1_y1, line=dict(color="Red", width=2), fillcolor="red", opacity=0.7)
        fig_plan.add_annotation(x=p1_center_x, y=p1_center_y, text="P1", showarrow=False, font=dict(color="white"))
        p2_center_x = L / 2 + C_C_Distance / 2.0
        p2_center_y = B / 2.0
        p2_x0, p2_x1 = p2_center_x - l_pedestal_2 / 2000.0, p2_center_x + l_pedestal_2 / 2000.0
        p2_y0, p2_y1 = p2_center_y - w_pedestal_2 / 2000.0, p2_center_y + w_pedestal_2 / 2000.0
        fig_plan.add_shape(type="rect", x0=p2_x0, y0=p2_y0, x1=p2_x1, y1=p2_y1, line=dict(color="Red", width=2), fillcolor="red", opacity=0.7)
        fig_plan.add_annotation(x=p2_center_x, y=p2_center_y, text="P2", showarrow=False, font=dict(color="white"))
        fig_plan.update_layout(title_text=f"Footing Plan View ({L}m x {B}m)", xaxis_title="Length (L) [m]", yaxis_title="Width (B) [m]", showlegend=False, yaxis_scaleanchor="x", yaxis_scaleratio=1, width=700, height=500)
        st.plotly_chart(fig_plan)


        # --- Sketch 2: Base Pressure Diagram (Along L) ---
        st.subheader("2. Unfactored Base Pressure Distribution (Critical SBC Case)")
        
        x_coords = np.linspace(0, L, 100)
        q_pressure_line = q_plot_row['Pu_unfactored (kN)'] / (L*B) + (q_plot_row['Mz_unfactored (kNm)'] * (L/2 - x_coords) / ((B*L**3)/12))
        
        fig_pressure = go.Figure()
        fig_pressure.add_trace(go.Scatter(x=x_coords, y=q_pressure_line, 
                                          mode='lines', name='Base Pressure', fill='tozeroy'))
        fig_pressure.add_hline(y=q_plot_row['Allow_SBC (kN/m2)'], line_dash="dash", line_color="red", 
                               annotation_text="Allowable SBC (Gross)", annotation_position="top right")
        fig_pressure.update_layout(title_text=f"Base Pressure (Critical LC: {q_plot_row['L/C']})",
                                   xaxis_title="Distance along L (m)", yaxis_title="Pressure (kN/m²)")
        st.plotly_chart(fig_pressure)
        
        # --- Sketch 3: Bending Moment Diagram (Along L) ---
        st.subheader("3. Bending Moment Diagram (Critical Structural Case)")

        x_L = np.linspace(0, L, 100)
        Pu1 = df_staad.loc[critical_Mu_row.name, 'P1_Fy kN'] * critical_Mu_row['Factored']
        Pu2 = df_staad.loc[critical_Mu_row.name, 'P2_Fy kN'] * critical_Mu_row['Factored']
        q_net_u = critical_Mu_row['q_net_u_max (kN/m2)']
        x1 = L/2 - C_C_Distance / 2.0
        x2 = L/2 + C_C_Distance / 2.0
        
        moment_data = []
        for x in x_L:
            M = (q_net_u * B * x**2) / 2.0
            if x >= x1:
                M -= Pu1 * (x - x1)
            if x >= x2:
                M -= Pu2 * (x - x2)
            moment_data.append(M)

        fig_moment = go.Figure()
        fig_moment.add_trace(go.Scatter(x=x_L, y=moment_data, mode='lines', 
                                        name='Bending Moment', fill='tozeroy'))
        fig_moment.add_vline(x=x1, line_dash="dot", line_color="red", annotation_text="P1")
        fig_moment.add_vline(x=x2, line_dash="dot", line_color="red", annotation_text="P2")
        fig_moment.update_layout(title_text=f"Bending Moment Diagram (Max Mu = {critical_Mu_row['Mu_max (kNm)']:.2f} kNm)",
                                 xaxis_title="Distance along L (m)", yaxis_title="Factored Moment $M_u$ (kNm)")
        st.plotly_chart(fig_moment)
        
        
        st.header("Detailed Design Results")
        st.dataframe(df_results)


if __name__ == "__main__":
    app()
