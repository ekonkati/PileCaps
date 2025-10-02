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
    
    # Pressure formula: q = Pu/A ± Mz*cy/Ix ± Mx*cx/Iy
    # Max/Min pressure points are at the corners (cx = L/2, cy = B/2)
    cx = L / 2
    cy = B / 2
    
    q_avg = Pu / A
    
    # Mz causes bending along the short axis (B), creating pressure variation along L
    # Mx causes bending along the long axis (L), creating pressure variation along B
    
    # Corner 1 (Max Pressure: + bending moment effects) -> x=+L/2, y=+B/2
    q1 = q_avg + (Mx * cx / I_y) + (Mz * cy / I_x) 
    
    # Corner 2 -> x=-L/2, y=+B/2
    q2 = q_avg - (Mx * cx / I_y) + (Mz * cy / I_x)
    
    # Corner 3 -> x=-L/2, y=-B/2
    q3 = q_avg - (Mx * cx / I_y) - (Mz * cy / I_x)
    
    # Corner 4 (Min Pressure) -> x=+L/2, y=-B/2
    q4 = q_avg + (Mx * cx / I_y) - (Mz * cy / I_x)
    
    return {
        'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
        'q_max': max(q1, q2, q3, q4),
        'q_min': min(q1, q2, q3, q4),
        'e_x': Mz / Pu,
        'e_y': Mx / Pu,
    }

def get_ast_required(Mu, fck, fy, d, B):
    """Calculates required steel area (Ast) per unit width (B) using IS 456 Limit State Method."""
    Mu_lim = 0.138 * fck * B * d**2 # For Fe415, Mu_lim is 0.138*fck*b*d^2 (using 0.138 for simplicity, though IS 456 specifies 0.133 for Fe415)
    
    if fy == 500:
        k = 0.133
    elif fy == 415:
        k = 0.138
    else: # Fe250
        k = 0.148
        
    Mu_lim = k * fck * B * d**2
        
    if Mu > Mu_lim:
        # Beam is over-reinforced or section depth is insufficient
        st.warning(f"Warning: Moment Mu ({Mu:.2f} kNm) exceeds Mu_limit ({Mu_lim/1e6:.2f} kNm) at column center. Increase depth (D).")
        return 0 
    
    # Convert Mu to N-mm
    Mu_Nmm = Mu * 1e6
    
    # Ast formula from IS 456:2000, Annex G
    Ast_req = (0.5 * fck / fy) * B * d * (1 - math.sqrt(1 - (4.6 * Mu_Nmm) / (fck * B * d**2)))
    
    # Check minimum steel requirement (0.12% of gross area for slabs/footings)
    Ast_min = 0.0012 * B * 1000 * d 
    
    return max(Ast_req, Ast_min) # Ast in mm^2/m

# --- 2. DATA PROCESSING AND APPLICATION LOGIC ---

def process_and_design(df_staad, L, B, D, fck, fy, sbc_gross, sbc_gross_seismic, C_C_Distance, w_pedestal, l_pedestal, gamma_conc, gamma_soil, D_f, Clear_Cover):
    """Main design logic that iterates through load cases."""
    
    df_results = pd.DataFrame()
    
    # 1. Calculate effective depth (d) and self-weight/surcharge
    d = D - Clear_Cover - 12 - 12/2 # Assuming 12mm bar and 12mm cover adjustment for simplicity
    d_m = d / 1000.0
    
    # Calculate soil and concrete weight
    footing_sw = L * B * D / 1000 * gamma_conc
    soil_surcharge = L * B * (D_f - D / 1000) * gamma_soil
    pedestal_sw_1 = w_pedestal[0] * l_pedestal[0] * (D_f / 1000 + 0.3) * gamma_conc # Simplified
    pedestal_sw_2 = w_pedestal[1] * l_pedestal[1] * (D_f / 1000 + 0.3) * gamma_conc # Simplified
    
    sw_surcharge = footing_sw + soil_surcharge + pedestal_sw_1 + pedestal_sw_2
    
    # 2. Iterate through load cases
    for index, row in df_staad.iterrows():
        lc_name = row['L/C']
        
        # Load factors (approximate from the file, but should be read from LC sheet)
        # Assuming 1.0 for unfactored design checks (SBC check) and 1.5/1.2 for factored checks
        load_factor = 1.0
        if 'EQ' in lc_name or 'WL' in lc_name:
            allowable_sbc = sbc_gross_seismic
            fact_f = 1.2
        else:
            allowable_sbc = sbc_gross
            fact_f = 1.5

        # --- UNFACTORED (SERVICE) LOAD FOR SBC CHECK ---
        
        # Extract forces and moments for the two columns (P1 and P2)
        P1_Fy = row['P1_Fy kN']
        P1_Mx = row['P1_Mx kNm']
        P1_Mz = row['P1_Mz kNm']
        
        P2_Fy = row['P2_Fy kN']
        P2_Mx = row['P2_Mx kNm']
        P2_Mz = row['P2_Mz kNm']
        
        # Total Vertical Load
        Pu_unfactored = P1_Fy + P2_Fy + sw_surcharge
        
        # Calculate moments about the center of the combined footing (CoG of footing)
        # CoG is at (L/2, B/2)
        # P1 is at (x1, B/2), P2 is at (x2, B/2)
        x1 = L/2 - C_C_Distance / 2.0
        x2 = L/2 + C_C_Distance / 2.0
        
        Mz_from_P1 = P1_Fy * (x1 - L/2)  # Should be zero if P1/P2 are centered on the length L
        Mz_from_P2 = P2_Fy * (x2 - L/2)  # Should be zero if P1/P2 are centered on the length L
        
        # Total moment about CoG for SBC Check
        Mx_total_unfactored = P1_Mx + P2_Mx
        Mz_total_unfactored = P1_Mz + P2_Mz + P1_Fy * (-C_C_Distance / 2.0) + P2_Fy * (C_C_Distance / 2.0)
        
        # Base Pressure Check
        q_pressures = calculate_base_pressure(Pu_unfactored, Mz_total_unfactored, Mx_total_unfactored, L, B)
        q_max = q_pressures['q_max']
        q_min = q_pressures['q_min']
        e_x = q_pressures['e_x']
        e_y = q_pressures['e_y']
        
        sbc_check = "SAFE" if q_max <= allowable_sbc and q_min >= 0 else "UNSAFE"
        ecc_check = "OK" if abs(e_x) <= L/6 and abs(e_y) <= B/6 else "FAIL"

        # --- FACTORED LOAD FOR STRUCTURAL DESIGN ---
        
        # Apply load factor (fact_f from above) to the column reactions only
        P1u_Fy = P1_Fy * fact_f
        P2u_Fy = P2_Fy * fact_f
        P1u_Mx = P1_Mx * fact_f
        P1u_Mz = P1_Mz * fact_f
        P2u_Mx = P2_Mx * fact_f
        P2u_Mz = P2_Mz * fact_f
        
        # Total factored load and moments (self-weight is typically not factored for net pressure)
        Pu_factored_col = P1u_Fy + P2u_Fy
        
        Mx_total_factored = P1u_Mx + P2u_Mx
        Mz_total_factored = P1u_Mz + P2u_Mz + P1u_Fy * (-C_C_Distance / 2.0) + P2u_Fy * (C_C_Distance / 2.0)
        
        # Net Upward Pressure (factored column loads minus self-weight/surcharge)
        q_net_u_avg = (Pu_factored_col - sw_surcharge) / (L * B) # Simplified net average
        
        # The WORK sheet uses a Max Factored Net Pr. value. Let's calculate the Net Upward pressure 
        # based on the factored loads applied to the footing.
        q_net_u_pressures = calculate_base_pressure(Pu_factored_col, Mz_total_factored, Mx_total_factored, L, B)
        q_net_u_max = q_net_u_pressures['q_max'] - (sw_surcharge / (L*B)) # Deduct average self-weight from gross factored pressure
        
        
        # Moment and Shear Calculation (Long Direction - Along L)
        # The critical moment section is at the face of the column.
        # Max Moment occurs at center of the footing (for hogging) or at the column face (for sagging)
        
        # Consider a strip of width B along L. The load is q_net_u (kN/m2) and two column loads.
        # For simplicity, calculate moment at the center of the footing (critical section for combined footing)
        
        # Moment at Centre (Max Hogging Moment)
        # This is a key step, simplified for the app. The exact calculation requires a beam-on-soil analysis.
        # For a rigid beam, the max hogging moment (Mu_hog) is typically between the columns.
        
        # Let Mu_hog be the simplified moment at the center (x=L/2)
        # R1, R2 are the factored column loads Pu1, Pu2
        # Mu = R1*(L/2 - x1) + R2*(L/2 - x2) - q_net_u_avg * B * (L/2)^2 / 2
        # Simplified assumption for Mu_max_hogging:
        Mu_hog = (P1u_Fy * C_C_Distance / 2) - (P2u_Fy * C_C_Distance / 2) # Moment caused by column loads on a centrally fixed beam
        
        
        # Max Sagging Moment
        # Occurs at the face of the column, treated as a cantilever.
        # Critical distance for cantilever: $x_{cant} = (L - C\_C\_Distance) / 2$
        x_cant = (L - C_C_Distance) / 2.0
        
        # Cantilever pressure: use the net upward pressure under the cantilever length
        q_cant = q_net_u_max # Use max pressure for conservative design
        
        # Factored Bending Moment at column face (max sagging)
        Mu_sag = (q_cant * B) * (x_cant**2 / 2) # M_u = (pressure * width) * (length^2 / 2)
        
        # Design for the larger moment (either hogging or sagging)
        Mu_max = max(abs(Mu_hog), abs(Mu_sag))
        
        # Required Steel Area (Longitudinal - Lengthwise)
        Ast_L = get_ast_required(Mu_max, fck, fy, d_m * 1000, B * 1000) # Ast in mm^2/m for width B
        
        # Punching Shear Check
        # Critical perimeter is d/2 from the face of the column.
        # Check around Column 2 (usually the most critical)
        
        col_w, col_l = w_pedestal[1]/1000, l_pedestal[1]/1000 # Pedestal dimensions in meters
        
        bo = 2 * ((col_w + d_m) + (col_l + d_m)) # Critical perimeter in meters
        
        # Max shear force for punching
        Vu_punch = P2u_Fy - (q_net_u_max * (col_w + d_m) * (col_l + d_m))
        
        # Nominal shear stress (tau_v)
        tau_v = (Vu_punch * 1000) / (bo * 1000 * d_m * 1000) # N/mm2
        
        # Allowable shear stress (tau_c')
        ks = 0.5 + B / L
        ks = min(ks, 1.0)
        
        tau_c_prime = 0.25 * math.sqrt(fck) # Max allowable as per IS 456
        tau_c_prime = min(tau_c_prime, 1.25) # Max allowable limit
        
        punch_check = "SAFE" if tau_v <= tau_c_prime else "UNSAFE"
        
        
        # Store results
        df_results.loc[index, 'L/C'] = lc_name
        df_results.loc[index, 'Factored'] = fact_f
        df_results.loc[index, 'Pu_unfactored (kN)'] = Pu_unfactored
        df_results.loc[index, 'Mx_unfactored (kNm)'] = Mx_total_unfactored
        df_results.loc[index, 'Mz_unfactored (kNm)'] = Mz_total_unfactored
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
    
    # Footing Dimensions (From WORK.csv: 2751 x 3000 mm)
    L = st.sidebar.number_input("Footing Length L (m)", value=3.0, min_value=1.0, step=0.1)
    B = st.sidebar.number_input("Footing Width B (m)", value=3.0, min_value=1.0, step=0.1)
    
    # Footing Thickness (From WORK.csv snippet: 500 mm)
    D_mm = st.sidebar.number_input("Footing Overall Depth D (mm)", value=500, min_value=300, step=50)
    
    # Clear cover (Standard IS code value)
    Clear_Cover = st.sidebar.number_input("Clear Cover (mm)", value=75, min_value=50, step=5)
    
    # Pedestal Data
    st.sidebar.subheader("Pedestal Data")
    C_C_Distance = st.sidebar.number_input("Column C-C Distance (m)", value=1.5, min_value=0.5, step=0.1)
    
    # Assuming two pedestals (P1, P2)
    w_pedestal_1 = st.sidebar.number_input("Pedestal 1 Width (mm)", value=600, min_value=300, step=50)
    l_pedestal_1 = st.sidebar.number_input("Pedestal 1 Length (mm)", value=400, min_value=300, step=50)
    w_pedestal_2 = st.sidebar.number_input("Pedestal 2 Width (mm)", value=600, min_value=400, step=50)
    l_pedestal_2 = st.sidebar.number_input("Pedestal 2 Length (mm)", value=400, min_value=400, step=50)
    
    # Material Properties (From DESIGN.csv)
    st.sidebar.subheader("Material Properties")
    fck = st.sidebar.number_input("Concrete Grade fck (N/mm²)", value=25, min_value=20, max_value=40)
    fy = st.sidebar.number_input("Steel Grade fy (N/mm²)", value=415, min_value=250, step=100)
    gamma_conc = st.sidebar.number_input("Unit weight of Concrete (kN/m³)", value=25.0, step=1.0)
    
    st.sidebar.subheader("Soil Properties")
    gamma_soil = st.sidebar.number_input("Unit Weight of Soil (kN/m³)", value=18.0, step=1.0)
    D_f = st.sidebar.number_input("Depth of foundation Df (m)", value=2.0, step=0.1)
    sbc_gross = st.sidebar.number_input("Gross SBC (DL+LL) (kN/m²)", value=286.0, step=10.0)
    sbc_gross_seismic = st.sidebar.number_input("Gross SBC (W/E) (kN/m²)", value=348.5, step=10.0)

    
    # --- Main Content: STAAD Input Upload (from 'staad input.csv' or 'Sheet1.csv') ---
    
    st.header("Upload STAAD Reactions 📥")
    uploaded_file = st.file_uploader("Upload STAAD Reactions CSV file (Expected columns: L/C, P1_Fy kN, P1_Mx kNm, P1_Mz kNm, P2_Fy kN, P2_Mx kNm, P2_Mz kNm, etc.)", type="csv")
    
    df_staad = None
    
    if uploaded_file is not None:
        try:
            # Read the file content, skipping initial header rows if needed (like the original file)
            # The structure is complex, so we'll try to find the start of the data
            uploaded_data = uploaded_file.getvalue().decode("utf-8")
            
            # Using StringIO to process the file in memory
            data_io = StringIO(uploaded_data)
            
            # Heuristic to find the start of the data (assuming L/C is the first column with values)
            lines = data_io.readlines()
            header_row_index = -1
            
            # Find the row that contains 'L/C' in a sensible position (Column 2 or 3)
            for i, line in enumerate(lines):
                if 'L/C' in line and any(col in line for col in ['Fx', 'Fy', 'Mx', 'Mz']):
                    header_row_index = i
                    break
            
            if header_row_index != -1:
                # Read the CSV starting from the identified header row
                data_io.seek(0)
                df_raw = pd.read_csv(data_io, skiprows=header_row_index, header=0, skipinitialspace=True)
                
                # Heuristic to clean and standardize column names (based on staad input.csv)
                # Keep relevant columns for P1 and P2
                
                # Check if the column names are from 'staad input.csv' structure (Multiple columns for P1, P2)
                cols_to_use = []
                for col in df_raw.columns:
                    if 'L/C' in col:
                        cols_to_use.append(col)
                    if 'P1' in col or 'P2' in col:
                        if 'Fy' in col or 'Mx' in col or 'Mz' in col:
                            cols_to_use.append(col)
                
                # Rename the column L/C to just 'L/C' and drop all NaN/unnamed rows/columns
                df_raw = df_raw.dropna(axis=1, how='all')
                
                # Simplified cleaning based on the structure of staad input.csv:
                # Need to manually map columns based on file structure
                # The structure is: P1 loads, then P2 loads
                
                # Let's use the explicit indices based on the staad input.csv file structure
                # L/C is col 2, P1_Fy is col 3, P1_Mx is col 6, P1_Mz is col 8
                # P2_Fy is col 12, P2_Mx is col 15, P2_Mz is col 17
                
                try:
                    df_staad = df_raw.iloc[:, [1, 3, 6, 8, 12, 15, 17]].copy()
                    df_staad.columns = ['L/C', 'P1_Fy kN', 'P1_Mx kNm', 'P1_Mz kNm', 'P2_Fy kN', 'P2_Mx kNm', 'P2_Mz kNm']
                    
                    # Convert to numeric, coercing errors to NaN
                    for col in df_staad.columns[1:]:
                        df_staad[col] = pd.to_numeric(df_staad[col], errors='coerce')
                        
                    df_staad = df_staad.dropna(subset=df_staad.columns[1:])
                    st.success("STAAD Reactions loaded successfully!")
                    st.dataframe(df_staad.head())
                    
                except Exception as e:
                    st.error(f"Error parsing STAAD data based on expected structure. Please ensure your CSV matches the `staad input.csv` format: {e}")
                    st.stop()
                    
            else:
                st.error("Could not find the header row containing 'L/C' and force/moment data. Please ensure the file format is correct.")
                st.stop()

        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
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
        
        # Filter for critical (UNSAFE) cases first
        critical_sbc = df_results[df_results['SBC_CHECK'] == 'UNSAFE']
        critical_ecc = df_results[df_results['ECC_CHECK'] == 'FAIL']
        critical_punch = df_results[df_results['PUNCH_CHECK'] == 'UNSAFE']
        
        if not critical_sbc.empty:
            st.error(f"**SBC CHECK FAILED** for {len(critical_sbc)} load case(s). Max Pressure: {critical_sbc['q_max (kN/m2)'].max():.2f} kN/m².")
            
        if not critical_ecc.empty:
            st.error(f"**ECCENTRICITY CHECK FAILED** for {len(critical_ecc)} load case(s).")
            
        if not critical_punch.empty:
            st.error(f"**PUNCHING SHEAR CHECK FAILED** for {len(critical_punch)} load case(s). Max $\\tau_v$: {critical_punch['Punch_tau_v (N/mm2)'].max():.2f} N/mm².")
            
        if critical_sbc.empty and critical_ecc.empty and critical_punch.empty:
            st.success("All checks (SBC, Eccentricity, Punching Shear) are SAFE for all load cases.")
        
        st.subheader("Structural Design Results (Based on Critical Moment)")
        
        # Find critical structural case (Max Mu and Max q_net_u)
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
        
        # Critical SBC case for pressure diagram
        q_plot_row = df_results.loc[df_results['q_max (kN/m2)'].idxmax()]
        
        # --- Sketch 1: Footing and Pedestals Plan View ---
        st.subheader("1. Footing Plan View")
        
        fig_plan = go.Figure()

        # Footing outline
        fig_plan.add_shape(type="rect", 
                           x0=0, y0=0, x1=L, y1=B,
                           line=dict(color="RoyalBlue", width=2), 
                           fillcolor="lightblue", opacity=0.5)

        # Pedestal 1 (Centered along B, distance C_C_Distance/2 from center)
        p1_center_x = L / 2 - C_C_Distance / 2.0
        p1_center_y = B / 2.0
        p1_x0 = p1_center_x - l_pedestal_1 / 2000.0
        p1_x1 = p1_center_x + l_pedestal_1 / 2000.0
        p1_y0 = p1_center_y - w_pedestal_1 / 2000.0
        p1_y1 = p1_center_y + w_pedestal_1 / 2000.0
        
        fig_plan.add_shape(type="rect", x0=p1_x0, y0=p1_y0, x1=p1_x1, y1=p1_y1, 
                           line=dict(color="Red", width=2), 
                           fillcolor="red", opacity=0.7)
        fig_plan.add_annotation(x=p1_center_x, y=p1_center_y, text="P1", showarrow=False, font=dict(color="white"))

        # Pedestal 2 (Centered along B, distance C_C_Distance/2 from center)
        p2_center_x = L / 2 + C_C_Distance / 2.0
        p2_center_y = B / 2.0
        p2_x0 = p2_center_x - l_pedestal_2 / 2000.0
        p2_x1 = p2_center_x + l_pedestal_2 / 2000.0
        p2_y0 = p2_center_y - w_pedestal_2 / 2000.0
        p2_y1 = p2_center_y + w_pedestal_2 / 2000.0

        fig_plan.add_shape(type="rect", x0=p2_x0, y0=p2_y0, x1=p2_x1, y1=p2_y1, 
                           line=dict(color="Red", width=2), 
                           fillcolor="red", opacity=0.7)
        fig_plan.add_annotation(x=p2_center_x, y=p2_center_y, text="P2", showarrow=False, font=dict(color="white"))

        fig_plan.update_layout(title_text=f"Footing Plan View ({L}m x {B}m)", 
                               xaxis_title="Length (L) [m]", yaxis_title="Width (B) [m]", 
                               showlegend=False, yaxis_scaleanchor="x", yaxis_scaleratio=1,
                               width=700, height=500)
        st.plotly_chart(fig_plan)

        # --- Sketch 2: Base Pressure Diagram (Along L) ---
        st.subheader("2. Unfactored Base Pressure Distribution (Critical SBC Case)")
        
        # Pressure calculation is linear for an assumed rigid footing
        x_coords = np.linspace(0, L, 100)
        
        # q_pressure is linear from q4 to q3 along the edge L=0 (y=-B/2) and q1 to q2 along the edge L (y=+B/2)
        # We'll plot the pressure along the centerline (y=B/2) for simplicity, where Mz/Ix = 0.
        
        # Calculate q_line at y=B/2 (from x=0 to x=L)
        # q(x) = Pu/A + Mx*cy/Ix + Mz*cx/Iy. We plot along B/2 where Mx is max.
        # But for base pressure, we are concerned about the variation along L-axis due to Mz.
        
        # To plot the pressure along the length L (x-axis), we use the eccentricity in the L direction (e_x or Mz/Pu).
        # q(x) = Pu/A ± Mz*(L/2 - x)*I_y/c_x^2
        
        # Simple trapezoidal distribution along L due to Mz
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

        # Create a simplified 1D plot of the bending moment (Mu) along the length L
        x_L = np.linspace(0, L, 100)
        
        # Factored column loads (simplified to critical case)
        Pu1 = df_staad.loc[critical_Mu_row.name, 'P1_Fy kN'] * critical_Mu_row['Factored']
        Pu2 = df_staad.loc[critical_Mu_row.name, 'P2_Fy kN'] * critical_Mu_row['Factored']
        q_net_u = critical_Mu_row['q_net_u_max (kN/m2)'] # Max net pressure
        
        moment_data = []
        for x in x_L:
            M = (q_net_u * B * x**2) / 2.0  # Moment from distributed load from the left end
            
            # Subtract moment due to P1 (if P1 is to the left of x)
            x1 = L/2 - C_C_Distance / 2.0
            if x >= x1:
                M -= Pu1 * (x - x1)
                
            # Subtract moment due to P2 (if P2 is to the left of x)
            x2 = L/2 + C_C_Distance / 2.0
            if x >= x2:
                M -= Pu2 * (x - x2)
            
            moment_data.append(M)

        fig_moment = go.Figure()
        fig_moment.add_trace(go.Scatter(x=x_L, y=moment_data, mode='lines', 
                                        name='Bending Moment', fill='tozeroy'))
        fig_moment.add_vline(x=L/2 - C_C_Distance / 2.0, line_dash="dot", line_color="red", annotation_text="P1")
        fig_moment.add_vline(x=L/2 + C_C_Distance / 2.0, line_dash="dot", line_color="red", annotation_text="P2")
        fig_moment.update_layout(title_text=f"Bending Moment Diagram (Max Mu = {critical_Mu_row['Mu_max (kNm)']:.2f} kNm)",
                                 xaxis_title="Distance along L (m)", yaxis_title="Factored Moment $M_u$ (kNm)")
        st.plotly_chart(fig_moment)
        
        
        st.header("Detailed Design Results")
        st.dataframe(df_results)
        st.info("Note: $M_x$ and $M_z$ moments were combined for $M_{total}$ and the design was simplified to the L-direction for demonstration purposes.")


if __name__ == "__main__":
    app()
