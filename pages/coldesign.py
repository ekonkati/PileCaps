import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# --- Configuration for Printability and Appearance ---
st.set_page_config(
    page_title="Printable Column Design Check & Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, printable look and feel
def apply_custom_css():
    st.markdown("""
        <style>
            /* Base font */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class*="st-emotion-"] {
                font-family: 'Inter', sans-serif;
            }

            /* Main Header & Title */
            h1 {
                color: #004d40; /* Dark Teal */
                border-bottom: 3px solid #004d40;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-weight: 700;
            }
            
            /* Print-specific styles: Hide non-essential UI elements for printing */
            @media print {
                /* Hide header, sidebar toggle, and buttons */
                .st-emotion-cache-18ni7ap, .st-emotion-cache-1v06a5k, 
                .st-emotion-cache-79elbk, .stButton, .st-emotion-cache-16cqjfa {
                    display: none !important; 
                }
                
                /* Ensure main content takes full width */
                .main .block-container {
                    padding-left: 0 !important;
                    padding-right: 0 !important;
                    max-width: 100% !important;
                }

                /* Ensure figures print without scroll/zoom controls */
                .js-plotly-plot .plotly .modebar {
                    display: none !important;
                }

                /* Force page break before Input Summary for clean printing */
                .print-page-break {
                    page-break-before: always;
                }

                /* Show a print title */
                @page { size: A4 portrait; margin: 1cm; }
            }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css()

# --- Plotly Visualization Functions ---

def draw_column_plan(b, h, C, D_bar, N_b, N_h):
    """Draws the column cross-section with reinforcement in plan view."""
    fig = go.Figure()

    # 1. Concrete outline (Rectangle)
    fig.add_shape(
        type="rect",
        x0=-b/2, y0=-h/2, x1=b/2, y1=h/2,
        line=dict(color="#606060", width=2),
        fillcolor="#ADD8E6", # Light blue for concrete
        name="Concrete Section"
    )

    # 2. Rebar Placement (Simplified based on N_b and N_h)
    bar_radius = D_bar / 2
    
    # Calculate effective dimension for bar centers
    b_eff = b - 2 * C
    h_eff = h - 2 * C

    # Generate bar coordinates
    bar_coords = []

    # Bars along B-faces (parallel to X-axis)
    y_coords = [-h_eff / 2, h_eff / 2]
    for y in y_coords:
        # Include end bars only once (at corners)
        if N_b > 1:
            x_step = b_eff / (N_b - 1)
            for i in range(N_b):
                x = -b_eff / 2 + i * x_step
                bar_coords.append((x, y))
        else: # Handle case with 1 bar (center)
            # This case is redundant since N_b minimum is 2 in the UI, but kept for robustness
            if N_h == 1: 
                 bar_coords.append((-b_eff / 2, y))
                 bar_coords.append((b_eff / 2, y))

    # Bars along H-faces (parallel to Y-axis) - Exclude corners already placed by B-faces
    x_coords = [-b_eff / 2, b_eff / 2]
    for x in x_coords:
        if N_h > 1:
            y_step = h_eff / (N_h - 1)
            for j in range(1, N_h - 1): # Start at 1, end at N_h-2 to skip corners
                y = -h_eff / 2 + j * y_step
                bar_coords.append((x, y))

    # Remove duplicate points (corners are calculated in both loops)
    unique_bar_coords = list(set(bar_coords))
    
    # Plot Bars
    if unique_bar_coords:
        bar_x = [c[0] for c in unique_bar_coords]
        bar_y = [c[1] for c in unique_bar_coords]
        
        fig.add_trace(go.Scatter(
            x=bar_x, y=bar_y,
            mode='markers',
            marker=dict(
                size=D_bar * 1.5, # Exaggerate size slightly for visibility
                color='red',
                line=dict(width=1, color='DarkRed')
            ),
            name=f'{len(unique_bar_coords)} No. $\\phi${D_bar} Bars',
            hovertext=[f'Bar at ({x:.0f}, {y:.0f})' for x, y in unique_bar_coords]
        ))

    # Layout Configuration
    fig.update_layout(
        title=f'Column Cross Section ({b}x{h} mm) - Plan View',
        xaxis_title="Width (mm)",
        yaxis_title="Depth (mm)",
        autosize=True,
        showlegend=True,
        width=500, height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(scaleanchor="x", scaleratio=1), # Maintain aspect ratio
        plot_bgcolor='white'
    )
    return fig

def draw_column_elevation(b, h, L, D_bar):
    """Draws a simplified column elevation with bars."""
    fig = go.Figure()
    
    # Concrete Elevation (Rectangular prism face)
    fig.add_shape(
        type="rect",
        x0=-b/2, y0=0, x1=b/2, y1=L,
        line=dict(color="#606060", width=1),
        fillcolor="#C0C0C0", # Grey for elevation
        name="Column Elevation"
    )

    # Reinforcement (two visible bars)
    bar_offset = (b / 2) - 50 # Arbitrary offset from edge
    
    # Left Bar
    fig.add_trace(go.Scatter(
        x=[bar_offset] * 2, y=[0, L],
        mode='lines',
        line=dict(color='red', width=D_bar/4),
        name=f"Rebar $\\phi${D_bar}"
    ))

    # Right Bar
    fig.add_trace(go.Scatter(
        x=[-bar_offset] * 2, y=[0, L],
        mode='lines',
        line=dict(color='red', width=D_bar/4),
        showlegend=False
    ))

    # Layout Configuration
    fig.update_layout(
        title=f'Column Elevation (L={L} mm)',
        xaxis_title="Width (mm)",
        yaxis_title="Height (mm)",
        autosize=True,
        width=300, height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False),
    )
    return fig

# --- Placeholder Calculation Logic ---

def calculate_column_design(b, h, L, fc, fy, Pu, Mux, Muy, As_total, C):
    """
    *** IMPORTANT: PLACEHOLDER FUNCTION ***
    
    You must replace the logic below with the actual engineering calculations 
    from your COLDESIGN.xlsm file (e.g., ACI 318, Eurocode, etc.)
    
    This function currently provides a SIMULATED result and intermediate steps.
    """
    
    # 1. Properties Calculation
    Ag = b * h
    rho_g = As_total / Ag
    Ec = 4700 * np.sqrt(fc) # ACI approx for Ec in MPa
    
    # 2. Simulate Nominal Capacity (Pn0 - pure axial capacity)
    Pn0_nominal = 0.85 * fc * (Ag - As_total) + fy * As_total # N
    
    # 3. Apply Column Limits (ACI 318 Example)
    alpha = 0.80 # Tied column
    Pn_max = alpha * Pn0_nominal
    
    # 4. Apply Reduction Factor (phi) - Example ACI
    phi = 0.65
    Phi_Pn_max = phi * Pn_max / 1000 # Convert N to kN

    # 5. Check Capacity (Simplified Biaxial Interaction Ratio)
    
    # Required resultant moment
    M_req = np.sqrt(Mux**2 + Muy**2) # Resultant Moment (N-mm)
    
    # Dummy Mn value for interaction ratio calculation (Replace with actual Mn)
    # This is a highly simplified proxy for the actual interaction surface check.
    Phi_Mn_proxy = 0.15 * Phi_Pn_max * 1000 * (min(b, h) - C) # N-mm (Simplified)

    if Phi_Mn_proxy == 0:
        interaction_ratio = Pu / (Phi_Pn_max * 1000)
    else:
        # P-M Interaction using Bresler's formula simplified for demonstration
        interaction_ratio = (Pu / (Phi_Pn_max * 1000)) + (M_req / Phi_Mn_proxy)
        
    # --- Simulated As Required (Required Steel Area) ---
    # Placeholder: Assuming required axial capacity is 90% of Pu + safety factor
    # This calculation MUST be replaced by your code's real P-M analysis output
    As_required = (Pu * 1.15) / (0.8 * fy) # Rough estimate in mm^2

    # Simple pass/fail based on interaction ratio
    status = "Pass" if interaction_ratio < 1.0 else "Fail"
    
    return {
        "Ag": Ag, "rho_g": rho_g, "Ec": Ec, "Pn0_nominal": Pn0_nominal,
        "Pn_max": Pn_max, "Phi_Pn_max": Phi_Pn_max, 
        "Interaction_Ratio": interaction_ratio,
        "Status": status,
        "Phi_Mn_proxy": Phi_Mn_proxy,
        "As_required": As_required, # Added As_required
    }


# --- Streamlit UI Layout ---

st.title("Structural Column Design Check & Visualization")
st.markdown("---")

# Instructions and User ID
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Enter all parameters below. Note the core formulas are placeholders."
    "\n2. Click 'Run Design Check' to see results and drawings."
    "\n3. **To print:** Use your browser's print function (Ctrl+P or Command+P)."
    " The page is optimized for a clean, single-page printable output."
)

# --- INPUT SECTION ---
col_geom, col_mat_load = st.columns([1.2, 2])

with col_geom:
    st.subheader("1. Geometric & Reinforcement Details")

    # Column Dimensions
    st.markdown("#### Section Geometry (mm)")
    dim_col1, dim_col2 = st.columns(2)
    with dim_col1:
        b = st.number_input("Width, b (mm)", min_value=100, value=400, step=10)
        L = st.number_input("Length, L (mm)", min_value=1000, value=3500, step=100)
    with dim_col2:
        h = st.number_input("Depth, h (mm)", min_value=100, value=600, step=10)
        C = st.number_input("Clear Cover, C (mm)", min_value=20, value=40, step=5)

    # Reinforcement Arrangement (UPDATED for standard selection)
    st.markdown("#### Reinforcement Arrangement")
    rebar_col1, rebar_col2 = st.columns(2)
    
    # Standard Bar Diameters (mm)
    standard_bar_diameters = [12, 16, 20, 25, 32]

    with rebar_col1:
        # User selects the bar diameter
        D_bar = st.selectbox("Bar Diameter, $\phi$ (mm)", standard_bar_diameters, index=3) # Default to 25mm
        N_b = st.number_input("Bars along B-face (Top/Bottom)", min_value=2, value=3, step=1)
    with rebar_col2:
        N_h = st.number_input("Bars along H-face (Left/Right)", min_value=2, value=4, step=1)
        
    # Calculate Total Steel Area (As_total = As_provided)
    N_total = 2 * N_b + 2 * (N_h - 2)
    As_bar = math.pi * (D_bar / 2)**2
    As_total = N_total * As_bar
    st.info(f"Provided $A_s$ ($A_{{s,prov}}$): **{N_total} $\\times$ $\\phi${D_bar}** bars = **{As_total:.0f} mm²**")


with col_mat_load:
    st.subheader("2. Material Properties & Loads")

    # Material Properties
    st.markdown("#### Material Properties")
    mat_col1, mat_col2 = st.columns(2)
    with mat_col1:
        fc = st.number_input("Concrete Strength, f'c (MPa)", min_value=15.0, value=30.0, step=1.0)
    with mat_col2: # This was incorrectly set to col_mat_load in the last version
        fy = st.number_input("Steel Yield Strength, fy (MPa)", min_value=300.0, value=420.0, step=10.0)

    # Loads (Factored)
    st.markdown("#### Factored Design Loads (Pu, Mu)")
    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        Pu_kN = st.number_input("Axial Load, Pu (kN)", min_value=0.0, value=1500.0, step=100.0)
        Pu = Pu_kN * 1000 # Convert to N
    with load_col2:
        Mux_kNm = st.number_input("Moment Mu,x (kNm)", min_value=0.0, value=50.0, step=5.0)
        Mux = Mux_kNm * 1000000 # Convert to N-mm
    with load_col3:
        Muy_kNm = st.number_input("Moment Mu,y (kNm)", min_value=0.0, value=30.0, step=5.0)
        Muy = Muy_kNm * 1000000 # Convert to N-mm


st.markdown("---")

# --- CALCULATION TRIGGER & RESULTS ---
if st.button("Run Design Check & Visualization", type="primary"):
    
    if b <= 0 or h <= 0 or L <= 0:
        st.error("Please provide valid, non-zero dimensions for the column section and length.")
    elif N_b < 2 or N_h < 2:
        st.error("Minimum of 2 bars required on opposite faces for a rectangular column.")
    else:
        # Run the placeholder calculation
        results = calculate_column_design(
            b, h, L, fc, fy, Pu, Mux, Muy, As_total, C
        )
        
        st.header("3. Detailed Design Procedure & Results")

        # Extract values for cleaner Markdown formatting
        Phi_Pn_max_kN = results['Phi_Pn_max']
        Pn_max_kN = results['Pn_max']/1000
        Pn0_nominal_kN = results['Pn0_nominal']/1000
        
        M_req_kNm = np.sqrt(Mux**2 + Muy**2)/1000000
        Phi_Mn_proxy_kNm = results['Phi_Mn_proxy']/1000000
        
        # Design Procedure Markdown
        st.markdown(f"""
        ### Design Procedure (Simulated Steps)

        1.  **Section Properties:**
            * Gross Area ($A_g$): ${b} \times {h} = {results['Ag']:,} \text{{ mm}}^2$
            * Steel Ratio ($\rho_g$): $A_s / A_g = {As_total:,.0f} / {results['Ag']:,} = {results['rho_g'] * 100:.2f}\%$
            * Concrete Modulus of Elasticity ($E_c$): $4700 \sqrt{{f'_c}} = {results['Ec']:.0f} \text{{ MPa}}$
            
        2.  **Pure Axial Capacity ($\boldsymbol{{P_{n0}}}$):** *(Replace with exact code formula)*
            $$P_{n0} = 0.85 f'_c (A_g - A_s) + f_y A_s$$
            $$P_{n0} = 0.85({{fc}})({{results['Ag']:,}} - {{As_total:,.0f}}) + {{fy}}({{As_total:,.0f}}) = {Pn0_nominal_kN:,.0f} \text{{ kN}}$$

        3.  **Maximum Factored Axial Capacity ($\boldsymbol{{\phi P_{n,max}}}$):** *(Replace with exact code formula)*
            * Reduction Factor $\phi = 0.65$ (Tied Column)
            * Max Nominal Capacity $\alpha P_{n0} = 0.80 P_{n0} = {Pn_max_kN:,.0f} \text{{ kN}}$
            $$\phi P_{n,max} = 0.65 \times {Pn_max_kN:,.0f} \text{{ kN}} = \mathbf{{Phi_Pn_max_kN:.2f} \text{{ kN}}}$$
            * **Check 1 (Axial):** Required $P_u = {Pu_kN:,.0f} \text{{ kN}}$. Provided $\phi P_{n,max} = {Phi_Pn_max_kN:.2f} \text{{ kN}}$.

        4.  **Biaxial Bending Interaction Check:** *(Placeholder using Simplified Summation)*
            * Required Resultant Moment $M_u = \sqrt{{M_{ux}^2 + M_{uy}^2}} \approx {M_req_kNm:,.1f} \text{{ kNm}}$
            * Simulated Moment Capacity $\phi M_{n} \approx {Phi_Mn_proxy_kNm:,.1f} \text{{ kNm}}$
            * **Interaction Ratio (I.R.):** *(The final calculation must come from your P-M diagram/surface analysis)*
            $$ \text{{I.R.}} \approx \frac{{P_u}}{{\phi P_n}} + \frac{{M_u}}{{\phi M_n}} = \mathbf{{results['Interaction_Ratio']:.3f}} $$
            
        ---
        ## FINAL DESIGN OUTPUT & SAFETY CHECK:
        """, unsafe_allow_html=True)
        
        # Display Final Status
        if results['Status'] == "Pass":
            st.success(f"✅ CAPACITY CHECK PASS: Interaction Ratio = {results['Interaction_Ratio']:.3f} (Required < 1.0)")
        else:
            st.error(f"❌ CAPACITY CHECK FAIL: Interaction Ratio = {results['Interaction_Ratio']:.3f} (Required < 1.0). Increase section or reinforcement.")
        
        # --- As Required vs As Provided Comparison ---
        As_required = results['As_required']
        As_provided = As_total
        
        st.markdown(f"""
        ### Longitudinal Reinforcement Check ($A_s$)
        
        * **Required Steel Area ($A_{{s,req}}$):** (Simulated) **{As_required:,.0f} $\text{{ mm}}^2$**
        * **Provided Steel Area ($A_{{s,prov}}$):** **{As_provided:,.0f} $\text{{ mm}}^2$**
        """, unsafe_allow_html=True)

        if As_provided >= As_required:
            st.success(f"✅ REINFORCEMENT PASS: Provided $A_{{s,prov}} = {As_provided:,.0f} \text{{ mm}}^2$ $\ge$ Required $A_{{s,req}} = {As_required:,.0f} \text{{ mm}}^2$.")
        else:
            st.warning(f"⚠️ REINFORCEMENT WARNING: Provided $A_{{s,prov}} = {As_provided:,.0f} \text{{ mm}}^2$ $< $ Required $A_{{s,req}} = {As_required:,.0f} \text{{ mm}}^2$. Adjust bar arrangement.")
        
        
        # --- Visualization Section ---
        st.subheader("4. Column Visualization")
        
        vis_col1, vis_col2 = st.columns(2)
        
        # Plan View
        with vis_col1:
            plan_fig = draw_column_plan(b, h, C, D_bar, N_b, N_h)
            st.plotly_chart(plan_fig, use_container_width=True)
        
        # Elevation View
        with vis_col2:
            elevation_fig = draw_column_elevation(b, h, L, D_bar)
            st.plotly_chart(elevation_fig, use_container_width=True)

        
        # Detailed Input Summary for Print
        st.markdown("""
        <div class="print-page-break"> 
            <h3 style="margin-top: 40px; color: #4b4b4b;">Input Summary for Record</h3>
        </div>
        """, unsafe_allow_html=True)

        data = {
            "Parameter": ["Concrete Strength, f'c", "Steel Yield, fy", "Width, b", "Depth, h", "Length, L", "Clear Cover, C", "Bar Diameter, $\phi$", "Total Bars", "Provided $A_s$", "Axial Load, $P_u$", "Moment $M_{u,x}$", "Moment $M_{u,y}$"],
            "Value": [
                f"{fc} MPa", f"{fy} MPa", f"{b} mm", f"{h} mm", f"{L} mm", f"{C} mm", f"{D_bar} mm", 
                f"{N_total}", f"{As_total:,.0f} mm²", f"{Pu_kN:,.0f} kN", f"{Mux_kNm:,.0f} kNm", f"{Muy_kNm:,.0f} kNm"
            ],
        }
        
        st.table(data)

# Instructions to print are generally found in the top right of the browser menu.
st.sidebar.markdown("---")
st.sidebar.markdown("**Print Tip:** Use `Ctrl+P` (or `Cmd+P`) to print the results view.")
