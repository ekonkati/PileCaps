import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd # Used for easily structuring Plotly data

# --- Constants and Setup ---
# ACI 318-19 Constants (simplified)
PHI_SHEAR = 0.75  # Strength reduction factor for shear
PHI_FLEXURE = 0.90 # Strength reduction factor for flexure

def calculate_pile_reactions(P, Mx, My, Nx, Ny, Sx, Sy):
    """
    Calculates the vertical reaction force for each pile in a rigid pile cap 
    using the elastic method.
    $R_i = P/N \pm (M_y \cdot x_i / \sum x^2) \pm (M_x \cdot y_i / \sum y^2)$

    Args:
        P (float): Total vertical load (kN).
        Mx (float): Moment about the X-axis (kN-m).
        My (float): Moment about the Y-axis (kN-m).
        Nx (int): Number of piles in the X-direction.
        Ny (int): Number of piles in the Y-direction.
        Sx (float): Spacing between piles in the X-direction (m).
        Sy (float): Spacing between piles in the Y-direction (m).

    Returns:
        tuple: (list of (x, y, R_i) tuples, max_R, sum_x2, sum_y2)
    """
    N = Nx * Ny
    if N == 0:
        return [], 0, 0, 0

    pile_coords = []
    
    # Calculate coordinate offsets from centroid
    # X-coordinates range from $-(N_x-1)/2 \cdot S_x$ to $(N_x-1)/2 \cdot S_x$
    x_offset = (Nx - 1) * Sx / 2.0
    for i in range(Nx):
        x = i * Sx - x_offset
        
        # Y-coordinates range from $-(N_y-1)/2 \cdot S_y$ to $(N_y-1)/2 \cdot S_y$
        y_offset = (Ny - 1) * Sy / 2.0
        for j in range(Ny):
            y = j * Sy - y_offset
            pile_coords.append((x, y))

    # Calculate sum of squares ($\sum x^2$ and $\sum y^2$)
    x_coords = np.array([coord[0] for coord in pile_coords])
    y_coords = np.array([coord[1] for coord in pile_coords])
    
    sum_x2 = np.sum(x_coords ** 2)
    sum_y2 = np.sum(y_coords ** 2)

    reactions = []
    max_R = 0.0

    # Calculate individual pile reactions
    for x, y in pile_coords:
        R_vertical = P / N
        
        # Moment My causes stress in the x-direction ($M \cdot x / \sum x^2$)
        if sum_x2 != 0:
            R_vertical += (My * x) / sum_x2
        
        # Moment Mx causes stress in the y-direction ($M \cdot y / \sum y^2$)
        if sum_y2 != 0:
            R_vertical += (Mx * y) / sum_y2

        reactions.append((x, y, R_vertical))
        
        # Determine maximum reaction force (for shear check)
        max_R = max(max_R, abs(R_vertical))
        
    return reactions, max_R, sum_x2, sum_y2

def check_punching_shear(R_max, H_cap, D_pile, fc_prime, LF=1.5):
    """
    Performs the two-way (punching) shear check for the critical pile cap depth.
    ACI 318-19 simplified check.
    """
    
    # Estimate effective depth 'd' (d = H_cap - cover - rebar/2). 
    # Use a conservative estimate of 150mm (0.15m) effective cover for large caps.
    d_eff = H_cap - 0.15 
    
    if d_eff <= 0:
        # Prevent division by zero or negative dimensions
        return 0, 0, d_eff, "Depth too small for effective cover.", 0
        
    # Critical section for two-way shear is at $d_{eff}/2$ from the pile face
    b_o_perimeter = np.pi * (D_pile + d_eff) # Circumference of a circle with radius $(D_{pile}/2 + d_{eff}/2)$
    
    # 1. Ultimate Shear Force ($V_u$)
    V_u_kN = R_max * LF
    V_u_N = V_u_kN * 1000 # Convert to Newtons
    
    # 2. Nominal Concrete Shear Capacity ($V_c$)
    # ACI 318-19 Eq. 22.6.5.2: $V_c = 0.17 \cdot \sqrt{f'_c} \cdot b_o \cdot d$ ($f'_c$ in MPa, $b_o, d$ in mm, $V_c$ in N)
    d_eff_mm = d_eff * 1000 
    b_o_mm = b_o_perimeter * 1000
    
    # $V_c$ is in Newtons
    V_c_N = 0.17 * np.sqrt(fc_prime) * b_o_mm * d_eff_mm
    
    # 3. Allowable Shear Capacity
    V_c_allowable_N = PHI_SHEAR * V_c_N
    
    check_status = "O.K." if V_u_N <= V_c_allowable_N else "NOT O.K. (Depth too small)"
    
    return V_u_N/1000, V_c_allowable_N/1000, d_eff, check_status, b_o_perimeter

def calculate_As_req(Mu, b, d, fc_prime, fy):
    """
    Calculates the required steel area $A_s$ (mm^2) for a rectangular section 
    given $M_u$ (kN-m), b (mm), d (mm), $f'_c$ (MPa), and $f_y$ (MPa).
    Uses the ACI simplified quadratic formula method.
    """
    Mu_Nmm = Mu * 1e6 # Convert kN-m to N-mm
    
    if Mu_Nmm <= 0 or d <= 0:
        return 0.0

    phi = PHI_FLEXURE
    
    # $f'_c$ and $f_y$ are in N/mm^2 (MPa)
    fc_prime_Mpa = fc_prime 
    fy_Mpa = fy 

    # 1. Calculate Required Resistance Factor $R_n$: $R_n = M_u / (\phi \cdot b \cdot d^2)$ in N/mm^2 (MPa)
    if b * d**2 == 0:
        return 99999.0
        
    Rn = (Mu_Nmm / phi) / (b * d**2)
    
    # Beta1 (for $f'_c \le 28$ MPa, $\beta_1=0.85$, decreases by 0.05 for every 7MPa increase over 28)
    if fc_prime_Mpa <= 28:
        beta1 = 0.85
    else:
        beta1 = max(0.85 - 0.05 * (fc_prime_Mpa - 28) / 7.0, 0.65)
        
    # Solving for reinforcement ratio $\rho$ using the quadratic formula for ACI $R_n$:
    # $\rho = (0.85 \cdot f'_c / f_y) \cdot (1 - \sqrt{1 - 2 \cdot R_n / (0.85 \cdot f'_c)})$
    
    term_under_sqrt = 1 - (2 * Rn) / (0.85 * fc_prime_Mpa)
    
    if term_under_sqrt < 0:
        # This means the required $R_n$ is too high, indicating insufficient depth/capacity
        return 99999.0 
        
    rho = (0.85 * fc_prime_Mpa / fy_Mpa) * (1 - np.sqrt(term_under_sqrt))
    
    # Required Steel Area ($A_s$) in $mm^2$
    As_req_mm2 = rho * b * d
    
    return As_req_mm2

def check_flexural_reinforcement(reactions, d_eff, Lx, Ly, fc_prime, fy, load_factor):
    """
    Calculates the required steel area for flexure in the pile cap (simplified).
    Moment is calculated by summing ultimate pile forces $\times$ distance to centroid (critical section).
    """
    
    if d_eff <= 0:
        return {}, "Effective depth is zero or negative."

    results = {}
    
    # 1. Calculate Design Moments ($M_u$)
    Mu_x_list = [] # Moment about X-axis (Resisted by steel parallel to Y)
    Mu_y_list = [] # Moment about Y-axis (Resisted by steel parallel to X)
    
    for x, y, R_i in reactions:
        # Use the signed reaction R_i for moment calculation, then take the absolute sum
        R_i_u = R_i * load_factor
        
        # Moment about X-axis (using y-distance): sum R*y for y > 0
        if y >= 0:
            Mu_x_list.append(R_i_u * y)
        
        # Moment about Y-axis (using x-distance): sum R*x for x > 0
        if x >= 0:
            Mu_y_list.append(R_i_u * x)

    # Max moment is the largest absolute value (tension or compression side)
    Mu_x = abs(sum(Mu_x_list)) # Total design moment in Y direction (about X-axis)
    Mu_y = abs(sum(Mu_y_list)) # Total design moment in X direction (about Y-axis)
    
    # d and b are converted to mm for the helper function
    d = d_eff * 1000 
    
    # Direction X Reinforcement (Resists $M_{u,y}$ over cap width $L_y$)
    b_x = Ly * 1000 # Cap width in mm ($L_y$)
    As_x_req = calculate_As_req(Mu_y, b_x, d, fc_prime, fy)
    results['Mu_y'] = Mu_y
    results['As_x_req'] = As_x_req

    # Direction Y Reinforcement (Resists $M_{u,x}$ over cap width $L_x$)
    b_y = Lx * 1000 # Cap width in mm ($L_x$)
    As_y_req = calculate_As_req(Mu_x, b_y, d, fc_prime, fy)
    results['Mu_x'] = Mu_x
    results['As_y_req'] = As_y_req

    # Check for minimum reinforcement requirement (ACI 318-19 9.6.1.1)
    # $\rho_{min} = \max(\frac{0.25 \cdot \sqrt{f'_c}}{f_y}, \frac{1.4}{f_y})$
    rho_min_factor = max(0.25 * np.sqrt(fc_prime), 1.4)
    rho_min = rho_min_factor / fy
    
    As_min_x = rho_min * b_x * d 
    As_min_y = rho_min * b_y * d 

    results['As_x_req'] = max(As_x_req, As_min_x)
    results['As_y_req'] = max(As_y_req, As_min_y)

    return results, "O.K."

# Create a visual representation of the pile group and the critical pile using Plotly
def draw_piles_plotly(reactions, Nx, Ny, Sx, Sy, Dp, R_max, H_cap):
    
    # Prepare data for plotting
    data = []
    max_abs_R = max([abs(r[2]) for r in reactions]) if reactions else 1.0

    # Calculate effective depth and critical pile
    d_eff = H_cap - 0.15 
    critical_pile_index = np.argmax([abs(r[2]) for r in reactions]) if reactions else -1
    
    for i, (x, y, R_i) in enumerate(reactions):
        is_critical = "⭐" if i == critical_pile_index else ""
        
        # Scale marker size based on absolute reaction (max size 30, min size 10)
        size = 10 + (abs(R_i) / max_abs_R) * 20 if max_abs_R > 0 else 15
        
        # Determine color (Red for compression, Blue for tension/uplift)
        color = 'red' if R_i >= 0 else 'blue'
        
        data.append({
            'x': x,
            'y': y,
            'R_i': R_i,
            'label': f"Pile {i+1}<br>R = {R_i:,.0f} kN {is_critical}",
            'color': color,
            'size': size
        })
        
    df = pd.DataFrame(data)

    fig = go.Figure()
    
    # Draw Piles (Scatter plot)
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        marker=dict(
            color=df['color'],
            size=df['size'],
            sizemode='diameter',
            opacity=0.8,
            line=dict(width=1, color='black')
        ),
        text=[f"{r:,.0f} kN" for r in df['R_i']],
        textposition="middle center",
        hoverinfo='text',
        hovertext=df['label'],
        name='Piles'
    ))

    # Draw Punching Shear Perimeter
    if critical_pile_index != -1 and d_eff > 0:
        crit_x = df.iloc[critical_pile_index]['x']
        crit_y = df.iloc[critical_pile_index]['y']
        punching_radius = Dp/2 + d_eff/2
        
        # Draw the critical section (Punching Perimeter)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=crit_x - punching_radius, y0=crit_y - punching_radius,
            x1=crit_x + punching_radius, y1=crit_y + punching_radius,
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color="green", width=2, dash="dash"),
            name='Punching Perimeter'
        )
        
        # Add annotation for Punching Section
        fig.add_annotation(
            x=crit_x + punching_radius, y=crit_y + punching_radius * 1.2, 
            text="Punching Section $b_o$", 
            showarrow=False, 
            font=dict(color="green", size=10)
        )


    # Calculate centroid offset
    x_offset_center = (Nx - 1) * Sx / 2.0
    y_offset_center = (Ny - 1) * Sy / 2.0
    
    # Draw Cap boundary (conceptual - 0.5m edge distance)
    Lx_cap = (Nx - 1) * Sx + 2 * 0.5 
    Ly_cap = (Ny - 1) * Sy + 2 * 0.5
    
    # Calculate corners of the cap for drawing the boundary rectangle
    x_min = -x_offset_center - 0.5
    y_min = -y_offset_center - 0.5
    
    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=x_min, y0=y_min,
        x1=x_min + Lx_cap, y1=y_min + Ly_cap,
        line=dict(color="gray", width=2, dash="dash"),
        fillcolor='rgba(0, 0, 0, 0)',
        name='Cap Boundary'
    )
    
    # Configure Layout
    fig.update_layout(
        title='Pile Group Plan View (Size and Color by Reaction)',
        xaxis_title="X-direction (m)",
        yaxis_title="Y-direction (m)",
        xaxis_range=[x_min - 0.2, x_min + Lx_cap + 0.2],
        yaxis_range=[y_min - 0.2, y_min + Ly_cap + 0.2],
        xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
        yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=False,
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Pile Group & Cap Design (Metric)")
st.title("🏗️ Pile Group Design (Elastic Method)")
st.caption("Calculates maximum pile reaction, performs two-way shear check, and determines flexural steel requirements based on ACI 318-19 (Metric Units: kN, m, MPa).")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Applied Loads (Net of Cap Weight)")
    P_net = st.number_input("Total Vertical Load $P_{net}$ (kN)", value=5000.0, min_value=0.0)
    Mx = st.number_input("Moment about X-axis $M_x$ (kN-m)", value=500.0, step=100.0)
    My = st.number_input("Moment about Y-axis $M_y$ (kN-m)", value=500.0, step=100.0)
    
    st.header("3. Concrete & Reinforcement")
    fc_prime = st.number_input("Concrete Strength $f'_c$ (MPa)", value=30.0, min_value=15.0, max_value=60.0, step=1.0)
    fy = st.number_input("Steel Yield Strength $f_y$ (MPa)", value=420.0, min_value=300.0, max_value=500.0, step=10.0)
    load_factor = st.number_input("Load Factor for $V_u$ and $M_u$ (e.g., 1.5)", value=1.5, min_value=1.0, step=0.1)


with col2:
    st.header("2. Pile Group Geometry")
    Nx = st.number_input("Piles in X-direction ($N_x$)", value=3, min_value=1, step=1)
    Ny = st.number_input("Piles in Y-direction ($N_y$)", value=3, min_value=1, step=1)
    Sx = st.number_input("Pile Spacing $S_x$ (m)", value=1.5, min_value=0.5, step=0.1)
    Sy = st.number_input("Pile Spacing $S_y$ (m)", value=1.5, min_value=0.5, step=0.1)
    Dp = st.number_input("Pile Diameter $D_p$ (m)", value=0.6, min_value=0.3, step=0.05)
    
    st.header("4. Pile Cap Thickness Check")
    H_cap = st.number_input("Pile Cap Total Depth $H$ (m)", value=1.2, min_value=0.5, step=0.1)


# --- Calculation and Output ---

st.divider()

# 1. Pile Group Analysis
reactions, R_max, sum_x2, sum_y2 = calculate_pile_reactions(P_net, Mx, My, Nx, Ny, Sx, Sy)

st.header("Calculation Results")

# --- First Column: Pile Reactions ---
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("Pile Group Analysis (Elastic Method)")
    
    # Display Group Properties
    st.markdown(f"**Total Piles ($N$):** ${Nx} \\times {Ny} = {Nx * Ny}$")
    st.markdown(f"**$\sum x^2$ (Pile Group Stiffness):** ${sum_x2:,.2f} \: \mathrm{{m^2}}$")
    st.markdown(f"**$\sum y^2$ (Pile Group Stiffness):** ${sum_y2:,.2f} \: \mathrm{{m^2}}$")
    
    # Display Max Reaction
    if R_max > 0:
        st.metric(
            label="Maximum Vertical Pile Reaction $R_{max}$",
            value=f"{R_max:,.2f} kN",
            help="This is the maximum load that any individual pile must resist, used for sizing the piles and the pile cap shear check."
        )
    else:
        st.info("Enter pile geometry and loads to calculate reactions.")
    
    st.markdown("---")
    
    # Display Pile Reaction Table
    st.subheader("Individual Pile Reactions (R)")
    
    # Find the critical pile for display
    critical_pile_index = np.argmax([abs(r[2]) for r in reactions]) if reactions else -1

    data = []
    for i, (x, y, R_i) in enumerate(reactions):
        is_critical = "⭐" if i == critical_pile_index else ""
        data.append({
            "Pile #": i + 1,
            "x (m)": f"{x:,.2f}",
            "y (m)": f"{y:,.2f}",
            "Reaction (kN)": f"{R_i:,.2f} {is_critical}",
        })
    
    st.dataframe(data, use_container_width=True, hide_index=True)

# --- Second Column: Shear Check ---
with col_res2:
    st.subheader("Pile Cap Punching Shear Check (ACI 318-19)")
    
    if R_max > 0 and H_cap > 0:
        V_u, V_c_allow, d_eff, check_status, b_o = check_punching_shear(R_max, H_cap, Dp, fc_prime, load_factor)
        
        st.markdown(f"**Effective Depth $d$ (Assumed):** ${d_eff:,.3f} \: \mathrm{{m}}$ (Total Depth $H$ - 150mm cover)")
        st.markdown(f"**Critical Perimeter $b_o$:** ${b_o:,.3f} \: \mathrm{{m}}$")
        
        st.markdown("---")

        st.metric(
            label="Ultimate Shear Force $V_u$",
            value=f"{V_u:,.2f} kN",
            help=f"$V_u = R_{{max}} \\times LF = {R_max:,.2f} \mathrm{{kN}} \\times {load_factor}$"
        )
        
        st.metric(
            label="Allowable Shear Capacity $\phi V_c$",
            value=f"{V_c_allow:,.2f} kN",
            help="Based on ACI 318-19 Eq. 22.6.5.2 (simplified) for Two-way Shear with $\phi=0.75$."
        )

        st.markdown("---")

        # Display Check Status
        if check_status == "O.K.":
            st.success(f"**Shear Check Status:** {check_status}")
            st.markdown(f"**Conclusion:** $\phi V_c$ ({V_c_allow:,.2f} kN) $\geq V_u$ ({V_u:,.2f} kN)")
        else:
            st.error(f"**Shear Check Status:** {check_status}")
            st.markdown(f"**Conclusion:** $\phi V_c$ ({V_c_allow:,.2f} kN) $< V_u$ ({V_u:,.2f} kN). **Increase $H_{cap}$.**")
    else:
        st.info("Complete the input fields to perform the shear check.")

# --- Flexural Check Section ---
st.divider()
st.subheader("Flexural Reinforcement Design (Simplified ACI Check)")
st.caption("Simplified check: Design moment $M_u$ is calculated by summing ultimate pile forces multiplied by the distance to the cap centroid. $\phi=0.9$.")

# Estimate Lx and Ly for calculating area (Cap dimensions assumed 0.5m past outermost piles)
Lx_cap = (Nx - 1) * Sx + 2 * 0.5 
Ly_cap = (Ny - 1) * Sy + 2 * 0.5

# Perform flexural check
if R_max > 0 and H_cap > 0 and fy > 0:
    # Get d_eff from shear check
    _, _, d_eff, _, _ = check_punching_shear(R_max, H_cap, Dp, fc_prime, load_factor) 
    
    flexural_results, status = check_flexural_reinforcement(reactions, d_eff, Lx_cap, Ly_cap, fc_prime, fy, load_factor)
    
    flex_col1, flex_col2 = st.columns(2)
    
    with flex_col1:
        st.markdown(f"#### X-Direction Reinf. (Perpendicular to Y-axis)")
        st.markdown(f"**Cap Width $L_y$ (b):** {Ly_cap:,.2f} m")
        st.metric(
            label="Design Moment $M_{u,y}$ (about Y-axis)", 
            value=f"{flexural_results['Mu_y']:,.2f} kN-m"
        )
        if flexural_results['As_x_req'] == 99999.0:
             st.error("Depth $H_{cap}$ is insufficient for flexure in X-dir.")
        else:
             st.metric(
                label="Required Steel Area $A_{s,x}$", 
                value=f"{flexural_results['As_x_req']:,.0f} $mm^2$",
                help="Includes minimum reinforcement requirements (ACI 318-19 9.6.1.1)"
             )

    with flex_col2:
        st.markdown(f"#### Y-Direction Reinf. (Perpendicular to X-axis)")
        st.markdown(f"**Cap Width $L_x$ (b):** {Lx_cap:,.2f} m")
        st.metric(
            label="Design Moment $M_{u,x}$ (about X-axis)", 
            value=f"{flexural_results['Mu_x']:,.2f} kN-m"
        )
        if flexural_results['As_y_req'] == 99999.0:
             st.error("Depth $H_{cap}$ is insufficient for flexure in Y-dir.")
        else:
            st.metric(
                label="Required Steel Area $A_{s,y}$", 
                value=f"{flexural_results['As_y_req']:,.0f} $mm^2$",
                help="Includes minimum reinforcement requirements (ACI 318-19 9.6.1.1)"
            )

else:
    st.info("Complete the input fields (especially Steel Strength) to perform the flexural check.")

# --- Visualization (Plotly Plan View) ---
st.divider()
st.header("Pile Group Plan View (Plotly Interactive)")
st.caption(f"Piles are colored/sized by reaction magnitude (Red = Compression, Blue = Tension).")

# Store H_cap in session state so draw_piles can access d_eff for plotting
st.session_state['H_cap'] = H_cap

if reactions:
    try:
        draw_piles_plotly(reactions, Nx, Ny, Sx, Sy, Dp, R_max, H_cap)
    except Exception as e:
        st.warning("Could not generate interactive plot. Ensure all inputs are valid.")
