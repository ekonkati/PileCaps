import streamlit as st
import numpy as np

# --- Constants and Setup ---
# ACI 318-19 Constants (simplified)
PHI_SHEAR = 0.75  # Strength reduction factor for shear

def calculate_pile_reactions(P, Mx, My, Nx, Ny, Sx, Sy):
    """
    Calculates the vertical reaction force for each pile in a rigid pile cap 
    using the elastic method.
    R_i = P/N +/- (My * xi / sum(x^2)) +/- (Mx * yi / sum(y^2))

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
    # X-coordinates range from -(Nx-1)/2 * Sx to (Nx-1)/2 * Sx
    x_offset = (Nx - 1) * Sx / 2.0
    for i in range(Nx):
        x = i * Sx - x_offset
        
        # Y-coordinates range from -(Ny-1)/2 * Sy to (Ny-1)/2 * Sy
        y_offset = (Ny - 1) * Sy / 2.0
        for j in range(Ny):
            y = j * Sy - y_offset
            pile_coords.append((x, y))

    # Calculate sum of squares (sum_x2 and sum_y2)
    x_coords = np.array([coord[0] for coord in pile_coords])
    y_coords = np.array([coord[1] for coord in pile_coords])
    
    sum_x2 = np.sum(x_coords ** 2)
    sum_y2 = np.sum(y_coords ** 2)

    reactions = []
    max_R = 0.0

    # Calculate individual pile reactions
    for x, y in pile_coords:
        R_vertical = P / N
        
        # Moment My causes stress in the x-direction
        if sum_x2 != 0:
            R_vertical += (My * x) / sum_x2
        
        # Moment Mx causes stress in the y-direction
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

    Args:
        R_max (float): Maximum vertical pile reaction (kN).
        H_cap (float): Pile cap total depth (m).
        D_pile (float): Pile diameter (m).
        fc_prime (float): Concrete compressive strength (MPa).
        LF (float): Load factor for ultimate shear calculation.

    Returns:
        tuple: (V_u, V_c_allowable, d_eff, check_status)
    """
    
    # Estimate effective depth 'd' (d = H_cap - cover - rebar/2). 
    # Use a conservative estimate of 150mm (0.15m) effective cover for large caps.
    d_eff = H_cap - 0.15 
    
    if d_eff <= 0:
        return 0, 0, d_eff, "Depth too small for effective cover.", 0
        
    # Critical section for two-way shear is at d_eff/2 from the pile face
    b_o_perimeter = np.pi * (D_pile + d_eff) # Circumference of a circle with radius (D_pile/2 + d_eff/2)
    
    # 1. Ultimate Shear Force (Vu)
    V_u_kN = R_max * LF
    V_u_N = V_u_kN * 1000 # Convert to Newtons
    
    # 2. Nominal Concrete Shear Capacity (Vc)
    # ACI 318-19 Eq. 22.6.5.2: Vc = 0.17 * sqrt(f'c) * bo * d (f'c in MPa, bo, d in mm, Vc in N)
    f_c_psi = fc_prime * 145.038 # Convert MPa to psi
    d_eff_mm = d_eff * 1000 
    b_o_mm = b_o_perimeter * 1000
    
    V_c_N = 0.17 * np.sqrt(fc_prime) * b_o_mm * d_eff_mm
    
    # 3. Allowable Shear Capacity
    V_c_allowable_N = PHI_SHEAR * V_c_N
    
    check_status = "O.K." if V_u_N <= V_c_allowable_N else "NOT O.K. (Depth too small)"
    
    return V_u_N/1000, V_c_allowable_N/1000, d_eff, check_status, b_o_perimeter

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Pile Group & Cap Design (Metric)")
st.title("🏗️ Pile Group Design (Elastic Method)")
st.caption("Calculates maximum pile reaction and performs two-way shear check based on ACI 318-19 simplified method (Metric Units: kN, m, MPa).")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Applied Loads (Net of Cap Weight)")
    P_net = st.number_input("Total Vertical Load $P_{net}$ (kN)", value=5000.0, min_value=0.0)
    Mx = st.number_input("Moment about X-axis $M_x$ (kN-m)", value=500.0, step=100.0)
    My = st.number_input("Moment about Y-axis $M_y$ (kN-m)", value=500.0, step=100.0)
    
    st.header("3. Concrete & Safety Factors")
    fc_prime = st.number_input("Concrete Strength $f'_c$ (MPa)", value=30.0, min_value=15.0, max_value=60.0, step=1.0)
    load_factor = st.number_input("Load Factor for $V_u$ (e.g., 1.5)", value=1.5, min_value=1.0, step=0.1)


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
            help="Based on ACI 318-19 Eq. 22.6.5.2 (simplified) for Two-way Shear with $\\phi=0.75$."
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

# --- Visualization (Simplified Plan View) ---
st.header("Pile Group Plan View (Conceptual)")
st.caption(f"Grid Size: {Nx} piles @ {Sx}m in X, {Ny} piles @ {Sy}m in Y.")

# Create a visual representation of the pile group and the critical pile
def draw_piles(reactions, Nx, Ny, Sx, Sy, Dp):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Calculate effective depth for plotting the punching perimeter
    # Note: d_eff must be calculated here or passed in, since it's not global
    H_cap_val = st.session_state.get('H_cap', 1.2) # Retrieve H_cap from session state or use default
    d_eff = H_cap_val - 0.15 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate overall cap dimensions (for drawing boundaries)
    Lx = (Nx - 1) * Sx + 2 * 0.5 # Approximate cap edge distance 0.5m
    Ly = (Ny - 1) * Sy + 2 * 0.5
    
    # Calculate centroid offset
    x_offset_center = (Nx - 1) * Sx / 2.0
    y_offset_center = (Ny - 1) * Sy / 2.0
    
    # Draw Cap boundary (conceptual)
    rect = patches.Rectangle(
        (-x_offset_center - 0.5, -y_offset_center - 0.5), 
        Lx, 
        Ly, 
        linewidth=2, 
        edgecolor='gray', 
        facecolor='none', 
        linestyle='--'
    )
    ax.add_patch(rect)
    
    max_abs_R = max([abs(r[2]) for r in reactions]) if reactions else 1.0
    
    # Find the critical pile for drawing
    critical_pile = None
    if reactions:
        critical_pile_index = np.argmax([abs(r[2]) for r in reactions])
        critical_pile = reactions[critical_pile_index]

    for x, y, R_i in reactions:
        # Determine color based on reaction (Red for max compression, Blue for max tension)
        color = 'red' if R_i == R_max else 'blue' if R_i == -R_max else 'gray'
        
        # Scale size based on reaction magnitude
        size_factor = (abs(R_i) / max_abs_R) * 0.4 + 0.6 # size between 0.6 and 1.0
        
        # Draw Pile
        circle = patches.Circle((x, y), Dp/2 * size_factor, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Add a label for the reaction value
        ax.text(x, y, f"{R_i:,.0f} kN", fontsize=8, ha='center', va='center', color='black' if abs(R_i) / max_abs_R < 0.8 else 'white')

    # Draw the critical pile's punching shear area
    if critical_pile and d_eff > 0:
        crit_x, crit_y, _ = critical_pile
        # Critical perimeter is a circle of diameter Dp + d_eff
        punching_radius = Dp/2 + d_eff/2
        punching_circle = patches.Circle(
            (crit_x, crit_y), 
            punching_radius, 
            color='lime', 
            alpha=0.2, 
            linestyle='--'
        )
        ax.add_patch(punching_circle)
        ax.text(crit_x + punching_radius, crit_y + punching_radius, "Punching Section", fontsize=8, color='green')


    # Set limits and aspect ratio
    x_span = max(Lx, 1.0)
    y_span = max(Ly, 1.0)
    
    ax.set_xlim([-x_offset_center - 0.75, x_offset_center + 0.75])
    ax.set_ylim([-y_offset_center - 0.75, y_offset_center + 0.75])

    ax.set_xlabel("X-direction (m)")
    ax.set_ylabel("Y-direction (m)")
    ax.set_title("Pile Group Plan (Color/Size based on Load)")
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

# Store H_cap in session state so draw_piles can access d_eff for plotting
st.session_state['H_cap'] = H_cap

if reactions:
    try:
        draw_piles(reactions, Nx, Ny, Sx, Sy, Dp)
    except Exception as e:
        st.warning(f"Could not generate plot. Error: {e}")
