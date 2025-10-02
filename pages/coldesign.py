import streamlit as st
import numpy as np

# --- Configuration for Printability and Appearance ---
st.set_page_config(
    page_title="Printable Column Design Check (Simulated)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, printable look and feel
# Uses st.markdown(..., unsafe_allow_html=True) to inject CSS.
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

            /* Input Sidebar Styling */
            .stSidebar {
                background-color: #f0f4f7;
                padding: 15px;
            }

            /* Results Section Styling */
            .stAlert {
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* Print-specific styles: Hide non-essential UI elements for printing */
            @media print {
                .st-emotion-cache-18ni7ap, .st-emotion-cache-1v06a5k, 
                .st-emotion-cache-79elbk, .stButton, .st-emotion-cache-16cqjfa {
                    display: none !important; /* Hide header, sidebar toggle, and buttons */
                }
                
                /* Ensure main content takes full width */
                .main .block-container {
                    padding-left: 0 !important;
                    padding-right: 0 !important;
                }

                /* Show a print title */
                @page { size: A4 portrait; margin: 1cm; }
            }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css()

# --- Placeholder Calculation Logic ---

def calculate_column_design(b, h, fc, fy, Pu, Mux, Muy, As_total):
    """
    *** IMPORTANT: PLACEHOLDER FUNCTION ***
    
    You must replace the logic below with the actual engineering calculations 
    from your COLDESIGN.xlsm file (e.g., ACI 318, Eurocode, etc.)
    
    This function currently provides a SIMULATED result.
    """
    
    # 1. Simulate Nominal Capacity (Replace with your actual Pn/Mn formulas)
    # Simplified capacity calculation for demonstration (based on concrete + steel)
    Ag = b * h
    # Dummy factors to simulate capacity contribution
    Pn_nominal = 0.85 * fc * (Ag - As_total) + fy * As_total 
    
    # 2. Apply Reduction Factor (phi) - Example ACI for Tie Columns
    phi = 0.65
    Phi_Pn = phi * Pn_nominal * 0.95 # Additional safety factor simulation

    # 3. Check Capacity
    # Simulate a critical interaction ratio (e.g., for biaxial bending check)
    # This is a highly simplified proxy for the actual interaction diagram check.
    interaction_ratio = (Pu / Phi_Pn) + (np.sqrt(Mux**2 + Muy**2) / (0.1 * Phi_Pn * np.sqrt(b*h)))
    
    # Simple pass/fail based on ratio
    status = "Pass" if interaction_ratio < 1.0 else "Fail"
    
    # Simulate required area based on load
    As_required = (Pu / (0.65 * fy)) / 0.8 

    return {
        "Phi_Pn": Phi_Pn / 1000, # Convert N to kN for display
        "Interaction_Ratio": interaction_ratio,
        "Status": status,
        "As_required": As_required,
    }


# --- Streamlit UI Layout ---

st.title("Structural Column Design Check")
st.markdown("---")

# Instructions and User ID (Important for collaboration and traceability)
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Enter all necessary parameters in the sections below."
    "\n2. Click the 'Run Design Check' button."
    "\n3. To print, use your browser's print function (Ctrl+P or Command+P)."
    " The page is optimized for a clean single-page output."
)

# Use columns for main layout
col_input, col_section = st.columns([2, 1])

with col_input:
    st.subheader("1. Design Inputs & Loads")
    
    # Material Properties
    st.markdown("#### Material Properties")
    mat_col1, mat_col2 = st.columns(2)
    with mat_col1:
        fc = st.number_input("Concrete Compressive Strength, f'c (MPa)", min_value=15.0, value=30.0, step=1.0)
    with mat_col2:
        fy = st.number_input("Steel Yield Strength, fy (MPa)", min_value=300.0, value=420.0, step=10.0)

    # Loads (Factored)
    st.markdown("#### Factored Design Loads (Pu, Mu)")
    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        Pu = st.number_input("Axial Load, Pu (kN)", min_value=0.0, value=1500.0, step=100.0) * 1000 # Convert to N
    with load_col2:
        Mux = st.number_input("Moment about X-axis, Mu,x (kNm)", min_value=0.0, value=50.0, step=5.0) * 1000000 # Convert to N-mm
    with load_col3:
        Muy = st.number_input("Moment about Y-axis, Mu,y (kNm)", min_value=0.0, value=30.0, step=5.0) * 1000000 # Convert to N-mm

with col_section:
    st.subheader("2. Section & Reinforcement")

    # Section Dimensions
    st.markdown("#### Column Section (mm)")
    dim_col1, dim_col2 = st.columns(2)
    with dim_col1:
        b = st.number_input("Width, b (mm)", min_value=100, value=400, step=10)
    with dim_col2:
        h = st.number_input("Depth, h (mm)", min_value=100, value=600, step=10)

    # Reinforcement
    st.markdown("#### Reinforcement Area")
    # This is a simplification; a real app would calculate As from bar count/size.
    # We will use total area for the placeholder calculation.
    As_total = st.number_input("Total Steel Area, As,total (mm²)", min_value=0.0, value=3200.0, step=100.0)


st.markdown("---")

# --- Calculation Trigger ---
if st.button("Run Design Check", type="primary"):
    
    if b <= 0 or h <= 0:
        st.error("Please provide valid, non-zero dimensions for the column section.")
    else:
        # Run the placeholder calculation
        results = calculate_column_design(
            b, h, fc, fy, Pu, Mux, Muy, As_total
        )
        
        st.subheader("3. Design Results Summary")
        
        # Display Results using metrics and a status box
        results_col1, results_col2, results_col3 = st.columns(3)
        
        with results_col1:
            st.metric(
                label="Factored Axial Capacity ($\phi P_n$)", 
                value=f"{results['Phi_Pn']:.2f} kN",
                help="The maximum factored axial load the section can carry (P-M Interaction Check is critical)."
            )

        with results_col2:
            interaction_ratio = results['Interaction_Ratio']
            st.metric(
                label="Interaction Ratio ($P_u/\phi P_n + M_u/\phi M_n$)", 
                value=f"{interaction_ratio:.3f}",
                delta="< 1.0 is required"
            )

        with results_col3:
            st.metric(
                label="Required Steel Area, $A_{s,req}$ (Simulated)", 
                value=f"{results['As_required']:.2f} mm²",
                help="A simulation of the minimum steel required based on axial load."
            )

        # Final Pass/Fail Status
        if results['Status'] == "Pass":
            st.success("✅ DESIGN CHECK PASS: Interaction Ratio < 1.0")
        else:
            st.error("❌ DESIGN CHECK FAIL: Interaction Ratio $\ge$ 1.0. Increase section size or reinforcement.")

        
        # Detailed Input Summary for Print
        st.markdown("""
        <div style="page-break-before: always;"> 
            <h3 style="margin-top: 40px; color: #4b4b4b;">Input Summary for Record</h3>
        </div>
        """, unsafe_allow_html=True)

        data = {
            "Parameter": ["Concrete Strength, f'c", "Steel Yield, fy", "Width, b", "Depth, h", "Axial Load, Pu", "Moment Mu,x", "Moment Mu,y", "Provided As"],
            "Value": [f"{fc} MPa", f"{fy} MPa", f"{b} mm", f"{h} mm", f"{Pu/1000:,.0f} kN", f"{Mux/1000000:,.0f} kNm", f"{Muy/1000000:,.0f} kNm", f"{As_total} mm²"],
        }
        
        st.table(data)

# Instructions to print are generally found in the top right of the browser menu.
st.sidebar.markdown("---")
st.sidebar.markdown("**Print Tip:** Use `Ctrl+P` (or `Cmd+P`) to print the results view.")
