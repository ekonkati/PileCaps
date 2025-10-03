import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Utility Functions for Data Loading and Extraction ---

# Global constant for the required currency denomination conversion (Lakhs to Crores)
# Based on snippet from Assu.csv: All the Projections in ,100000,1,,,
# Rs. Lakhs / 100 = Rs. Crores
CONVERSION_FACTOR = 100

@st.cache_data
def load_data(filepath, header_row_offset):
    """Loads a non-standard CSV file by skipping initial rows and inferring headers."""
    try:
        # Determine the number of rows to skip to reach the header row
        skip_rows = header_row_offset - 1
        
        # Load the data, using the row at 'header_row_offset' as the header
        df = pd.read_csv(filepath, skiprows=skip_rows)
        
        # Clean up columns by stripping whitespace and removing NaN column names
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return df
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def extract_summary_metric(df, row_name, column_name):
    """Extracts a single numerical metric from the Summary/Assu-like dataframes."""
    try:
        # Find the row containing the metric name (case-insensitive and partial match)
        row = df[df.apply(lambda x: x.astype(str).str.contains(row_name, case=False, na=False)).any(axis=1)]
        
        if row.empty:
            return np.nan

        # Assuming the value is in the specified column relative to the row name
        value = row[column_name].iloc[0]
        
        # Try to convert to float, return 0 if conversion fails
        return float(value) if pd.notna(value) and value != '' else 0.0
    except Exception:
        # Fallback to 0.0 if any extraction logic fails
        return 0.0

def process_financial_statements(pl_df, cf_df):
    """Processes P&L and CF data for plotting and key metric display."""
    
    # 1. Standardize P&L Columns
    pl_df.columns = [col.replace('.', '').replace(',', '').strip() for col in pl_df.columns]
    
    # Identify Year and Financial Columns
    # The columns 2 to 16 are the years (1 to 15) in the PL.csv snippet
    year_cols = pl_df.columns[3:18] 
    
    # Extract key rows (Revenue and Expenses)
    # The snippet shows key financials start at specific rows
    pl_data = {}
    
    # Revenue (Row above the first value in the PL snippet, seems to be the total revenue row)
    revenue_row_index = pl_df[pl_df['Year'].astype(str).str.contains('1', na=False)].index[0] - 2 
    pl_data['Total Revenue'] = pl_df.iloc[revenue_row_index][year_cols].apply(pd.to_numeric, errors='coerce') / CONVERSION_FACTOR
    
    # Opex (Approximation: Sum of fuel, manpower, power, etc. We'll use a placeholder row or estimate)
    # Based on general structure, let's find the 'Opex' row, or approximate it.
    # The snippet doesn't clearly label Total Opex, so we'll approximate based on the structure around the first data point.
    opex_data = pl_df[pl_df.iloc[:, 2].astype(str).str.contains('Total Operating Expenses', na=False)] 
    
    if not opex_data.empty:
        pl_data['Total Opex'] = opex_data[year_cols].iloc[0].apply(pd.to_numeric, errors='coerce') / CONVERSION_FACTOR
    else:
        # As a fallback, use the row before PAT for PBT, or similar structure
        pl_data['Total Opex'] = [0] * len(year_cols) # Fallback to zero

    # PAT (Profit After Tax)
    pat_row_index = pl_df[pl_df.iloc[:, 2].astype(str).str.contains('Profit After Tax', na=False)].index[0]
    pl_data['PAT'] = pl_df.iloc[pat_row_index][year_cols].apply(pd.to_numeric, errors='coerce') / CONVERSION_FACTOR

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pl_data, index=[f'Year {i+1}' for i in range(len(year_cols))])
    plot_df.index.name = 'Year'
    
    # Add Year Ending for better visualization
    year_ending_row = pl_df[pl_df.iloc[:, 2].astype(str).str.contains('Year Ending', na=False)]
    plot_df['Year Ending'] = year_ending_row[year_cols].iloc[0].values
    
    return plot_df.set_index('Year Ending')


# --- 2. Load Data ---

# Assumptions and Financial Summary are key entry points
df_assu = load_data('Bareily.xlsx - Assu.csv', header_row_offset=77) 
df_summary = load_data('Bareily.xlsx - Summary.csv', header_row_offset=3)
df_pl = load_data('Bareily.xlsx - PL.csv', header_row_offset=6)
df_tl = load_data('Bareily.xlsx - TL.csv', header_row_offset=13)
df_opex_assum = load_data('Bareily.xlsx - Assumptions.csv', header_row_offset=3)


# --- 3. Extract Key Metrics ---

# Valuation Metrics (from Summary.csv snippet)
pirr = extract_summary_metric(df_summary, 'PIRR', 'P&D') * 100
eirr = extract_summary_metric(df_summary, 'EIRR', 'P&D') * 100
pnpv = extract_summary_metric(df_summary, 'PNPV@14%', 'P&D')

# Project Cost (from Summary.csv snippet)
capex = extract_summary_metric(df_summary, 'Capex', 'P&D')
debt = extract_summary_metric(df_summary, 'Debt', 'P&D')
equity = extract_summary_metric(df_summary, 'Equity', 'P&D')
debt_equity = extract_summary_metric(df_summary, 'Debt:Equity', 'P&D')

# Tipping Fee and RDF Price (from Assumptions.csv)
tipping_fee_row = df_opex_assum[df_opex_assum.iloc[:, 0].astype(str).str.contains('Tipping fee', na=False)]
tipping_fee_desc = tipping_fee_row.iloc[0, 1] if not tipping_fee_row.empty else "N/A"
rdf_price_row = df_opex_assum[df_opex_assum.iloc[:, 0].astype(str).str.contains('RDF Sale price', na=False)]
rdf_price_desc = rdf_price_row.iloc[0, 1] if not rdf_price_row.empty else "N/A"


# --- 4. Process Financial Statements for Visualization ---
df_plot_data = process_financial_statements(df_pl, pd.DataFrame())


# --- 5. Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Bareilly MSW Project Financial Model Analysis")

st.title("🗑️ Bareilly MSW Project Finance Dashboard")
st.markdown("A consolidated analysis of the 15-year Project Finance Model (Values in **Rs. Crores** unless noted otherwise)")

---

# --- KPI Section (Valuation and Project Cost) ---
st.header("Financial Feasibility & Project Capital")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Equity IRR (EIRR)", f"{eirr:.2f}%", help="Annualized return to the equity investors.")
with col2:
    st.metric("Project IRR (PIRR)", f"{pirr:.2f}%", help="Annualized return on the total project investment.")
with col3:
    st.metric("NPV @ 14% (Rs. Cr.)", f"{pnpv:.2f}", help="Net Present Value of the project at a 14% discount rate.")
with col4:
    st.metric("Debt-to-Equity Ratio", f"{debt_equity:.2f}", help="Financing structure of the project.")

st.subheader("Total Project Cost (CAPEX)")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total CAPEX (Rs. Cr.)", f"{capex:.2f}")
col_b.metric("Term Debt (Rs. Cr.)", f"{debt:.2f}")
col_c.metric("Sponsor Equity (Rs. Cr.)", f"{equity:.2f}")

---

# --- P&L & Cash Flow Trends ---
st.header("Profitability & Operational Trends (Rs. Crores)")

if not df_plot_data.empty:
    st.line_chart(
        df_plot_data[['Total Revenue', 'Total Opex', 'PAT']],
        use_container_width=True
    )
    
    st.markdown("The chart shows the trajectory of key financials over the 15-year project period.")
    
    # Key Financial Highlights
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Key Revenue Assumptions")
        st.info(f"""
        - **Tipping Fee:** {tipping_fee_desc}
        - **RDF Sale Price:** {rdf_price_desc}
        """)
    with col6:
        st.subheader("Debt Service Summary (First Year)")
        # Extract first year data from TL.csv
        try:
            tl_row = df_tl[df_tl['Month No'].astype(str).str.contains('12', na=False)].iloc[0]
            first_year_interest = tl_row['Interest'] / CONVERSION_FACTOR
            first_year_repayment = tl_row['Principal Repayment'] / CONVERSION_FACTOR
            st.info(f"""
            - **Year 1 Interest Payment:** Rs. {first_year_interest:.2f} Cr.
            - **Year 1 Principal Repayment:** Rs. {first_year_repayment:.2f} Cr.
            - **Interest Rate:** 11% (from Assumptions.csv snippet)
            """)
        except:
            st.warning("Could not extract detailed Year 1 Debt Service data.")
    
else:
    st.error("Financial Statement data could not be processed for visualization.")


---
# --- Raw Data Viewer (For Inspection) ---
st.header("Raw Financial Statement Data")
st.markdown("Inspect the underlying annual financial statements.")

tab_pl, tab_sum = st.tabs(["P&L Statement", "Overall Summary"])

with tab_pl:
    # Use the cleaned plot data for a clearer P&L view
    st.dataframe(df_plot_data.reset_index())

with tab_sum:
    st.dataframe(df_summary)

st.caption("Note: All numbers are approximations due to the highly fragmented nature of the source data.")
