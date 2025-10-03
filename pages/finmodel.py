import streamlit as st
import pandas as pd
import io
import numpy as np

# --- Data Loading and Cleaning Functions ---
# Note: In a real-world scenario, you would use st.cache_data for performance.
# Since we are simulating file uploads, direct loading is used here.

def load_data(file_name, skiprows=0, header=0, usecols=None):
    """Loads a CSV file into a DataFrame, handling potential initial empty rows."""
    try:
        # The estimatedRowsAboveHeader suggests where the header *might* start.
        # We try to load the file, skipping up to 20 rows max to find the data block.
        for skip in range(skiprows, 20):
            try:
                df = pd.read_csv(file_name, skiprows=skip, header=header, usecols=usecols)
                if not df.empty and df.iloc[:, 0].dropna().empty:
                    # If the first column (A) is all NaN/empty after loading, the header might be one row up/down.
                    # We assume a non-empty first column indicates a successful load of the main table.
                    # This check is a heuristic for complex CSV exports from Excel.
                    continue
                return df
            except pd.errors.ParserError:
                continue
            except Exception as e:
                # If loading fails entirely, break and raise the initial error
                raise e
        return pd.read_csv(file_name, skiprows=skiprows, header=header, usecols=usecols)

    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return pd.DataFrame()

def get_financial_df(file_name, name_row_index, year_row_index, num_years=15):
    """Loads a financial statement file, cleans headers, and prepares for display/charting."""
    try:
        # Load the CSV, skipping the initial blank rows
        df = load_data(file_name, skiprows=4, header=None)

        if df.empty:
            return df

        # The data structure has the main row/column labels in the first few columns
        # and years in subsequent columns.
        
        # Determine columns that contain numerical data (Years)
        # Assuming the year index row (Row 1 in the original CSV starting from 'Year')
        # is actually at index 1 in the loaded DataFrame.
        # Year values are around the 4th or 5th column.
        
        # 1. Extract the row containing the years (e.g., 1, 2, 3...)
        year_row = df.iloc[0].dropna().astype(str)
        
        # 2. Extract the row containing the line item names (e.g., 'Term Loan', 'Profit After Tax (PAT)')
        name_col = df.iloc[:, 0].fillna(df.iloc[:, 1]).fillna(df.iloc[:, 2]).fillna(df.iloc[:, 3])
        
        # 3. Re-align data and headers
        data_start_row = 3 # Data starts usually after 'Year' and 'Year Ending' rows, and an empty row.

        # The actual data usually starts around column index 3 (after empty columns and name columns)
        data_df = df.iloc[data_start_row:, 3:].reset_index(drop=True)

        # Create new column names for the time series
        year_columns = [f"Year {int(y)}" for y in year_row.values[:num_years] if str(y).replace('.', '', 1).isdigit() and float(y) == int(float(y))]
        
        # If year_columns is not long enough, use generic names
        if len(year_columns) < num_years:
             year_columns = [f"Year {i}" for i in range(1, num_years + 1)]

        # Drop the first few header rows and use the cleaned column names
        final_df = df.iloc[data_start_row:]
        final_df = final_df.iloc[:, 3:3+num_years]
        final_df.columns = year_columns[:final_df.shape[1]]

        # Use the name column as the index
        name_index = name_col.iloc[data_start_row:].reset_index(drop=True)
        final_df.index = name_index[:final_df.shape[0]]

        # Clean up column and index names and convert to numeric
        final_df = final_df.apply(pd.to_numeric, errors='coerce').fillna(0).round(2)
        final_df.index.name = 'Particulars'
        
        return final_df

    except Exception as e:
        st.warning(f"Could not fully clean financial data for {file_name}. Displaying raw data. Error: {e}")
        return load_data(file_name, skiprows=4) # Fallback to a simpler load


def get_summary_data(file_name):
    """Extracts key metrics and Capex summary."""
    df = load_data(file_name, skiprows=1) # Data starts around row 2

    if df.empty:
        return {'metrics': pd.DataFrame(), 'capex': pd.DataFrame()}
    
    # 1. Extract Key Metrics (right side)
    metrics_data = df.iloc[2:9, 7:9].dropna(how='all')
    metrics_data.columns = ['Metric', 'Value']
    metrics_data = metrics_data.set_index('Metric')

    # 2. Extract Capex Assumptions (left side)
    capex_data = df.iloc[4:12, 1:3].dropna(how='all')
    capex_data.columns = ['Particulars', 'Amount (Crs)']
    capex_data = capex_data.set_index('Particulars')
    
    # Clean up numerical values
    try:
        metrics_data['Value'] = pd.to_numeric(metrics_data['Value'], errors='coerce').round(4)
        capex_data['Amount (Crs)'] = pd.to_numeric(capex_data['Amount (Crs)'], errors='coerce').round(2)
    except:
        # Ignore errors if data cleanup fails, use raw data
        pass

    return {'metrics': metrics_data, 'capex': capex_data}

def get_material_balance_df(file_name):
    """Loads and cleans the Material Balance data."""
    # Data starts around row 4 (header at 3)
    df = load_data(file_name, skiprows=4, header=None, usecols=[1, 2, 3, 4])
    if df.empty:
        return df

    df.columns = ['STAGE', 'PROCESS DESCRIPTION', 'QUANTITY (TPD)', '% OF INPUT']
    df = df.dropna(subset=['PROCESS DESCRIPTION']).reset_index(drop=True)
    df = df.replace('-', np.nan)
    
    # Convert quantities to numeric, coercion errors for text rows
    df['QUANTITY (TPD)'] = pd.to_numeric(df['QUANTITY (TPD)'], errors='coerce').round(2)
    df['% OF INPUT'] = pd.to_numeric(df['% OF INPUT'], errors='coerce').round(2)
    
    # Fill down STAGE for visual grouping
    df['STAGE'] = df['STAGE'].replace('', np.nan).fillna(method='ffill')

    return df

def get_assumptions_df(file_name):
    """Loads and cleans the simple list of assumptions."""
    # Assumptions start around row 3
    df = load_data(file_name, skiprows=3, header=None, usecols=[0, 1])
    if df.empty:
        return df
    
    df.columns = ['Index', 'Assumption']
    df = df.dropna(subset=['Assumption']).reset_index(drop=True)
    df = df[df['Assumption'] != ''] # Remove rows with empty assumption text
    
    return df

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Bareilly MSW Project Financial Model")
st.title("Bareilly MSW Project Financial Model Analysis")
st.markdown("This dashboard provides an interactive overview of the key financial projections, assumptions, and physical balance for the Municipal Solid Waste (MSW) Processing and Disposal (P&D) project in Bareilly.")

# --- File Paths ---
# Use an in-memory dictionary to simulate access to the uploaded files
FILE_MAP = {
    "Summary": "Bareily.xlsx - Summary.csv",
    "Assumptions": "Bareily.xlsx - Assumptions.csv",
    "Material Balance": "Bareily.xlsx - Material Balance_1.csv",
    "P&L": "Bareily.xlsx - PL.csv",
    "Cash Flow": "Bareily.xlsx - CF.csv",
    "Balance Sheet": "Bareily.xlsx - BS.csv",
    "Opex": "Bareily.xlsx - Opex P&D.csv",
    "Capex Detail": "Bareily.xlsx - PC.csv",
    "Debt (TL)": "Bareily.xlsx - TL.csv",
}

# --- Create Tabs ---
tab_summary, tab_assumptions, tab_material, tab_financials, tab_capex, tab_opex_debt = st.tabs([
    "Summary & Metrics", 
    "Assumptions & Policies", 
    "Material Balance & Flow", 
    "Financial Statements", 
    "Capital Expenditure",
    "Opex & Debt"
])

# --- Tab 1: Summary & Metrics ---
with tab_summary:
    st.header("Project Summary & Key Financial Metrics")
    
    summary_data = get_summary_data(FILE_MAP["Summary"])
    metrics_df = summary_data['metrics']
    capex_summary_df = summary_data['capex']

    if not metrics_df.empty:
        st.subheader("Key Financial Metrics")
        col1, col2, col3 = st.columns(3)
        
        # Display key metrics as big numbers
        col1.metric("PIRR (Project IRR)", f"{metrics_df.loc['PIRR', 'Value'] * 100:.2f} %")
        col2.metric("EIRR (Equity IRR)", f"{metrics_df.loc['EIRR', 'Value'] * 100:.2f} %")
        col3.metric("PNPV @ 14%", f"₹ {metrics_df.loc['PNPV@14%', 'Value']:.2f} Crs")
        
        st.markdown("---")

    if not capex_summary_df.empty:
        st.subheader("Project Financing & Capex Summary (in ₹ Crs)")
        
        # Display Capex details
        st.dataframe(capex_summary_df, use_container_width=True)

        # Chart for Debt:Equity
        financing_data = capex_summary_df.loc[['Debt', 'Equity']].copy()
        financing_data.columns = ['Amount']
        
        st.subheader("Debt vs. Equity Composition")
        st.bar_chart(financing_data, height=300)

# --- Tab 2: Assumptions & Policies ---
with tab_assumptions:
    st.header("Project Assumptions and Financial Policies")
    
    assumptions_df = get_assumptions_df(FILE_MAP["Assumptions"])

    if not assumptions_df.empty:
        st.subheader("Core Assumptions (Excerpt)")
        st.dataframe(assumptions_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Highlights from Assumptions:**
        - **Project Scope:** P&D process for 500 TPD with $1\%$ escalation YOY.
        - **Financing:** Debt and Equity portion is **$70:30$**, with Interest on term loan taken as $11\%$.
        - **Revenue:** Tipping fee is $\sim$Rs. 1500/ton with $5\%$ escalation YOY. RDF Sale price is $\sim$Rs. 250/ton with $2\%$ escalation YOY.
        - **Operational Days:** 365 days.
        - **Tax Benefit:** Section 80 IA available for a few initial years.
        """)

# --- Tab 3: Material Balance & Flow ---
with tab_material:
    st.header("Waste Material Balance and Processing Flow (Per Day)")
    
    material_balance_df = get_material_balance_df(FILE_MAP["Material Balance"])

    if not material_balance_df.empty:
        st.subheader("Material Balance (300 TPD Input Example)")
        st.dataframe(
            material_balance_df.style.highlight_max(subset=['QUANTITY (TPD)'], axis=0),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Key Output Breakdown")
        
        # Extract final output (assuming last rows contain the final outputs)
        final_outputs = material_balance_df[material_balance_df['PROCESS DESCRIPTION'].str.contains('Saleable RDF|Compost|Inert for Landfill', na=False, case=False)].tail(3).set_index('PROCESS DESCRIPTION')['QUANTITY (TPD)']
        
        if not final_outputs.empty:
            st.bar_chart(final_outputs, height=300)
            st.markdown(f"The model details the conversion from **Total Waste Input** (e.g., 300 TPD in this example) through sorting, windrowing, and screening to final products like **Saleable RDF** and **Compost**, with inert waste for Landfill.")

# --- Tab 4: Financial Statements ---
with tab_financials:
    st.header("Projected Financial Statements (15 Years)")

    st.subheader("Data Selection")
    statement_choice = st.radio("Select Statement:", ["Profit & Loss (P&L)", "Cash Flow (CF)", "Balance Sheet (BS)"], horizontal=True)

    file_key = statement_choice.split('(')[0].strip()
    
    # Map selection to file key and load the respective data
    financial_df = get_financial_df(FILE_MAP[file_key], name_row_index=3, year_row_index=0)

    if not financial_df.empty:
        st.subheader(f"Projected {statement_choice} (Amounts in ₹ Lakhs)")
        
        # Use a slider to select the number of years to display
        max_years = financial_df.shape[1]
        years_to_show = st.slider("Select Projection Period (Years)", 1, max_years, max_years, key=f'slider_{file_key}')
        
        st.dataframe(financial_df.iloc[:, :years_to_show], use_container_width=True)

        st.subheader("Key Line Item Trends")
        
        # Charting key items based on the statement type
        if statement_choice == "Profit & Loss (P&L)":
            chart_items = ['Revenue from Tipping Fee', 'Total Operating Expenses', 'Profit Before Tax (PBT)', 'Profit After Tax (PAT)']
        elif statement_choice == "Cash Flow (CF)":
            chart_items = ['Profit After Tax (PAT)', 'Net Cash from Operating Activities', 'Closing Balance']
        elif statement_choice == "Balance Sheet (BS)":
            chart_items = ['Total Assets', 'Total Liabilities', 'Equity']
        
        chart_data = financial_df.loc[financial_df.index.intersection(chart_items), :years_to_show]
        if not chart_data.empty:
            st.line_chart(chart_data.T)
        else:
            st.info("No key line items available for charting in this view.")

# --- Tab 5: Capital Expenditure ---
with tab_capex:
    st.header("Detailed Capital Expenditure (Capex) Breakdown")
    
    # Loading the detailed PC (Project Cost) sheet.
    # We will try to load the main block of Capex items.
    pc_df = load_data(FILE_MAP["Capex Detail"], skiprows=15, header=None, usecols=[0, 1, 2, 3])
    
    if not pc_df.empty:
        # Heuristic cleanup: assume columns are Item, Units/Qty, Rate/Amount, Total Cost
        pc_df.columns = ['Category', 'Sub-Category', 'Amount/Unit', 'Total Cost in Year 1']
        
        # Clean data (Remove empty/header-like rows)
        pc_df = pc_df.dropna(subset=['Sub-Category']).reset_index(drop=True)
        pc_df = pc_df[~pc_df['Sub-Category'].astype(str).str.contains('Required \+ Standby|Units|Total', case=False)]
        
        # Fill down Category for better grouping
        pc_df['Category'] = pc_df['Category'].replace('', np.nan).fillna(method='ffill').str.strip()
        
        st.subheader("Major Capex Components (Amounts in ₹ Lakhs)")
        st.dataframe(pc_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Capex Summary:**
        - **P&D Processing Vehicles:** ₹ 6.62 Crs
        - **Civil Cost:** ₹ 43.73 Crs (covering Compost Plant, Landfill, and Supporting Infrastructure)
        - The model includes specific details on Landfill Equipment (Excavator, Sheep foot roller) and Civil works (Weighbridge, Storm water drain).
        """)

# --- Tab 6: Opex & Debt ---
with tab_opex_debt:
    st.header("Operating Expenses and Debt Structure")
    
    st.subheader("Data Selection")
    choice = st.radio("Select Detail:", ["Operating Expenses (Opex)", "Term Loan Repayment (TL)"], horizontal=True)

    if choice == "Operating Expenses (Opex)":
        opex_df = get_financial_df(FILE_MAP["Opex"], name_row_index=3, year_row_index=0)
        st.subheader("Operating Expenses Projection (Amounts in ₹ Lakhs)")
        if not opex_df.empty:
            st.dataframe(opex_df, use_container_width=True)
            
            # Charting key Opex items
            chart_opex_items = ['Salaries & Wages - Drivers', 'LandFill Cost', 'Total']
            chart_data = opex_df.loc[opex_df.index.intersection(chart_opex_items), :]
            
            if not chart_data.empty:
                st.line_chart(chart_data.T)
        else:
            st.warning("Opex data could not be loaded or cleaned successfully.")

    elif choice == "Term Loan Repayment (TL)":
        # Term Loan data is complex (monthly/yearly view). We'll simplify to the annual summary.
        tl_df_annual = get_financial_df(FILE_MAP["Debt (TL)"], name_row_index=12, year_row_index=0)
        
        st.subheader("Annual Term Loan Debt Servicing (Amounts in ₹ Lakhs)")
        if not tl_df_annual.empty:
            # We are interested in Interest, Principal, and Outstanding Debt
            interest_row = tl_df_annual.index[tl_df_annual.index.str.contains('Interest', na=False)].tolist()
            principal_row = tl_df_annual.index[tl_df_annual.index.str.contains('Principal Repayment', na=False)].tolist()
            outstanding_row = tl_df_annual.index[tl_df_annual.index.str.contains('Outstanding Debt', na=False)].tolist()
            
            key_debt_rows = [r for r in [interest_row, principal_row, outstanding_row] if r]
            
            if key_debt_rows:
                 # Flatten the list of lists and get unique indices
                key_indices = list(set([item for sublist in key_debt_rows for item in sublist]))
                st.dataframe(tl_df_annual.loc[key_indices], use_container_width=True)
                
                # Chart Outstanding Debt
                outstanding_debt_data = tl_df_annual.loc[key_indices].filter(like='Outstanding Debt', axis=0).T
                
                if not outstanding_debt_data.empty:
                    st.subheader("Outstanding Debt Over Time")
                    st.line_chart(outstanding_debt_data)
            else:
                 st.warning("Could not identify key debt rows (Interest, Principal, Outstanding). Displaying raw cleaned data.")
                 st.dataframe(tl_df_annual, use_container_width=True)
        else:
            st.warning("Term Loan data could not be loaded or cleaned successfully.")
