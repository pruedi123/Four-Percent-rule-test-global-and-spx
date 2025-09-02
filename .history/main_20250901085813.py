import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Real-Return 4% Rule Simulator", layout="wide")

@st.cache_data
def try_load_default_csv(default_path: str) -> pd.DataFrame | None:
    p = Path(default_path)
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return None

@st.cache_data
def load_from_upload(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def simulate_real_withdrawals(
    factors: pd.Series,
    years: int,
    starting_balance: float,
    annual_withdrawal: float,
    start_row: int = 0,
    step_rows: int = 12,
) -> pd.DataFrame:
    """
    Begin-of-year constant real withdrawal on a series of **real** annual total return factors.
    Uses factors at rows: start_row, start_row+12, start_row+24, ...
    """
    records = []
    balance = float(starting_balance)
    num_rows = len(factors)

    for y in range(1, years + 1):
        idx = start_row + (y - 1) * step_rows
        if idx >= num_rows:
            break
        factor = float(factors.iloc[idx])
        begin_balance = balance
        spend = annual_withdrawal
        end_balance = (begin_balance - spend) * factor
        records.append({
            "Year": y,
            "Row Index Used": idx,
            "Begin Balance": begin_balance,
            "Withdrawal": spend,
            "Real Factor": factor,
            "End Balance": end_balance,
        })
        balance = end_balance

    return pd.DataFrame.from_records(records)

st.title("Real-Return Withdrawal Simulator (Begin-of-Year Spend)")
st.caption(
    "Uses monthly-start **annual real factors**. Year 1 uses the first row in the selected allocation, "
    "year 2 uses the row 12 down, and so on."
)

# --- Data input ---
left, right = st.columns([1,1])
with left:
    st.markdown("**Option A:** Place `all_factors.csv` next to this file (recommended).")
with right:
    st.markdown("**Option B:** Upload a CSV below.")

DEFAULT_PATH = Path(__file__).parent / "all_factors.csv"

df = try_load_default_csv(str(DEFAULT_PATH))

uploaded = st.file_uploader("Or upload all_factors.csv", type=["csv"])
if uploaded is not None:
    df = load_from_upload(uploaded)

if df is None:
    st.error("Couldn't find `all_factors.csv`. Put it in the same folder as this app or upload it above.")
    st.stop()

# Identify allocation columns like '60e'
alloc_cols = [c for c in df.columns if c.endswith("e") and any(ch.isdigit() for ch in c)]
if not alloc_cols:
    st.error("No allocation columns found (e.g., '60e'). Check your CSV headers.")
    st.stop()

default_alloc = "60e" if "60e" in df.columns else alloc_cols[0]

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    alloc = st.selectbox("Allocation (column)", options=alloc_cols, index=alloc_cols.index(default_alloc))
with col2:
    years = st.slider("Number of years", min_value=1, max_value=60, value=30, step=1)
with col3:
    starting_balance = st.number_input(
        "Starting portfolio value", min_value=0.0, value=1_000_000.0, step=50_000.0, format="%.2f"
    )
with col4:
    annual_withdrawal = st.number_input(
        "Initial annual withdrawal (constant real)", min_value=0.0, value=40_000.0, step=1_000.0, format="%.2f"
    )

with st.expander("Advanced options"):
    start_row = st.number_input("Start at data row index", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    step_rows = st.number_input("Row step per year (12 = use next year factor 12 rows down)", min_value=1, max_value=120, value=12, step=1)

# Run the sim
factors = df[alloc]
sim_df = simulate_real_withdrawals(
    factors=factors,
    years=int(years),
    starting_balance=float(starting_balance),
    annual_withdrawal=float(annual_withdrawal),
    start_row=int(start_row),
    step_rows=int(step_rows),
)

st.subheader("Simulation Results")
if sim_df.empty:
    st.warning("No rows simulated. Try reducing years or start_row, or ensure the dataset has enough rows.")
else:
    final_balance = sim_df["End Balance"].iloc[-1]
    total_withdrawn = sim_df["Withdrawal"].sum()
    years_run = int(sim_df["Year"].iloc[-1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Years Simulated", f"{years_run}")
    m2.metric("Total Withdrawn (Real $)", f"${total_withdrawn:,.0f}")
    m3.metric("Final Balance (Real $)", f"${final_balance:,.0f}")

    st.dataframe(
        sim_df.style.format({
            "Begin Balance": "${:,.0f}",
            "Withdrawal": "${:,.0f}",
            "Real Factor": "{:.6f}",
            "End Balance": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.line_chart(sim_df.set_index("Year")[
        ["Begin Balance", "End Balance"]
    ])

    csv = sim_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="real_withdrawal_sim_results.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Because factors are **real**, the withdrawal stays constant to represent constant purchasing power.")
