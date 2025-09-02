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
    start_row = st.number_input("Begin simulation row index", min_value=0, max_value=max(0, len(df) - 1), value=28, step=1)
    step_rows = st.number_input("Row step per year (12 = use next year factor 12 rows down)", min_value=1, max_value=120, value=12, step=1)

# Helper: fast final-balance simulation for a single start row
def simulate_final_balance(
    factors: pd.Series,
    years: int,
    starting_balance: float,
    annual_withdrawal: float,
    start_row: int,
    step_rows: int,
) -> float | None:
    """Return end balance after `years` or None if path is incomplete/NaN.
    Uses begin-of-year withdrawal then applies real factor.
    """
    num_rows = len(factors)
    last_idx = start_row + (years - 1) * step_rows
    if last_idx >= num_rows:
        return None
    balance = float(starting_balance)
    for y in range(years):
        idx = start_row + y * step_rows
        factor = factors.iloc[idx]
        if pd.isna(factor):
            return None
        balance = (balance - annual_withdrawal) * float(factor)
    return balance

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

# --- Batch summary metrics (all start rows) ---
factors_series = df[alloc]
max_start = len(factors_series) - (int(years) - 1) * int(step_rows)
max_start = max(0, max_start)

batch_results = []
for s in range(0, max_start):
    final_bal = simulate_final_balance(
        factors=factors_series,
        years=int(years),
        starting_balance=float(starting_balance),
        annual_withdrawal=float(annual_withdrawal),
        start_row=int(s),
        step_rows=int(step_rows),
    )
    if final_bal is None:
        continue
    batch_results.append(final_bal)

if batch_results:
    starts_evaluated = len(batch_results)
    success_rate = sum(1 for b in batch_results if b > 0) / starts_evaluated
    m4, m5 = st.columns(2)
    m4.metric("Starts evaluated", f"{starts_evaluated:,}")
    m5.metric("% with Final Balance > 0", f"{success_rate*100:,.2f}%")

st.markdown("---")
st.subheader("Batch: run all available starting rows")

st.caption(
    "Runs every valid starting row for the chosen **allocation**, jumping `step_rows` each year, "
    "for the selected number of years. Skips paths that would run off the dataset or contain NaNs."
)

run_all = st.checkbox("Run batch across all starting rows", value=False)

if run_all:
    factors_series = df[alloc]
    max_start = len(factors_series) - (int(years) - 1) * int(step_rows)
    max_start = max(0, max_start)

    results = []
    for s in range(0, max_start):
        final_bal = simulate_final_balance(
            factors=factors_series,
            years=int(years),
            starting_balance=float(starting_balance),
            annual_withdrawal=float(annual_withdrawal),
            start_row=int(s),
            step_rows=int(step_rows),
        )
        if final_bal is None:
            continue
        begin_label = None
        # Try to pull the begin month label if present
        if "begin month" in df.columns:
            begin_label = df.loc[s, "begin month"]
        results.append({
            "Start Row": s,
            "Begin Month": begin_label,
            "Final Balance": final_bal,
            "Success (Final>0)": final_bal > 0,
        })

    if not results:
        st.warning("No valid starting rows found for this horizon.")
    else:
        res_df = pd.DataFrame(results)
        success_rate = res_df["Success (Final>0)"].mean() if len(res_df) else 0.0
        st.metric("Starts evaluated", f"{len(res_df):,}")
        st.metric("% with Final Balance > 0", f"{success_rate*100:,.2f}%")

        # Format and show table
        fmt_df = res_df.copy()
        if "Begin Month" in fmt_df.columns:
            fmt_df["Begin Month"] = fmt_df["Begin Month"].astype(str)
        st.dataframe(
            fmt_df.style.format({
                "Final Balance": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv_all = res_df.to_csv(index=False)
        st.download_button(
            label="Download batch results (CSV)",
            data=csv_all,
            file_name="batch_all_start_rows_results.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Because factors are **real**, the withdrawal stays constant to represent constant purchasing power.")
