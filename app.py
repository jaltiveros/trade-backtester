import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Consistency Matrix 2026 Pro", layout="wide")

if 'confirmed_df' not in st.session_state:
    st.session_state.confirmed_df = None
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# --- 2. DATA LOADING & CACHING ---
@st.cache_data(ttl=300) 
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, "trading_data.parquet")
    db_path = os.path.join(base_dir, "Trading-Database.db")
    
    if not os.path.exists(parquet_path):
        st.error("❌ 'trading_data.parquet' not found.")
        return pd.DataFrame(), [], "N/A"
    
    try:
        mtime = os.path.getmtime(parquet_path)
        last_sync = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        df = pd.read_parquet(parquet_path)
        
        with sqlite3.connect(db_path) as conn:
            try:
                fomc_query = "SELECT DISTINCT Event_Date FROM CalendarEvents WHERE Name LIKE '%FOMC%'"
                fomc_dates = pd.to_datetime(pd.read_sql(fomc_query, conn)['Event_Date']).dt.normalize().tolist()
            except:
                fomc_dates = []
        
        df.columns = [c.strip().replace(' ', '_').replace('.', '').replace('/', '') for c in df.columns]
        df['Date_Opened'] = pd.to_datetime(df['Date_Opened'], errors='coerce')
        df = df.dropna(subset=['Date_Opened'])
        df['Day_Name'] = df['Date_Opened'].dt.day_name().astype('category')
        df['PL'] = df['PL'].astype('float32')
        return df, fomc_dates, last_sync
    except Exception as e: 
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], "Error"

df_raw, fomc_blacklist, last_sync_time = load_data()

# --- 3. SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.info(f"📁 **Data Last Synced:**\n{last_sync_time}")

profiles = {
    "Custom (Stability Focus)": [5, 15, 40, 30, 10],
    "Robustness (Recommended)": [15, 25, 30, 30, 0],
    "Adaptive (Recent Momentum)": [40, 30, 20, 10, 0],
    "Institutional (Max History)": [10, 15, 25, 25, 25]
}

with st.sidebar:
    st.subheader("📅 Date Range")
    lookback_mode = st.radio("Selection Mode", ["Presets", "Custom Range"], horizontal=True)
    db_min, db_max = (df_raw['Date_Opened'].min(), df_raw['Date_Opened'].max()) if not df_raw.empty else (pd.Timestamp.now(), pd.Timestamp.now())
    
    if lookback_mode == "Presets":
        period = st.selectbox("Lookback", ["3M", "6M", "12M", "24M", "All Time"], index=4)
        start_date = db_min if period == "All Time" else max(pd.Timestamp(db_max) - relativedelta(months={"3M":3,"6M":6,"12M":12,"24M":24}.get(period,0)), pd.Timestamp("1678-01-01"))
        end_date = db_max
    else:
        date_range = st.date_input("Pick Range", value=(db_min.to_pydatetime(), db_max.to_pydatetime()))
        start_date, end_date = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) if len(date_range) == 2 else (db_min, db_max)

    st.divider()
    selected_profile = st.selectbox("Choose a Profile", list(profiles.keys()))
    p_vals = profiles[selected_profile]
    w_3m, w_6m, w_12m, w_24m, w_all = [st.slider(f"{k} PF Weight", 0, 100, v) for k, v in zip(["3M", "6M", "12M", "24M", "All-Time"], p_vals)]

    with st.form("filter_form"):
        st.subheader("🎯 Ranking & News")
        filter_mode = st.radio("Filter Logic:", ["Top X Results", "Min Weighted PF"])
        top_x_val = st.number_input("Show Top X", value=10)
        min_pf_val = st.number_input("Min Weighted PF", value=1.0)
        min_trades = st.number_input("Min Trades", value=5)
        avoid_fomc = st.toggle("Avoid FOMC Days", value=True)
        sel_strats = st.multiselect("Strategies", sorted(df_raw['Strategy'].unique()), default=sorted(df_raw['Strategy'].unique()))
        acc_size, multiplier = st.number_input("Account Size", value=50000.0), st.number_input("Multiplier", value=1)
        time_buffer = st.slider("Buffer (Mins)", 0, 120, 15)
        st.form_submit_button("Apply Changes")

# --- 4. HELPERS ---
def calc_pf(series):
    if series.empty: return 0.0
    w, l = series[series > 0].sum(), abs(series[series <= 0].sum())
    return w/l if l > 0 else (5.0 if w > 0 else 1.0)

def calc_inst_metrics(series, days_span):
    if series.empty or days_span <= 0: return (0.0, 0.0)
    total_pl = series.sum()
    equity = series.cumsum()
    peak = equity.cummax()
    drawdown = (peak - equity).max()
    
    years = max(days_span / 365.25, 0.1)
    cagr = total_pl / years
    mar = cagr / drawdown if drawdown > 0 else (cagr if cagr > 0 else 0.0)
    return (mar, cagr)

def style_mtx(df):
    if df.empty: return df
    color_cols = [c for c in df.columns if any(x in c for x in ["PF", "Score", "MAR", "Calmar"])]
    styled = df.style.background_gradient(subset=color_cols, cmap='RdYlGn', vmin=0.7, vmax=2.2).format({c: "{:.2f}" for c in color_cols})
    if 'Weighted Score' in df.columns: styled = styled.set_properties(subset=['Weighted Score'], **{'font-weight': 'bold'})
    return styled

@st.cache_data
def get_mtx(df, ref_end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades):
    if df.empty: return df
    
    m_all = df.groupby(['Strategy', 'Time_Opened'], observed=False)['PL'].apply(calc_pf).reset_index().rename(columns={'PL': 'PF (All)'})
    
    # 1. MAR (All History)
    full_span_days = (df['Date_Opened'].max() - df['Date_Opened'].min()).days
    mar_data = df.groupby(['Strategy', 'Time_Opened'], observed=False)['PL'].apply(lambda x: calc_inst_metrics(x, full_span_days)).reset_index()
    m_all['MAR (All)'] = mar_data['PL'].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else 0.0)

    # 2. Calmar (Trailing 36 Months)
    calmar_cutoff = ref_end_date - relativedelta(months=36)
    df_36m = df[df['Date_Opened'] >= calmar_cutoff]
    if not df_36m.empty:
        cal_data = df_36m.groupby(['Strategy', 'Time_Opened'], observed=False)['PL'].apply(lambda x: calc_inst_metrics(x, 36*30.44)).reset_index()
        m_all = m_all.merge(cal_data.rename(columns={'PL': 'Cal_Raw'}), on=['Strategy', 'Time_Opened'], how='left')
        m_all['Calmar (36M)'] = m_all['Cal_Raw'].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else 0.0)
        m_all = m_all.drop(columns=['Cal_Raw'])
    else:
        m_all['Calmar (36M)'] = 0.0

    for l, m in [("24M", 24), ("12M", 12), ("6M", 6), ("3M", 3)]:
        sub = df[(df['Date_Opened'] >= max(ref_end_date - relativedelta(months=m), pd.Timestamp("1678-01-01"))) & (df['Date_Opened'] <= ref_end_date)]
        m_all = m_all.merge(sub.groupby(['Strategy', 'Time_Opened'], observed=False)['PL'].apply(calc_pf).reset_index().rename(columns={'PL': f'PF ({l})'}), on=['Strategy', 'Time_Opened'], how='left')
    
    num_cols = m_all.select_dtypes(include=[np.number]).columns
    m_all[num_cols] = m_all[num_cols].fillna(1.0)
    m_all['Weighted Score'] = (m_all['PF (3M)']*(w_3m/100)) + (m_all['PF (6M)']*(w_6m/100)) + (m_all['PF (12M)']*(w_12m/100)) + (m_all['PF (24M)']*(w_24m/100)) + (m_all['PF (All)']*(w_all/100))
    final = m_all.merge(df.groupby(['Strategy', 'Time_Opened'], observed=False).size().reset_index(name='Total Trades'), on=['Strategy', 'Time_Opened'])
    final['Confidence'] = final['Total Trades'].apply(lambda x: "🟢 High" if x>=20 else ("🟡 Med" if x>=10 else "🔴 Low"))
    return final[final['Total Trades'] >= min_trades].sort_values('Weighted Score', ascending=False).rename(columns={'Time_Opened': 'Time'})

def get_diverse_picks(df, buffer_mins, target_count, filter_mode, threshold):
    if df.empty: return df
    df = df.copy().sort_values('Weighted Score', ascending=False)
    df['temp_time'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S')
    selected = []
    for _, row in df.iterrows():
        if filter_mode == "Top X Results" and len(selected) >= target_count: break
        if filter_mode == "Min Weighted PF" and row['Weighted Score'] < threshold: continue
        if not any(row['Strategy'] == s['Strategy'] and abs((row['temp_time'] - s['temp_time']).total_seconds()/60) < buffer_mins for s in selected):
            selected.append(row)
    return pd.DataFrame(selected).drop(columns=['temp_time']) if selected else pd.DataFrame()

# --- 5. DATA FILTERING ---
fomc_set = set(pd.to_datetime(fomc_blacklist).normalize())
working_df = df_raw[(df_raw['Strategy'].isin(sel_strats)) & (df_raw['Date_Opened'] >= start_date) & (df_raw['Date_Opened'] <= end_date)].copy()
if avoid_fomc:
    working_df = working_df[~working_df['Date_Opened'].dt.normalize().isin(fomc_set)]

# --- 6. PRE-CALCULATION ---
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
master_pool = get_mtx(working_df, end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades)
day_pools = {day: get_mtx(working_df[working_df['Day_Name'] == day], end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades) for day in days}

# --- 7. UI SELECTION ---
@st.fragment
def render_selection_ui():
    st.title("🛡️ Consistency Matrix")
    all_selections = []
    k_suffix = f"_res_{st.session_state.reset_counter}"
    display_cols = ['Strategy', 'Time', 'Weighted Score', 'PF (3M)', 'PF (6M)', 'PF (12M)', 'PF (24M)', 'PF (All)', 'MAR (All)', 'Calmar (36M)', 'Confidence', 'Total Trades']

    with st.expander("🏆 Global Top Picks", expanded=True):
        g_tabs = st.tabs(["Overview"] + [d[:3] for d in days])
        with g_tabs[0]:
            g_ov = get_diverse_picks(master_pool, time_buffer, top_x_val, filter_mode, min_pf_val)
            if not g_ov.empty:
                evt_g = st.dataframe(style_mtx(g_ov[display_cols]), width='stretch', hide_index=True, on_select="rerun", key=f"g_ov{k_suffix}")
                if evt_g.selection.rows: all_selections.append(g_ov.iloc[evt_g.selection.rows])
        for i, day in enumerate(days):
            with g_tabs[i+1]:
                d_picks = get_diverse_picks(day_pools[day], time_buffer, top_x_val, filter_mode, min_pf_val)
                if not d_picks.empty:
                    evt_gd = st.dataframe(style_mtx(d_picks[display_cols]), width='stretch', hide_index=True, on_select="rerun", key=f"g_{day}{k_suffix}")
                    if evt_gd.selection.rows: 
                        sel = d_picks.iloc[evt_gd.selection.rows].copy(); sel['Day_Name'] = day; all_selections.append(sel)

    st.write("### 📁 Strategy Breakdown")
    for s_name in sel_strats:
        s_ov = master_pool[master_pool['Strategy'] == s_name].head(top_x_val)
        if not s_ov.empty:
            with st.expander(f"📂 {s_name}"):
                s_tabs = st.tabs(["Overview"] + [d[:3] for d in days])
                s_disp = [c for c in display_cols if c != 'Strategy']
                with s_tabs[0]:
                    evt_s = st.dataframe(style_mtx(s_ov[s_disp]), width='stretch', hide_index=True, on_select="rerun", key=f"s_{s_name}{k_suffix}")
                    if evt_s.selection.rows: all_selections.append(s_ov.iloc[evt_s.selection.rows])
                for i, day in enumerate(days):
                    with s_tabs[i+1]:
                        d_pool_s = day_pools[day][day_pools[day]['Strategy'] == s_name].head(top_x_val)
                        if not d_pool_s.empty:
                            evt_sd = st.dataframe(style_mtx(d_pool_s[s_disp]), width='stretch', hide_index=True, on_select="rerun", key=f"sd_{s_name}_{day}{k_suffix}")
                            if evt_sd.selection.rows:
                                sel = d_pool_s.iloc[evt_sd.selection.rows].copy(); sel['Day_Name'] = day; all_selections.append(sel)

    if all_selections:
        queue_df = pd.concat(all_selections).drop_duplicates(subset=['Strategy', 'Time', 'Day_Name'] if 'Day_Name' in pd.concat(all_selections).columns else ['Strategy', 'Time']).copy()
        if 'Day_Name' not in queue_df.columns: queue_df['Day_Name'] = "All"
        
        st.divider()
        st.subheader("📍 Selection Queue")
        q_col1, q_col2 = st.columns([4, 1])
        with q_col1: st.dataframe(queue_df[['Strategy', 'Day_Name', 'Time', 'Weighted Score', 'Confidence', 'Total Trades']], width='stretch', hide_index=True)
        with q_col2:
            if st.button("🚀 CONFIRM", width='stretch', type="primary"):
                mask = pd.Series(False, index=df_raw.index)
                for _, sel in queue_df.iterrows():
                    m = (df_raw['Strategy'] == sel['Strategy']) & (df_raw['Time_Opened'] == sel['Time'])
                    if sel['Day_Name'] != "All": m &= (df_raw['Day_Name'] == sel['Day_Name'])
                    mask |= m
                st.session_state.confirmed_df = df_raw[mask].copy()
                st.rerun() 
            if st.button("🗑️ CLEAR", width='stretch'):
                st.session_state.reset_counter += 1
                st.session_state.confirmed_df = None
                st.rerun()

render_selection_ui()

# --- 8. INSTITUTIONAL ANALYTICS & LOGS ---
if st.session_state.get('confirmed_df') is not None:
    df_f = st.session_state.confirmed_df.copy().sort_values(['Date_Opened', 'Time_Opened'])
    fomc_impact_df = df_f[df_f['Date_Opened'].dt.normalize().isin(fomc_set)]
    pl_avoided = (fomc_impact_df['PL'] * multiplier).sum()
    
    if avoid_fomc:
        df_f = df_f[~df_f['Date_Opened'].dt.normalize().isin(fomc_set)]

    df_f['EC'] = acc_size + (df_f['PL'] * multiplier).cumsum()
    df_f['Peak'] = df_f['EC'].cummax()
    df_f['Drawdown'] = df_f['EC'] - df_f['Peak']
    df_f['Drawdown_Pct'] = (df_f['Drawdown'] / df_f['Peak']) * 100
    
    max_dd_pct, max_recovery = df_f['Drawdown_Pct'].min(), 0
    max_dd_cash = df_f['Drawdown'].min()
    
    if (df_f['Drawdown'] < 0).any():
        is_in_dd = df_f['Drawdown'] < 0
        dd_groups = (is_in_dd != is_in_dd.shift()).cumsum()
        max_recovery = df_f[is_in_dd].groupby(dd_groups)['Date_Opened'].apply(lambda x: (x.max() - x.min()).days).max()

    p_span = max((df_f['Date_Opened'].max() - df_f['Date_Opened'].min()).days, 1)
    p_mar, p_cagr_cash = calc_inst_metrics(df_f['PL'] * multiplier, p_span)
    p_cagr_pct = (p_cagr_cash / acc_size) * 100

    wins, losses = df_f[df_f['PL'] > 0]['PL']*multiplier, df_f[df_f['PL'] <= 0]['PL']*multiplier
    win_rate = len(wins)/len(df_f) if not df_f.empty else 0
    expectancy = (win_rate * wins.mean()) - ((1-win_rate) * abs(losses.mean())) if not df_f.empty else 0

    st.divider()
    st.subheader("🛡️ Portfolio Analytics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit", f"${(df_f['PL']*multiplier).sum():,.2f}")
    m2.metric("Portfolio PF", f"{calc_pf(df_f['PL']):.2f}")
    m3.metric("Annual CAGR ($)", f"${p_cagr_cash:,.2f}")
    m4.metric("Annual CAGR (%)", f"{p_cagr_pct:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Portfolio MAR/Calmar", f"{p_mar:.2f}")
    m6.metric("Max Drawdown ($)", f"${max_dd_cash:,.2f}")
    m7.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}%")
    m8.metric("Recovery Time", f"{max_recovery} Days")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f['Date_Opened'], y=df_f['EC'], name="Equity", line=dict(color="#00FFCC")))
    fig.add_trace(go.Scatter(x=df_f['Date_Opened'], y=df_f['Drawdown'], name="Drawdown", fill='tozeroy', line=dict(color="rgba(255,76,76,0.4)"), yaxis="y2"))
    fig.update_layout(template="plotly_dark", height=400, yaxis=dict(title="Balance"), yaxis2=dict(overlaying="y", side="right", showgrid=False), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    st.write("#### 📅 Monthly Performance Matrix")
    df_f['Year'] = df_f['Date_Opened'].dt.year
    df_f['Month'] = df_f['Date_Opened'].dt.month_name()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_f['Month'] = pd.Categorical(df_f['Month'], categories=months, ordered=True)
    
    pivot = df_f.pivot_table(index='Year', columns='Month', values='PL', aggfunc=lambda x: (x*multiplier).sum(), observed=False).fillna(0)
    pivot['Total'] = pivot.sum(axis=1)
    
    avg_row = pivot.mean(axis=0).to_frame().T
    avg_row.index = ["Average"]
    pivot.index = pivot.index.astype(str)
    final_pivot = pd.concat([pivot, avg_row])
    
    max_abs = max(abs(pivot[months].values.min()), abs(pivot[months].values.max()), 1)
    st.dataframe(
        final_pivot.style.background_gradient(cmap='RdYlGn', vmin=-max_abs, vmax=max_abs, subset=(pivot.index, months), axis=None)
        .format("${:,.2f}")
        .set_properties(subset=['Total'], **{'font-weight': 'bold', 'background-color': '#262730'})
        .set_properties(subset=pd.IndexSlice[["Average"], :], **{'font-weight': 'bold', 'background-color': '#1a1c23', 'color': '#00FFCC'}), 
        width='stretch'
    )

    st.write("#### 📜 Detailed Trade Log")
    available_cols = df_f.columns.tolist()
    log_cols = ['Date_Opened', 'Time_Opened', 'Strategy', 'PL']
    if 'Legs' in available_cols: log_cols.insert(3, 'Legs')
    
    log_df = df_f[log_cols].copy()
    log_df['PL'] = log_df['PL'] * multiplier
    log_df['Date_Opened'] = log_df['Date_Opened'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        log_df.style.map(lambda x: 'color: #00FFCC' if isinstance(x, (int, float)) and x > 0 
                                else ('color: #FF4C4C' if isinstance(x, (int, float)) and x < 0 else ''), subset=['PL'])
        .format({'PL': "${:,.2f}"}),
        width='stretch',
        hide_index=True
    )