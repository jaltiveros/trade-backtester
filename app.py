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
    "Custom (Stability Focus)": [0, 20, 45, 30, 5],
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
    st.subheader("⚖️ Metric Timeframe Weights")
    selected_profile = st.selectbox("Choose a Profile", list(profiles.keys()))
    p_vals = profiles[selected_profile]
    w_3m, w_6m, w_12m, w_24m, w_all = [st.slider(f"{k} Weight (%)", 0, 100, v) for k, v in zip(["3M", "6M", "12M", "24M", "All-Time"], p_vals)]
    total_tw = w_3m + w_6m + w_12m + w_24m + w_all
    if total_tw != 100:
        st.warning(f"Total weight: {total_tw}% (Should be 100%)")

    st.divider()
    st.subheader("🏆 Total Score Weighting")
    sw_sortino = st.slider("Sortino Weight", 0, 100, 10)
    sw_pf = st.slider("PF Weight", 0, 100, 75)
    sw_calmar = st.slider("Calmar Weight", 0, 100, 15)
    total_sw = sw_sortino + sw_pf + sw_calmar
    if total_sw != 100:
        st.warning(f"Total weight: {total_sw}% (Should be 100%)")

    with st.form("filter_form"):
        st.subheader("🎯 Ranking & News")
        rank_by = st.selectbox("Rank By:", ["Total Score", "Weighted PF", "Weighted Sortino", "MAR (All)", "Calmar (36M)"])
        
        f_c1, f_c2 = st.columns(2)
        top_x_val = f_c1.number_input("Show Top X", value=25, min_value=1)
        min_trades = f_c2.number_input("Min Trades", value=5, min_value=1)
        
        min_pf_val = f_c1.number_input("Min Weighted PF", value=0.0, step=0.1)
        min_sr_val = f_c2.number_input("Min Weighted Sortino", value=0.0, step=0.1)
        min_mar_val = f_c1.number_input("Min MAR (All)", value=0.0, step=0.1)
        min_calmar_val = f_c2.number_input("Min Calmar (36M)", value=0.0, step=0.1)
        
        avoid_fomc = st.toggle("Avoid FOMC Days", value=True)
        fomc_only = st.toggle("FOMC Days Only", value=False)
        fomc_before = st.toggle("Day Before FOMC", value=False)
        fomc_after = st.toggle("Day After FOMC", value=False)
        
        sel_strats = st.multiselect("Strategies", sorted(df_raw['Strategy'].unique()), default=sorted(df_raw['Strategy'].unique()))
        acc_size, multiplier = st.number_input("Account Size", value=150000.0), st.number_input("Contracts", value=1)
        time_buffer = st.slider("Buffer (Mins)", 0, 120, 10)
        st.form_submit_button("Apply Changes")

# --- 4. HELPERS ---
def calc_pf(series):
    if series.empty: return 0.0
    w, l = series[series > 0].sum(), abs(series[series <= 0].sum())
    return w/l if l > 0 else (5.0 if w > 0 else 1.0)

def calc_sortino(series):
    if series.empty: return 0.0
    mean_ret = series.mean()
    downside = series[series < 0]
    if downside.empty: return 5.0 if mean_ret > 0 else 0.0
    std_downside = np.std(downside)
    return mean_ret / std_downside if std_downside > 0 else (5.0 if mean_ret > 0 else 0.0)

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
    
    pf_cols = [c for c in df.columns if "PF" in c]
    sr_cols = [c for c in df.columns if "Sortino" in c]
    mar_cols = [c for c in df.columns if "MAR" in c or "Calmar" in c]
    score_cols = [c for c in df.columns if "Score" in c]
    
    styled = df.style.background_gradient(subset=pf_cols, cmap='RdYlGn', vmin=0.8, vmax=2.2)
    styled = styled.background_gradient(subset=sr_cols, cmap='RdYlGn', vmin=0.5, vmax=3.0)
    styled = styled.background_gradient(subset=mar_cols, cmap='RdYlGn', vmin=0.1, vmax=1.2)
    styled = styled.background_gradient(subset=score_cols, cmap='Blues', vmin=1.0, vmax=4.0)

    format_dict = {c: "{:.2f}" for c in pf_cols + sr_cols + mar_cols + score_cols}
    styled = styled.format(format_dict)
    
    if 'Total Score' in df.columns:
        styled = styled.set_properties(subset=['Total Score'], **{'font-weight': 'bold', 'font-size': '1.1em'})
    if 'Weighted PF' in df.columns: 
        styled = styled.set_properties(subset=['Weighted PF'], **{'font-weight': 'bold'})
    if 'Weighted Sortino' in df.columns: 
        styled = styled.set_properties(subset=['Weighted Sortino'], **{'font-weight': 'bold'})
        
    return styled

@st.cache_data
def get_mtx(df, ref_end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades, sort_col, sw_pf, sw_sortino, sw_calmar):
    if df.empty: return pd.DataFrame()
    
    grouped = df.groupby(['Strategy', 'Time_Opened'], observed=False)['PL']
    m_all = grouped.apply(calc_pf).reset_index().rename(columns={'PL': 'PF (All)'})
    m_all['Sortino (All)'] = grouped.apply(calc_sortino).values

    full_span_days = (df['Date_Opened'].max() - df['Date_Opened'].min()).days
    mar_data = grouped.apply(lambda x: calc_inst_metrics(x, full_span_days)).reset_index()
    m_all['MAR (All)'] = mar_data['PL'].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else 0.0)

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
        sub_grouped = sub.groupby(['Strategy', 'Time_Opened'], observed=False)['PL']
        m_all = m_all.merge(sub_grouped.apply(calc_pf).reset_index().rename(columns={'PL': f'PF ({l})'}), on=['Strategy', 'Time_Opened'], how='left')
        m_all = m_all.merge(sub_grouped.apply(calc_sortino).reset_index().rename(columns={'PL': f'Sortino ({l})'}), on=['Strategy', 'Time_Opened'], how='left')
    
    num_cols = m_all.select_dtypes(include=[np.number]).columns
    m_all[num_cols] = m_all[num_cols].fillna(1.0)
    
    weights = [w_3m/100, w_6m/100, w_12m/100, w_24m/100, w_all/100]
    m_all['Weighted PF'] = sum(m_all[c] * w for c, w in zip(['PF (3M)', 'PF (6M)', 'PF (12M)', 'PF (24M)', 'PF (All)'], weights))
    m_all['Weighted Sortino'] = sum(m_all[c] * w for c, w in zip(['Sortino (3M)', 'Sortino (6M)', 'Sortino (12M)', 'Sortino (24M)', 'Sortino (All)'], weights))
    
    # --- TOTAL SCORE CALCULATION ---
    # Normalize weights to ensure they sum to 1.0 even if sliders don't
    sum_w = max(sw_pf + sw_sortino + sw_calmar, 1)
    m_all['Total Score'] = (
        (m_all['Weighted PF'] * (sw_pf / sum_w)) +
        (m_all['Weighted Sortino'] * (sw_sortino / sum_w)) +
        (m_all['Calmar (36M)'] * (sw_calmar / sum_w))
    )
    
    final = m_all.merge(df.groupby(['Strategy', 'Time_Opened'], observed=False).size().reset_index(name='Total Trades'), on=['Strategy', 'Time_Opened'])
    final['Confidence'] = final['Total Trades'].apply(lambda x: "🟢 High" if x>=20 else ("🟡 Med" if x>=10 else "🔴 Low"))
    
    return final[final['Total Trades'] >= min_trades].sort_values(sort_col, ascending=False).rename(columns={'Time_Opened': 'Time'})

def get_diverse_picks(df, buffer_mins, target_count, min_pf, min_sr, min_mar, min_calmar, sort_col):
    if df is None or df.empty: return pd.DataFrame()
    
    required = ['Weighted PF', 'Weighted Sortino', 'MAR (All)', 'Calmar (36M)', 'Total Score']
    if not all(col in df.columns for col in required): return pd.DataFrame()

    filtered_df = df[
        (df['Weighted PF'] >= min_pf) & 
        (df['Weighted Sortino'] >= min_sr) &
        (df['MAR (All)'] >= min_mar) & 
        (df['Calmar (36M)'] >= min_calmar)
    ].copy()
    
    if filtered_df.empty: return pd.DataFrame()
    
    filtered_df = filtered_df.sort_values(sort_col, ascending=False)
    filtered_df['temp_time'] = pd.to_datetime(filtered_df['Time'].astype(str), format='%H:%M:%S')
    selected = []
    for _, row in filtered_df.iterrows():
        if len(selected) >= target_count: break
        if not any(row['Strategy'] == s['Strategy'] and abs((row['temp_time'] - s['temp_time']).total_seconds()/60) < buffer_mins for s in selected):
            selected.append(row)
##--    return pd.DataFrame(selected).drop(columns=['temp_time']) if selected else pd.DataFrame()

# Convert list back to DF and sort by Time chronologically for the UI
    if not selected: return pd.DataFrame()
    res_df = pd.DataFrame(selected).drop(columns=['temp_time'])
    return res_df.sort_values('Time')

# --- 5. DATA FILTERING ---
fomc_set = set(pd.to_datetime(fomc_blacklist).normalize())
pre_fomc_set = {d - pd.Timedelta(days=1) for d in fomc_set}
post_fomc_set = {d + pd.Timedelta(days=1) for d in fomc_set}

working_df = df_raw[(df_raw['Strategy'].isin(sel_strats)) & (df_raw['Date_Opened'] >= start_date) & (df_raw['Date_Opened'] <= end_date)].copy()

if avoid_fomc: working_df = working_df[~working_df['Date_Opened'].dt.normalize().isin(fomc_set)]

if fomc_only or fomc_before or fomc_after:
    target_dates = set()
    if fomc_only: target_dates.update(fomc_set)
    if fomc_before: target_dates.update(pre_fomc_set)
    if fomc_after: target_dates.update(post_fomc_set)
    working_df = working_df[working_df['Date_Opened'].dt.normalize().isin(target_dates)]

# --- 6. PRE-CALCULATION ---
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
master_pool = get_mtx(working_df, end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades, rank_by, sw_pf, sw_sortino, sw_calmar)
day_pools = {day: get_mtx(working_df[working_df['Day_Name'] == day], end_date, w_3m, w_6m, w_12m, w_24m, w_all, min_trades, rank_by, sw_pf, sw_sortino, sw_calmar) for day in days}

# --- 7. UI SELECTION ---
@st.fragment
def render_selection_ui():
    st.title("🛡️ Consistency Matrix")
    
    if master_pool.empty:
        st.warning("⚠️ No trades found matching these filters.")
        return

    st.subheader("⚠️ Estimated Risk of Double Stop")
    risk_cols = st.columns(min(len(sel_strats), 5))
    for i, s_name in enumerate(sorted(sel_strats)):
        s_data = working_df[working_df['Strategy'] == s_name]
        losses = s_data[s_data['PL'] <= 0]['PL']
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        risk_factor = 2 if "IC" in s_name else 1
        raw_risk = (avg_loss * multiplier * risk_factor)
        double_stop = (np.ceil(raw_risk / 50) * 50)
        risk_cols[i % 5].metric(f"{s_name}", f"${double_stop:,.2f}")
    
    st.divider()
    
    all_selections = []
    k_suffix = f"_res_{st.session_state.reset_counter}"
    
    display_cols = [
        'Strategy', 'Time', 'Total Score', 'Weighted PF', 'Weighted Sortino', 'Calmar (36M)',
##--        'PF (3M)', 'Sortino (3M)',
        'PF (6M)', 'Sortino (6M)', 
        'PF (12M)', 'Sortino (12M)', 'PF (24M)', 'Sortino (24M)', 
        'PF (All)', 'Sortino (All)', 'MAR (All)', 
        'Confidence', 'Total Trades'
    ]

    with st.expander("🏆 Global Top Picks", expanded=True):
        g_tabs = st.tabs(["Overview"] + [d[:3] for d in days])
        with g_tabs[0]:
            g_ov = get_diverse_picks(master_pool, time_buffer, top_x_val, min_pf_val, min_sr_val, min_mar_val, min_calmar_val, rank_by)
            if not g_ov.empty:
                evt_g = st.dataframe(style_mtx(g_ov[display_cols]), width='stretch', hide_index=True, on_select="rerun", key=f"g_ov{k_suffix}")
                if evt_g.selection.rows: all_selections.append(g_ov.iloc[evt_g.selection.rows])
        
        for i, day in enumerate(days):
            with g_tabs[i+1]:
                if not day_pools[day].empty:
                    d_picks = get_diverse_picks(day_pools[day], time_buffer, top_x_val, min_pf_val, min_sr_val, min_mar_val, min_calmar_val, rank_by)
                    if not d_picks.empty:
                        evt_gd = st.dataframe(style_mtx(d_picks[display_cols]), width='stretch', hide_index=True, on_select="rerun", key=f"g_{day}{k_suffix}")
                        if evt_gd.selection.rows: 
                            sel = d_picks.iloc[evt_gd.selection.rows].copy(); sel['Day_Name'] = day; all_selections.append(sel)

    st.write("### 📁 Strategy Breakdown")
    for s_name in sel_strats:
        s_base = master_pool[master_pool['Strategy'] == s_name]
        if s_base.empty: continue
            
        s_ov_filtered = s_base[
            (s_base['Weighted PF'] >= min_pf_val) & 
            (s_base['Weighted Sortino'] >= min_sr_val) &
            (s_base['MAR (All)'] >= min_mar_val) & 
            (s_base['Calmar (36M)'] >= min_calmar_val)
        ].head(top_x_val).sort_values('Time')
        
        day_results = {}
        has_any_day_data = False
        for day in days:
            d_pool = day_pools[day][day_pools[day]['Strategy'] == s_name] if not day_pools[day].empty else pd.DataFrame()
            if not d_pool.empty:
                d_filtered = d_pool[
                    (d_pool['Weighted PF'] >= min_pf_val) & 
                    (d_pool['Weighted Sortino'] >= min_sr_val) &
                    (d_pool['MAR (All)'] >= min_mar_val) & 
                    (d_pool['Calmar (36M)'] >= min_calmar_val)
                ].head(top_x_val).sort_values('Time')
                day_results[day] = d_filtered
                if not d_filtered.empty: has_any_day_data = True
            else:
                day_results[day] = pd.DataFrame()

        if not s_ov_filtered.empty or has_any_day_data:
            with st.expander(f"📂 {s_name}"):
                s_tabs = st.tabs(["Overview"] + [d[:3] for d in days])
                s_disp = [c for c in display_cols if c != 'Strategy']
                
                with s_tabs[0]:
                    if not s_ov_filtered.empty:
                        evt_s = st.dataframe(style_mtx(s_ov_filtered[s_disp]), width='stretch', hide_index=True, on_select="rerun", key=f"s_{s_name}{k_suffix}")
                        if evt_s.selection.rows: all_selections.append(s_ov_filtered.iloc[evt_s.selection.rows])
                    else:
                        st.info("No trades meet 'Overview' filters.")

                for i, day in enumerate(days):
                    with s_tabs[i+1]:
                        if day in day_results and not day_results[day].empty:
                            evt_sd = st.dataframe(style_mtx(day_results[day][s_disp]), width='stretch', hide_index=True, on_select="rerun", key=f"sd_{s_name}_{day}{k_suffix}")
                            if evt_sd.selection.rows:
                                sel = day_results[day].iloc[evt_sd.selection.rows].copy(); sel['Day_Name'] = day; all_selections.append(sel)
                        else:
                            st.info(f"No trades meet filters for {day}.")

    if all_selections:
        queue_df = pd.concat(all_selections).drop_duplicates(subset=['Strategy', 'Time', 'Day_Name'] if 'Day_Name' in pd.concat(all_selections).columns else ['Strategy', 'Time']).copy()
        if 'Day_Name' not in queue_df.columns: queue_df['Day_Name'] = "All"
        st.divider()
        st.subheader("📍 Selection Queue")
        q_col1, q_col2 = st.columns([4, 1])
        with q_col1: st.dataframe(queue_df[['Strategy', 'Day_Name', 'Time', 'Total Score', 'Weighted PF', 'Weighted Sortino', 'Confidence', 'Total Trades']], width='stretch', hide_index=True)
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
    df_f = st.session_state.confirmed_df.copy().reset_index(drop=True)
    df_f.columns = [str(c).strip().replace(' ', '_').replace('.', '').replace('/', '') for c in df_f.columns]
    if 'Strategy' in df_f.columns:
        df_f['Strategy'] = df_f['Strategy'].astype(str)

    df_f = df_f.sort_values(['Date_Opened', 'Time_Opened'])
    
    if avoid_fomc:
        df_f = df_f[~df_f['Date_Opened'].dt.normalize().isin(fomc_set)]
    if fomc_only or fomc_before or fomc_after:
        valid_dates = set()
        if fomc_only: valid_dates.update(fomc_set)
        if fomc_before: valid_dates.update(pre_fomc_set)
        if fomc_after: valid_dates.update(post_fomc_set)
        df_f = df_f[df_f['Date_Opened'].dt.normalize().isin(valid_dates)]

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

    st.divider()
    st.subheader("🛡️ Portfolio Analytics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit", f"${(df_f['PL']*multiplier).sum():,.2f}")
    m2.metric("Portfolio PF", f"{calc_pf(df_f['PL']):.2f}")
    m3.metric("Annual CAGR ($)", f"${p_cagr_cash:,.2f}")
    m4.metric("Annual CAGR (%)", f"{p_cagr_pct:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Portfolio MAR", f"{p_mar:.2f}")
    m6.metric("Max Drawdown ($)", f"${max_dd_cash:,.2f}")
    m7.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}%")
    m8.metric("Recovery Time", f"{max_recovery} Days")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f['Date_Opened'], y=df_f['EC'], name="Equity", line=dict(color="#00FFCC")))
    fig.add_trace(go.Scatter(x=df_f['Date_Opened'], y=df_f['Drawdown'], name="Drawdown", fill='tozeroy', line=dict(color="rgba(255,76,76,0.4)"), yaxis="y2"))
    fig.update_layout(template="plotly_dark", height=400, yaxis=dict(title="Balance"), yaxis2=dict(overlaying="y", side="right", showgrid=False), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    st.subheader("🗓️ Daily Risk Profile")
    st.write("*(Header Risk Estimate × Actual Trade Count in Queue)*")

    reference_risk_map = {}
    for s_name in sel_strats:
        s_data = working_df[working_df['Strategy'] == s_name]
        losses = s_data[s_data['PL'] <= 0]['PL']
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        raw_risk = avg_loss * multiplier * 2 
        reference_risk_map[s_name] = (np.ceil(raw_risk / 50) * 50)

    daily_risk_data = []
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    for day in days_order:
        day_queue = df_f[df_f['Day_Name'] == day]
        if not day_queue.empty:
            unique_selections = day_queue[['Strategy', 'Time_Opened']].drop_duplicates()
            strat_counts = unique_selections['Strategy'].value_counts()
            
            total_day_risk = 0
            strat_summary = []
            for s_name, count in strat_counts.items():
                unit_risk = reference_risk_map.get(s_name, 0)
                total_day_risk += (unit_risk * count)
                strat_summary.append(f"{s_name} (x{count})")
            
            pct_risk = (total_day_risk / acc_size) * 100 if acc_size > 0 else 0
            
            daily_risk_data.append({
                "Day": day,
                "Entries": ", ".join(strat_summary),
                "Combined Risk ($)": total_day_risk,
                " % Risk": pct_risk,
                "Unique Trades": len(unique_selections)
            })

    if daily_risk_data:
        risk_df = pd.DataFrame(daily_risk_data)
        total_risk_val = risk_df['Combined Risk ($)'].sum()
        total_trades_val = risk_df['Unique Trades'].sum()
        total_pct_risk = (total_risk_val / acc_size) * 100 if acc_size > 0 else 0
        
        total_row = pd.DataFrame([{
            "Day": "TOTAL",
            "Entries": "Full Selection",
            "Combined Risk ($)": total_risk_val,
            " % Risk": total_pct_risk,
            "Unique Trades": total_trades_val
        }])
        
        final_risk_df = pd.concat([risk_df, total_row], ignore_index=True)

        st.dataframe(
            final_risk_df.style.background_gradient(
                subset=pd.IndexSlice[final_risk_df.index[:-1], ['Combined Risk ($)', ' % Risk']], 
                cmap='YlOrRd'
            )
            .format({
                'Combined Risk ($)': "${:,.2f}",
                ' % Risk': "{:.2f}%"
            })
            .set_properties(
                subset=pd.IndexSlice[final_risk_df.index[-1:], :], 
                **{'font-weight': 'bold', 'background-color': '#1a1c23', 'color': '#00FFCC'}
            ),
            width='stretch', 
            hide_index=True
        )  

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
    log_cols = ['Date_Opened', 'Time_Opened', 'Strategy', 'PL']
    if 'Legs' in df_f.columns: log_cols.insert(3, 'Legs')
    
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