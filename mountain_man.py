import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mountain Man Pro Dashboard",
    page_icon="🍺",
    layout="wide"
)

# --- BASE CONSTANTS & ASSUMPTIONS (FROM SPREADSHEET) ---
BASE_YEAR = 2005
PROJECTION_YEARS = list(range(BASE_YEAR + 1, BASE_YEAR + 6)) # 2006 to 2010

BASE_METRICS = {
    'revenue_2005': 50440000, 'cogs_2005': 34803600, 'op_profit_2005': 4640480,
    'lager_volume_2005': 520000, 'price_2005': 97.00, 'cost_lager_2005': 66.93,
    'cost_light_2005': 71.62, 'light_market_2005': 18744303, 'regional_beer_growth': 0.04,
    'sg&a_base': 9583600, 'other_exp_base': 1412320, 'fc_incremental_yr1': 750000,
    'sg&a_incremental': 900000, 'discount_rate': 0.12
}
BASE_METRICS['total_base_fc'] = BASE_METRICS['sg&a_base'] + BASE_METRICS['other_exp_base']
BASE_METRICS['total_launch_fc_yr1'] = BASE_METRICS['total_base_fc'] + BASE_METRICS['fc_incremental_yr1'] + BASE_METRICS['sg&a_incremental']
BASE_METRICS['total_launch_fc_ongoing'] = BASE_METRICS['total_base_fc'] + BASE_METRICS['sg&a_incremental']

# --- HELPER CALCULATION FUNCTION (FROM DASH APP) ---
@st.cache_data # Use Streamlit's caching to speed up recalculations
def run_financial_model(decline_rate, cannibalization_rate, share_gain_bps, price_inflation, cost_inflation):
    """Runs the full financial model and returns key dataframes and metrics."""
    # The main dataframe will now ONLY hold projection years for NPV calculation
    df_proj = pd.DataFrame(index=PROJECTION_YEARS)

    # Projection Loop for 2006-2010
    for i, year in enumerate(PROJECTION_YEARS):
        # Do Nothing Scenario
        lager_vol_dn = BASE_METRICS['lager_volume_2005'] * ((1 - decline_rate) ** (i + 1))
        unit_price = BASE_METRICS['price_2005'] * ((1 + price_inflation) ** (i + 1))
        cost_per_unit_lager = BASE_METRICS['cost_lager_2005'] * ((1 + cost_inflation) ** (i + 1))
        rev_dn = lager_vol_dn * unit_price
        cogs_dn = lager_vol_dn * cost_per_unit_lager
        df_proj.loc[year, 'op_profit_dn'] = rev_dn - cogs_dn - BASE_METRICS['total_base_fc']
        df_proj.loc[year, 'rev_dn'] = rev_dn

        # Launch Scenario
        light_market_size = BASE_METRICS['light_market_2005'] * ((1 + BASE_METRICS['regional_beer_growth']) ** (i + 1))
        light_vol = light_market_size * (share_gain_bps * (i + 1))
        remnant_lager_vol = lager_vol_dn - (light_vol * cannibalization_rate)
        
        rev_lager_launch = remnant_lager_vol * unit_price
        rev_light_launch = light_vol * unit_price
        
        cost_per_unit_light = BASE_METRICS['cost_light_2005'] * ((1 + cost_inflation) ** (i + 1))
        total_cogs_launch = (remnant_lager_vol * cost_per_unit_lager) + (light_vol * cost_per_unit_light)
        total_gp_launch = (rev_lager_launch + rev_light_launch) - total_cogs_launch
        
        total_fc_launch = BASE_METRICS['total_launch_fc_yr1'] if i == 0 else BASE_METRICS['total_launch_fc_ongoing']
        df_proj.loc[year, 'op_profit_launch'] = total_gp_launch - total_fc_launch
        df_proj.loc[year, 'rev_lager_launch'] = rev_lager_launch
        df_proj.loc[year, 'rev_light_launch'] = rev_light_launch

    # *** NPV CALCULATION IS NOW CORRECTLY ON THE PROJECTION DATAFRAME ***
    npv_do_nothing = npf.npv(BASE_METRICS['discount_rate'], df_proj['op_profit_dn'])
    npv_launch = npf.npv(BASE_METRICS['discount_rate'], df_proj['op_profit_launch'])
    npv_difference = npv_launch - npv_do_nothing

    # *** FIX IS HERE: Create the chart dataframe and populate it explicitly ***
    # Create a separate dataframe for charting that includes the 2005 base year
    df_chart = pd.DataFrame(index=[BASE_YEAR] + PROJECTION_YEARS)

    # Populate the 2005 base year data
    df_chart.loc[BASE_YEAR, 'op_profit_dn'] = BASE_METRICS['op_profit_2005']
    df_chart.loc[BASE_YEAR, 'op_profit_launch'] = BASE_METRICS['op_profit_2005']
    df_chart.loc[BASE_YEAR, 'rev_lager_launch'] = BASE_METRICS['revenue_2005']
    df_chart.loc[BASE_YEAR, 'rev_light_launch'] = 0
    df_chart.loc[BASE_YEAR, 'rev_dn'] = BASE_METRICS['revenue_2005']

    # Populate the projection years (2006-2010) column by column
    for col in df_proj.columns:
        df_chart.loc[PROJECTION_YEARS, col] = df_proj[col]
    
    return df_chart, npv_do_nothing, npv_launch, npv_difference

# --- HEADER ---
st.title("🍺 Mountain Man Brewing Co. Financial Dashboard")
st.markdown("A comprehensive tool for strategic scenario planning and sensitivity analysis.")
st.markdown("---")

# --- SIDEBAR FOR CONTROLS ---
st.sidebar.header("🕹️ Strategic Levers")

decline_rate = st.sidebar.slider("Lager 'Status Quo' Decline Rate (%)", 0.0, 5.0, 2.0, 0.5)
cannibalization_rate = st.sidebar.slider("Cannibalization Rate (%)", 0, 100, 25, 1)
share_gain_bps = st.sidebar.slider("Annual Light Mkt Share Gain (bps)", 0, 50, 25, 1)
price_inflation = st.sidebar.slider("Annual Price Inflation (CPI %)", 0.0, 5.0, 2.0, 0.5)
cost_inflation = st.sidebar.slider("Annual Cost Inflation (PPI %)", 0.0, 5.0, 3.0, 0.5)

# --- MODEL EXECUTION ---
df_chart, npv_dn, npv_l, npv_diff = run_financial_model(
    decline_rate / 100.0, cannibalization_rate / 100.0, share_gain_bps / 10000.0,
    price_inflation / 100.0, cost_inflation / 100.0
)

# --- KPI DISPLAY ---
st.header("📊 Key Financial Outcomes")
col1, col2, col3 = st.columns(3)
col1.metric("NPV (Do Nothing)", f"${npv_dn:,.0f}")
col2.metric("NPV (Launch Light)", f"${npv_l:,.0f}")
col3.metric("Incremental NPV of Launch", f"${npv_diff:,.0f}", f"{'+' if npv_diff >= 0 else ''}${npv_diff:,.0f}",
            help="Positive value indicates the launch creates value compared to the status quo.")

st.markdown("---")

# --- MAIN DASHBOARD LAYOUT ---
col_ts, col_sens = st.columns([7, 5]) # 70/50 split for columns

with col_ts:
    st.subheader("Time Series Projections")
    # --- TIME SERIES CHARTS ---
    df_op_melt = df_chart[['op_profit_dn', 'op_profit_launch']].reset_index().melt(id_vars='index', var_name='Scenario', value_name='Operating Profit')
    df_op_melt['Scenario'] = df_op_melt['Scenario'].map({'op_profit_dn': 'Do Nothing', 'op_profit_launch': 'Launch Light'})
    fig_op_profit = px.line(df_op_melt, x='index', y='Operating Profit', color='Scenario', markers=True, title='Operating Profit Projections (2005-2010)', labels={'index': 'Year'}, color_discrete_map={'Do Nothing': 'orange', 'Launch Light': 'royalblue'})
    fig_op_profit.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01))
    st.plotly_chart(fig_op_profit, use_container_width=True)

    df_rev_melt = df_chart[['rev_lager_launch', 'rev_light_launch']].reset_index().melt(id_vars='index', var_name='Source', value_name='Revenue')
    df_rev_melt['Source'] = df_rev_melt['Source'].map({'rev_lager_launch': 'Lager Revenue', 'rev_light_launch': 'Light Revenue'})
    fig_revenue = px.bar(df_rev_melt, x='index', y='Revenue', color='Source', title='Revenue Mix (Launch Scenario)', labels={'index': 'Year'}, color_discrete_map={'Lager Revenue': '#8B0000', 'Light Revenue': '#FFD700'})
    fig_revenue.add_trace(go.Scatter(x=df_chart.index, y=df_chart['rev_dn'], mode='lines', name='Revenue (Do Nothing)', line=dict(color='grey', dash='dot')))
    fig_revenue.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig_revenue, use_container_width=True)

with col_sens:
    st.subheader("Sensitivity Analysis")
    # --- SENSITIVITY ANALYSIS CHART ---
    cann_range = np.arange(0, 1.01, 0.02)
    sensitivity_data = []
    # Re-run model only for sensitivity calculation
    for rate in cann_range:
        _, _, _, npv_diff_sens = run_financial_model(
            decline_rate / 100.0, rate, share_gain_bps / 10000.0,
            price_inflation / 100.0, cost_inflation / 100.0
        )
        sensitivity_data.append({'Cannibalization Rate': rate * 100, 'Incremental NPV': npv_diff_sens})
    
    sens_df = pd.DataFrame(sensitivity_data)
    fig_sensitivity = px.area(sens_df, x='Cannibalization Rate', y='Incremental NPV', title='NPV Sensitivity to Cannibalization')
    fig_sensitivity.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="NPV Break-Even")
    fig_sensitivity.add_vline(x=cannibalization_rate, line_dash="dash", line_color="black", annotation_text="Current")
    fig_sensitivity.update_traces(hovertemplate="Cannibalization: %{x:.1f}%<br>Incremental NPV: %{y:$,.0f}")

    st.plotly_chart(fig_sensitivity, use_container_width=True)
