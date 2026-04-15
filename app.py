import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

st.title("Stock Comparison and Analysis App")

st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Enter ticker symbols (comma separated)",
    value="AAPL,MSFT"
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date.today() - timedelta(days=365)
)

end_date = st.sidebar.date_input(
    "End date",
    value=date.today()
)

tickers = [t.strip().upper() for t in tickers_input.split(",")if t.strip()]

if len(tickers) <2 or len(tickers) >5:
    st.error("Please enter between 2 and 5 ticker symbols.")
    st.stop()

if (end_date - start_date).days < 365:
    st.error("Please select a date range of at least 1 year.")
    st.stop()

@st.cache_data
def load_data(tickers,start,end):
    try:
        all_tickers = tickers + ["^GSPC"]
        data = yf.download(all_tickers,start=start, end=end,auto_adjust=True)["Close"]
        return data
    except Exception as e:
        return None
    
with st.spinner("Downloading data..."):
    data = load_data(tickers, start_date, end_date)

if data is None or data.empty:
    st.error("Could not download data. Check your ticker symbols and try again.")
    st.stop()

bad_tickers = [t for t in tickers if t not in data.columns or data[t].isna().all()]
if bad_tickers:
    st.error(f"The following tickers returned no data: {', '.join(bad_tickers)}. Please check your symbols.")
    st.stop()

missing_pct = data[tickers].isna().mean()
tickers_to_drop = missing_pct[missing_pct > 0.05].index.tolist()
if tickers_to_drop:
    st.warning(f"Dropping {', '.join(tickers_to_drop)} due to more than 5% missing data.")
    tickers = [t for t in tickers if t not in tickers_to_drop]
    data = data.drop(columns=tickers_to_drop)

if len(tickers) < 2:
    st.error("Not enough valid tickers remaining. Please enter at least 2 valid symbols.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Price & Returns", "Risk & Distribution", "Correlation & Diversification"])

with tab1:


    import plotly.express as px

    selected = st.multiselect(
        "Select stocks to display",
        options=tickers,
        default=tickers
    )

    if selected:
        fig = px.line(
            data[selected],
            title="Adjusted Closing Prices",
            labels={"value": "Price (USD)", "variable": "Stock"}
        )
        st.plotly_chart(fig)
    returns = data.pct_change().dropna()

    st.subheader("Summary Statistics")

    summary = pd.DataFrame({
        "Annualized Return": returns.mean()*252,
        "Annualized Volatility": returns.std()*(252**0.5),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurt(),
        "Min Daily Return": returns.min(),
        "Max Daily Return": returns.max()
    })

    st.dataframe(summary.T.style.format("{:.4f}"))

    st.subheader("Cumulative Wealth Index ($10,000 Investment)")

    stock_returns = returns[tickers]
    portfolio_returns = stock_returns.mean(axis=1)
    portfolio_returns.name = "Equal-Weight Portfolio"

    wealth = (1+ returns).cumprod()*10000
    portfolio_wealth = (1+portfolio_returns).cumprod()*10000

    wealth_combined = wealth.join(portfolio_wealth)

    fig2 = px.line(
        wealth_combined,
        title="Growth of $10,000 Investment",
        labels = {"value": "Portfolio Value (USD)", "variable":"Asset"}
    )
    st.plotly_chart(fig2)

with tab2:

    st.subheader("Rolling Volatility")

    window = st.slider("Rolling window (days)", min_value=10, max_value=120, value=30, step=5)

    rolling_vol = returns[tickers].rolling(window).std() * (252 ** 0.5)

    fig3 = px.line(
        rolling_vol,
        title=f"Rolling {window}-Day Annualized Volatility",
        labels={"value": "Annualized Volatility", "variable": "Stock"}
    )
    st.plotly_chart(fig3)

    st.subheader("Return Distribution")

    import numpy as np
    from scipy import stats

    selected_stock = st.selectbox("Select a stock", options=tickers)

    stock_ret = returns [selected_stock].dropna()

    mu, std = stats.norm.fit(stock_ret)
    x = np.linspace(stock_ret.min(), stock_ret.max(),100)
    normal_curve = stats.norm.pdf(x,mu,std)

    plot_type = st.radio("Select plot type", options=["Histogram", "Q-Q Plot"])

    if plot_type == "Histogram":
        fig4 = px.histogram(
            stock_ret,
            nbins=50,
            histnorm="probability density",
            title = f"{selected_stock} Daily Return Distribution",
            labels = {"value": "Daily Return", "count": "Density"}
        )
        fig4.add_scatter(x=x, y=normal_curve, mode="lines", name="Normal Fit")
        st.plotly_chart(fig4)
        
    elif plot_type == "Q-Q Plot":
        qq = stats.probplot(stock_ret)
        qq_df = pd.DataFrame({
            "Theoretical Quantiles": qq[0][0],
            "Sample Quantiles": qq[0][1]
        })
        fig_qq = px.scatter(
            qq_df,
            x="Theoretical Quantiles",
            y="Sample Quantiles",
            title=f"{selected_stock} Q-Q Plot"
        )
        line_y = qq[1][1] + qq[1][0] * qq[0][0]
        fig_qq.add_scatter(
            x=qq[0][0],
            y=line_y,
            mode="lines",
            name="Normal Line"
        )
        st.plotly_chart(fig_qq)

    jb_stat, jb_p = stats.jarque_bera(stock_ret)
    st.markdown(f"**Jarque-Bera Statistic** {jb_stat:.4f}")
    st.markdown(f"**P-value:** " + f"{jb_p:.4f}")
    if jb_p < 0.05:
        st.warning("Rejects normality (p < 0.05)")
    else:
        st.success("Fails to reject normality (p >= 0.05)")

        st.subheader("Return Distribution Box Plot")

    fig5 = px.box(
            returns[tickers],
            title = "Daily Return Distributions",
            labels={"value": "Daily Return","variable":"Stock"}
        )

    st.plotly_chart(fig5)

with tab3:

    st.subheader("Correlation Heatmap")

    corr_matrix = returns[tickers].corr()

    fig6 = px.imshow(
        corr_matrix,
        text_auto = ".2f",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        title="Pairwise Correlation of Daily Returns"
    )
    st.plotly_chart(fig6)

    st.subheader("Return Scatter Plot")

    col1, col2 = st.columns(2)
    with col1:
        stock_a = st.selectbox("Select Stock A", options=tickers, index=0)
    with col2:
        stock_b = st.selectbox("Select Stock B", options=tickers, index=1)

    fig7 = px.scatter(
        x=returns[stock_a],
        y=returns[stock_b],
        title=f"{stock_a} vs {stock_b} Daily Returns",
        labels={"x":f"{stock_a} Return", "y": f"{stock_b} Return"}
    )
    st.plotly_chart(fig7)

    st.subheader("Rolling Correlation")

    roll_window = st.slider(
        "Rolling window (days)",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
        key="roll_corr_window"
    )

    rolling_corr = returns[stock_a].rolling(roll_window).corr(returns[stock_b])

    fig8 = px.line(
        rolling_corr,
        title=f"Rolling {roll_window}-Day Correlation: {stock_a} vs {stock_b}",
    labels={"value": "Correlation", "index": "Date"}
    )
    st.plotly_chart(fig8)

    st.subheader("Two-Asset Portfolio Explorer")

    weight_a = st.slider(
        f"Weight on {stock_a} (%)",
        min_value =0,
        max_value=100,
        value=50,
        step=5,
        key="portfolio_weight"
    )

    w= weight_a/100
    ann_returns = returns[tickers].mean()*252
    ann_cov = returns[tickers].cov()*252

    ret_a = ann_returns[stock_a]
    ret_b = ann_returns[stock_b]
    vol_a = np.sqrt(ann_cov.loc[stock_a][stock_a])
    vol_b = np.sqrt(ann_cov.loc[stock_b][stock_b])
    cov_ab = ann_cov.loc[stock_a][stock_b]

    port_return = w * ret_a + (1-w) * ret_b
    port_vol = np.sqrt(w**2 * vol_a**2 + (1-w)**2 * vol_b**2 + 2*w*(1-w)*cov_ab)

    st.metric(f"Portfolio Annualized Return", f"{port_return:.2%}")
    st.metric(f"Portfolio Annualized Volatility", f"{port_vol:.2%}")

    weights = np.linspace(0, 1, 101)
port_vols = np.sqrt(
    weights**2 * vol_a**2 +
    (1 - weights)**2 * vol_b**2 +
    2 * weights * (1 - weights) * cov_ab
)

vol_curve_df = pd.DataFrame({
    "Weight on " + stock_a: weights * 100,
    "Portfolio Volatility": port_vols
})

fig9 = px.line(
    vol_curve_df,
    x="Weight on " + stock_a,
    y="Portfolio Volatility",
    title=f"Portfolio Volatility Curve: {stock_a} vs {stock_b}",
    labels={"x": f"Weight on {stock_a} (%)", "y": "Annualized Volatility"}
)

fig9.add_vline(
    x=weight_a,
    line_dash="dash",
    line_color="red",
    annotation_text="Current"
)

st.plotly_chart(fig9)

st.info(
    "Combining two stocks can produce a portfolio with lower volatility than either stock "
    "individually. This diversification effect is stronger when the correlation between "
    "the two stocks is lower."
)


with st.sidebar.expander("About"):
    st.write("""
    **Data Source:** Yahoo Finance via yfinance
    
    **Returns:** Simple (arithmetic) returns calculated as daily percentage change in adjusted closing price.
    
    **Annualization:** Mean daily return × 252 for annualized return. Daily standard deviation × √252 for annualized volatility. 252 trading days per year.
    
    **Cumulative Wealth Index:** Growth of $10,000 using (1 + r).cumprod()
    
    **Equal-Weight Portfolio:** Average of daily returns across all selected stocks.
    
    **Two-Asset Portfolio:** Volatility calculated using the standard two-asset portfolio variance formula including the covariance term.
    
    **Normality Test:** Jarque-Bera test at 5% significance level.
    """)