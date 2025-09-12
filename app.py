import os
from datetime import datetime
from typing import List, Dict, Any

import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# =========================
# Environment & Page Config
# =========================
load_dotenv()  # Load .env if present
DEFAULT_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="🤖 Stock Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------- THEME TIP --------------
# To force light theme, create .streamlit/config.toml with:
# [theme]\nbase = "light"\nprimaryColor = "#4CAF50"\nbackgroundColor = "#FFFFFF"\nsecondaryBackgroundColor = "#F5F5F5"\ntextColor = "#000000"

# =========================
# Utilities & Caching Layers
# =========================
def human_format(num: Any) -> str:
    """Convert large numbers into human-readable form (e.g. 2.83T)."""
    try:
        num = float(num)
    except Exception:
        return str(num)
    magnitude = 0
    units = ["", "K", "M", "B", "T", "P"]
    while abs(num) >= 1000 and magnitude < len(units) - 1:
        magnitude += 1
        num /= 1000.0
    # remove .00 if not needed
    if num.is_integer():
        return f"{int(num)}{units[magnitude]}"
    return f"{num:.2f}{units[magnitude]}"


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_history(symbols: List[str], period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch adjusted close prices for symbols/period.
    Returns wide DataFrame with columns per symbol.
    """
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            prices = data["Close"].copy()
        else:
            prices = data.copy()
        # Ensure DataFrame for single symbol case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=symbols[0])
        return prices.dropna(how="all")
    except Exception as e:
        st.error(f"Error fetching price history: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_company_info(symbol: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector", "N/A"),
            "market_cap": human_format(info.get("marketCap")) if info.get("marketCap") else "N/A",
            "summary": info.get("longBusinessSummary", "N/A"),
        }
    except Exception as e:
        return {"name": symbol, "sector": "N/A", "market_cap": "N/A", "summary": f"Info unavailable: {e}"}

@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_company_news(symbol: str, limit: int = 5) -> list[dict]:
    """Return normalized news: [{title, url, provider, published_at}]"""
    try:
        t = yf.Ticker(symbol)
        raw = t.news or []
        items: list[dict] = []
        for it in raw:
            c = it.get("content") or {}
            title = c.get("title") or it.get("title") or c.get("summary")
            url = ((c.get("clickThroughUrl") or {}).get("url")) or ((c.get("canonicalUrl") or {}).get("url")) \
                  or it.get("link") or it.get("url")
            provider = (c.get("provider") or {}).get("displayName") or it.get("publisher") \
                       or (it.get("provider") or {}).get("name") or ""
            pub = c.get("pubDate") or c.get("displayTime") or it.get("providerPublishTime") or it.get("published_at")
            if title and url:
                items.append({"title": title, "url": url, "provider": provider, "published_at": pub})
        return items[:limit]
    except Exception:
        return []


# =========================
# Basic Quant Helpers
# =========================

def period_change(prices: pd.DataFrame) -> pd.Series:
    """
    Percent change from the first to the last row of `prices`,
    regardless of whether the fetched period is 3mo/6mo/ytd/1y/2y.
    Returns a sorted Series (desc).
    """
    if prices.empty:
        return pd.Series(dtype=float)
    filled = prices.ffill().bfill()
    first = filled.iloc[0]
    last = filled.iloc[-1]
    return (last / first - 1.0).sort_values(ascending=False)

# =========================
# AI Agents (Gemini via agno)
# ========================= 


def make_agent(description: str, instructions: List[str]) -> Agent:
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description=description,
        instructions=instructions,
        show_tool_calls=False,
        markdown=True,
    )

market_analyst = make_agent(
    "Analyze and compare stocks performance over time.",
    [
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over the selected period.",
        "Rank stocks based on their relative performance.",
    ],
)

company_researcher = make_agent(
    "Research and summarize company profiles, financials and recent news.",
    [
        "Fetch company details such as name, sector, market cap, and business summary.",
        "Summarize recent news articles related to the company relevant to investors.",
        "Provide sector, market cap, and business overview.",
    ],
)

stock_strategy_agent = make_agent(
    "Provide stock investment strategies based on market and company analysis. Recommend top stocks.",
    [
        "Based on the market analysis and company research, suggest investment strategies.",
        "Consider factors like market trends, company performance, and recent news.",
        "Recommend top stocks to invest in with reasons.",
    ],
)

team_lead = make_agent(
    "Aggregate stock analysis, company research, and investment strategy.",
    [
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis.",
    ],
)

# Graceful check for API key
HAS_KEY = bool(os.environ.get("GOOGLE_API_KEY"))

# =========================
# Core Analysis Functions
# =========================

def compare_stocks(symbols: List[str], period: str = "6mo") -> Dict[str, float]:
    prices = fetch_history(symbols, period=period)
    if prices.empty:
        return {}
    returns = period_change(prices)
    return {sym: float(returns.get(sym, float("nan"))) for sym in symbols}


def analyze_market(symbols: List[str], period: str = "6mo") -> str:
    perf = compare_stocks(symbols, period)
    if not perf:
        return "No valid stock data available for the given symbols."
    if not HAS_KEY:
        # Fallback summary without AI
        ranked = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        lines = [f"**{s}**: {p*100:.2f}%" for s, p in ranked]
        return "\n".join(["AI disabled (no GOOGLE_API_KEY). Basic ranking:"] + lines)
    resp = market_analyst.run(f"Compare these stock performances over {period}: {perf}. Rank them and add a brief insight.")
    return getattr(resp, "content", str(resp))


def get_company_analysis(symbol: str) -> str:
    info = fetch_company_info(symbol)
    news = fetch_company_news(symbol)

    news_lines = []
    for n in news:
        title, url, provider, pub = n["title"], n["url"], n.get("provider", ""), n.get("published_at")
        pub_str = ""
        if isinstance(pub, (int, float)):
            try:
                from datetime import datetime
                pub_str = datetime.fromtimestamp(pub).strftime("%Y-%m-%d")
            except Exception:
                pub_str = ""
        elif isinstance(pub, str):
            pub_str = pub.split("T")[0]
        suffix = f" ({pub_str})" if pub_str else ""
        prov = f" — {provider}" if provider else ""
        news_lines.append(f"- {title}{suffix}: {url}{prov}")

    if not news_lines:
        news_lines.append("- No reliable news items available.")

    prompt = (
        f"Company: {info['name']} ({symbol})\n"
        f"Sector: {info['sector']}\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n\n"
        f"Recent News:\n" + "\n".join(news_lines) + "\n\n"
        "Summarize the company's profile and any investor-relevant developments."
    )

    if not os.environ.get("GOOGLE_API_KEY"):
        return (
            f"**{info['name']}** ({symbol}) — Sector: {info['sector']}, Market Cap: {info['market_cap']}\n\n"
            f"{(info['summary'] or 'N/A')}\n\n" + "\n".join(news_lines)
        )

    analysis = company_researcher.run(prompt)
    return analysis.content


def get_stock_recommendations(symbols: List[str], period: str = "6mo") -> str:
    market_analysis = analyze_market(symbols, period)
    companies = {s: get_company_analysis(s) for s in symbols}

    if not HAS_KEY:
        # Fallback message
        return (
            "AI disabled (no GOOGLE_API_KEY). Based on recent returns, favor top performers but diversify.\n\n"
            + market_analysis
        )

    prompt = (
        f"Market Analysis: {market_analysis}\n\n"
        f"Company Analyses: {companies}\n\n"
        "Recommend 2-4 stocks to consider now. Provide reasons, risks, and a suggested holding horizon."
    )
    resp = stock_strategy_agent.run(prompt)
    return getattr(resp, "content", str(resp))


def generate_final_investment_report(symbols: List[str], period: str = "6mo") -> str:
    market_analysis = analyze_market(symbols, period)
    company_analyses = {s: get_company_analysis(s) for s in symbols}
    recommendations = get_stock_recommendations(symbols, period)

    if not HAS_KEY:
        # Compose locally
        parts = [
            "# Investment Report",
            "\n## Market Analysis\n" + market_analysis,
            "\n## Company Analyses\n" + "\n\n".join([f"### {s}\n{txt}" for s, txt in company_analyses.items()]),
            "\n## Recommendations\n" + recommendations,
        ]
        return "\n".join(parts)

    prompt = (
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Stock Recommendations:\n{recommendations}\n\n"
        "Provide a concise, investor-friendly report. Include a ranked list of the symbols from best to worst for the selected period, with reasoning."
    )
    resp = team_lead.run(prompt)
    return getattr(resp, "content", str(resp))

# =========================
# UI — Sidebar Controls
# =========================
with st.sidebar:
    st.markdown("## 📊 Symbols & Settings")
    default_syms = "AAPL, MSFT, GOOGL"
    input_symbols = st.text_input("Symbols (comma-separated)", default_syms)

    colA, colB = st.columns(2)
    with colA:
        period = st.selectbox("Period", ["3mo", "6mo", "ytd", "1y", "2y"], index=1)
    with colB:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    normalize = st.toggle("Normalize to 100 at start", value=True, help="Index each series to 100 on the first day for easier comparison.")
    show_ma = st.toggle("Show Moving Average (20)", value=False)

    run_btn = st.button("🚀 Generate Investment Report", use_container_width=True, type="primary")

symbols: List[str] = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]

# =========================
# Header
# =========================
st.markdown(
    """
    <div style="text-align:center;">
      <h1>🤖 Stock Investment Advisor</h1>
      <p style="color:#6c757d;">AI‑powered stock analysis, company research, and recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Tabs Layout
# =========================
overview_tab, companies_tab, ai_report_tab = st.tabs(["Overview", "Company Profiles", "AI Report"])


# -------- Overview Tab --------
with overview_tab:
    prices = fetch_history(symbols, period=period, interval=interval)

    if prices.empty:
        st.warning("No price data found. Try different symbols or period.")
    else:
        # KPI cards for selected period change
        returns = period_change(prices)  # sorted desc
        cols_per_row = 4  # adjust to taste (e.g., 5)

        symbols_sorted = list(returns.index)
        for start in range(0, len(symbols_sorted), cols_per_row):
            batch = symbols_sorted[start:start + cols_per_row]
            row = st.columns(len(batch))
            for i, sym in enumerate(batch):
                with row[i]:
                    st.metric(
                        label=f"{sym} Return ({period})",
                        value=f"{returns[sym]*100:.2f}%"
                    )


        # Chart
        chart_df = prices.copy().ffill().bfill()
        if normalize and not chart_df.empty:
            chart_df = chart_df / chart_df.iloc[0] * 100.0

        fig = go.Figure()
        for sym in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[sym], mode="lines", name=sym))

        if show_ma and not chart_df.empty:
            ma = chart_df.rolling(20).mean()
            for sym in ma.columns:
                fig.add_trace(go.Scatter(x=ma.index, y=ma[sym], mode="lines", name=f"{sym} MA20", line=dict(dash="dash")))

        fig.update_layout(
            title=f"Price Comparison ({period}{', normalized' if normalize else ''})",
            xaxis_title="Date",
            yaxis=dict(
                title="Indexed Level" if normalize else "Price (USD)",
                tickfont=dict(size=12),
                automargin=True
        ),
        xaxis=dict(
            tickfont=dict(size=12),
            automargin=True
        ),
        template="plotly_white",  # or "plotly_dark" if in dark mode
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),  # extra left margin so label isn’t squeezed
    )

        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        corr = prices.pct_change().corr().fillna(0)
        with st.expander("📚 Correlation Matrix", expanded=False):
            heat = px.imshow(corr, text_auto=True, aspect="auto", title="Daily Return Correlations")
            st.plotly_chart(heat, use_container_width=True)

# -------- Companies Tab --------
with companies_tab:
    if not symbols:
        st.info("Enter symbols to view company details and news.")
    else:
        for sym in symbols:
            info = fetch_company_info(sym)
            news = fetch_company_news(sym)
            with st.container(border=True):
                st.subheader(f"{info['name']} ({sym})")
                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    st.markdown(
                        f"**Sector:** {info['sector']}  \n**Market Cap:** {info['market_cap']}"
                    )
                with c2:
                    st.markdown("**Summary (brief):**")
                with c3:
                    st.caption((info["summary"] or "N/A")[:400] + ("..." if info.get("summary") and len(info["summary"]) > 400 else ""))

                
                if news:
                    st.markdown("**Latest News:**")
                    for n in news:
                        t, u, p = n["title"], n["url"], n.get("provider", "")
                        suffix = f" — _{p}_" if p else ""
                        st.markdown(f"- [{t}]({u}){suffix}")

                else:
                    st.caption("No recent news available.")




# -------- AI Report Tab --------
with ai_report_tab:
    st.markdown("Generate a consolidated, investor‑friendly report. This may take a few moments.")
    if run_btn:
        if not symbols:
            st.error("Please enter at least one symbol.")
        else:
            with st.spinner("Analyzing stocks and generating report..."):
                report = generate_final_investment_report(symbols, period=period)

            st.success("Report ready!")
            st.markdown(report)

            # Offer download
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"investment_report_{ts}.md"
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name=fname,
                mime="text/markdown",
                use_container_width=True,
            )

# Footer
st.markdown(
    """
    <hr/>
    <div style="text-align:center; font-size: 0.9rem; color:#6c757d;">
      Built with Streamlit · Data: Yahoo Finance · LLM: Gemini via agno
    </div>
    """,
    unsafe_allow_html=True,
)
