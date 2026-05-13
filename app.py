import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.ingestion import fetch_stock_data, fetch_news
from src.processing import analyze_sentiment

st.set_page_config(page_title="AI Finance Sentiment Tracker", layout="wide")

st.markdown("""
<style>
/* ── Global ── */
.stApp {
    background: linear-gradient(160deg, #0e1c2f 0%, #152032 50%, #111c2e 100%);
}
[data-testid="stSidebar"] {
    background: rgba(14,22,40,0.97);
    border-right: 1px solid rgba(99,130,200,0.12);
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }
h1,h2,h3,h4 { color: #e8edf8 !important; }
p, label, .stMarkdown { color: #b0bbd4; }

/* ── Ticker header ── */
.ticker-header {
    display: flex; align-items: center; gap: 22px;
    padding: 20px 28px; border-radius: 16px;
    background: linear-gradient(135deg, rgba(30,50,90,0.6), rgba(20,35,65,0.4));
    border: 1px solid rgba(99,130,200,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.ticker-name  { font-size: 2em; font-weight: 800; color: #e8f0ff; letter-spacing: -0.02em; }
.ticker-price { font-size: 1.7em; font-weight: 600; color: #c8d8f8; }
.ticker-change-up   { font-size: 0.95em; font-weight: 700; color: #2ecc8f; background: rgba(46,204,143,0.15); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(46,204,143,0.25); }
.ticker-change-down { font-size: 0.95em; font-weight: 700; color: #e05555; background: rgba(224,85,85,0.15); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(224,85,85,0.25); }
.ticker-model { margin-left: auto; font-size: 0.7em; color: #445; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,50,90,0.45), rgba(20,35,65,0.3));
    border: 1px solid rgba(99,130,200,0.18);
    border-radius: 14px; padding: 20px 14px;
    text-align: center; min-height: 110px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    transition: all 0.2s;
}
.kpi-card:hover {
    border-color: rgba(99,130,200,0.4);
    background: linear-gradient(135deg, rgba(40,65,115,0.55), rgba(25,45,80,0.4));
    box-shadow: 0 4px 20px rgba(60,100,200,0.12);
    transform: translateY(-1px);
}
.kpi-label { font-size: 0.66em; color: #6a7fa8; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px; }
.kpi-value { font-size: 1.9em; font-weight: 700; line-height: 1; color: #e0eaff; }
.kpi-sub   { font-size: 0.73em; color: #5a7090; margin-top: 7px; min-height: 18px; }

/* ── Signal banner ── */
.signal-banner {
    padding: 14px 24px; border-radius: 12px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
    backdrop-filter: blur(4px);
}
.signal-label  { font-size: 1.1em; font-weight: 800; letter-spacing: 0.05em; }
.signal-reason { font-size: 0.85em; color: #8898bb; }

/* ── News cards ── */
.news-card {
    background: rgba(25,42,75,0.4);
    border-radius: 10px; padding: 12px 16px; margin: 6px 0;
    border-left: 4px solid; transition: all 0.15s;
    box-shadow: 0 1px 8px rgba(0,0,0,0.15);
}
.news-card:hover { background: rgba(35,58,105,0.55); transform: translateX(2px); }
.news-title { font-size: 0.87em; font-weight: 600; color: #c8d5ec; margin-bottom: 6px; }
.news-meta  { font-size: 0.71em; color: #4a6080; display: flex; align-items: center; gap: 8px; }
.score-bar-wrap { flex: 1; background: rgba(255,255,255,0.07); border-radius: 4px; height: 3px; }
.score-bar      { height: 3px; border-radius: 4px; }
.score-val      { font-size: 0.72em; font-weight: 700; min-width: 38px; text-align: right; }

/* ── Claude summary box ── */
.claude-box {
    background: linear-gradient(135deg, rgba(20,32,70,0.8), rgba(15,25,55,0.7));
    border: 1px solid rgba(99,110,250,0.25); border-left: 4px solid #636EFA;
    border-radius: 12px; padding: 18px 24px;
    font-size: 0.93em; line-height: 1.78; color: #b8c8e8;
    margin-bottom: 16px; box-shadow: 0 4px 20px rgba(60,80,200,0.1);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_price_data(path):
    # Handle legacy MultiIndex CSV format from yfinance (two junk header rows)
    peek = pd.read_csv(path, nrows=2)
    if not peek.empty and str(peek.iloc[0, 0]) == "Ticker":
        df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


@st.cache_data
def load_sentiment_data(path):
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def get_live_price(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if len(hist) >= 2:
            prev, curr = float(hist["Close"].iloc[-2]), float(hist["Close"].iloc[-1])
            chg = curr - prev
            return curr, chg, chg / prev * 100
        elif len(hist) == 1:
            return float(hist["Close"].iloc[-1]), None, None
    except Exception:
        pass
    return None, None, None


def validate_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        if info and "symbol" in info:
            return True, None
        suggestions = [t for t in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] if ticker.upper() in t]
        return False, suggestions[:3] if suggestions else None
    except Exception:
        return False, None


def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(data, window=20, n_std=2):
    mid = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    return mid, mid + std * n_std, mid - std * n_std


def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def compute_ai_signal(sentiment_df, prices_df):
    """Buy/Hold/Sell signal based on FinBERT sentiment score + RSI."""
    if sentiment_df is None or sentiment_df.empty or prices_df is None or prices_df.empty:
        return "N/A", "Données insuffisantes", "#888888"

    score = float(sentiment_df["positive"].mean() - sentiment_df["negative"].mean())
    rsi_series = calculate_rsi(prices_df).dropna()
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    if score > 0.4 and 30 < rsi < 65:
        return "FORT ACHAT", f"Sentiment très positif ({score:.2f}) · RSI sain ({rsi:.1f})", "#00CC96"
    elif score > 0.15 and rsi < 70:
        return "ACHAT", f"Sentiment positif ({score:.2f}) · RSI ({rsi:.1f})", "#00AA77"
    elif score < -0.3 or rsi > 78:
        return "VENTE", f"Sentiment négatif ({score:.2f}) · RSI élevé ({rsi:.1f})", "#EF553B"
    elif score < -0.1 or rsi > 72:
        return "PRUDENCE", f"Sentiment mitigé ({score:.2f}) · RSI ({rsi:.1f})", "#FF9900"
    else:
        return "CONSERVER", f"Sentiment neutre ({score:.2f}) · RSI ({rsi:.1f})", "#636EFA"


def generate_market_summary(ticker, sentiment_df, prices_df):
    """Call Claude API to generate a natural-language market summary with prompt caching."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_AVAILABLE:
        return None

    client = _anthropic.Anthropic(api_key=api_key)

    score = float(sentiment_df["positive"].mean() - sentiment_df["negative"].mean())
    total = len(sentiment_df)
    pos_pct = int(100 * (sentiment_df["sentiment"] == "positive").sum() / total) if total else 0
    neg_pct = int(100 * (sentiment_df["sentiment"] == "negative").sum() / total) if total else 0

    rsi_series = calculate_rsi(prices_df).dropna()
    rsi = f"{float(rsi_series.iloc[-1]):.1f}" if not rsi_series.empty else "N/A"

    close = prices_df["Close"].dropna()
    if len(close) >= 2:
        lookback = min(22, len(close) - 1)
        price_chg = f"{(float(close.iloc[-1]) - float(close.iloc[-lookback])) / float(close.iloc[-lookback]) * 100:.1f}%"
    else:
        price_chg = "N/A"

    top = sentiment_df.assign(score=lambda d: d["positive"] - d["negative"]).nlargest(3, "score")
    headlines = "\n".join(f"- {row['title']} (score: {row['score']:.2f})" for _, row in top.iterrows())

    user_content = (
        f"Ticker : {ticker}\n"
        f"Score sentiment moyen : {score:.3f}\n"
        f"Distribution : {pos_pct}% positif, {neg_pct}% négatif\n"
        f"RSI (14j) : {rsi}\n"
        f"Variation cours (~1 mois) : {price_chg}\n\n"
        f"Top actualités :\n{headlines}\n\n"
        "Fournis une analyse de marché concise (3-4 phrases) en français."
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": (
                    "Tu es un analyste financier expert. Tu fournis des résumés de marché "
                    "concis et professionnels en français, basés sur les données de sentiment "
                    "NLP et les indicateurs techniques. Ton ton est informatif mais prudent — "
                    "tu rappelles toujours que l'analyse est indicative et non un conseil en investissement."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def compute_rolling_correlation(prices_df, sentiment_df, window=14):
    """Rolling correlation entre score de sentiment journalier et rendements.
    La fenêtre s'adapte automatiquement si moins de `window` jours sont disponibles."""
    if "publishedAt" not in sentiment_df.columns:
        return None, window

    sent = sentiment_df.copy()
    sent["publishedAt"] = pd.to_datetime(sent["publishedAt"], errors="coerce", utc=True)
    sent = sent.dropna(subset=["publishedAt"])
    sent["date"] = sent["publishedAt"].dt.tz_localize(None).dt.normalize()
    sent["score"] = sent["positive"] - sent["negative"]
    daily_sent = sent.groupby("date")["score"].mean().reset_index()

    returns = prices_df["Close"].pct_change()
    returns.index = pd.to_datetime(returns.index).normalize()
    returns_df = returns.reset_index()
    returns_df.columns = ["date", "returns"]

    merged = pd.merge(daily_sent, returns_df, on="date", how="inner").dropna()
    n = len(merged)
    if n < 3:
        return None, window

    # Adaptive window: use requested window if enough data, otherwise shrink to n-2
    actual_window = min(window, max(3, n - 2))

    merged = merged.sort_values("date")
    merged["rolling_corr"] = merged["score"].rolling(actual_window).corr(merged["returns"])
    return merged, actual_window


def render_news_cards(sentiment_df, max_items=12):
    color_map = {"positive": "#00CC96", "negative": "#EF553B", "neutral": "#636EFA"}
    emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}

    cards = []
    for _, row in sentiment_df.head(max_items).iterrows():
        sent = row.get("sentiment", "neutral")
        color = color_map.get(sent, "#888")
        emoji = emoji_map.get(sent, "⚪")
        score = float(row.get("positive", 0)) - float(row.get("negative", 0))
        title = str(row.get("title", ""))
        publisher = str(row.get("publisher", "Unknown"))
        link = str(row.get("link", ""))

        title_html = (
            f'<a href="{link}" target="_blank" style="color:#dde;text-decoration:none;">{title}</a>'
            if link and link != "nan" else title
        )
        bar_w = int(min(abs(score) * 100, 100))
        bar_color = "#00CC96" if score > 0 else "#EF553B"

        cards.append(f"""
        <div class="news-card" style="border-color:{color};">
            <div class="news-title">{emoji} {title_html}</div>
            <div class="news-meta">
                <span>{publisher}</span>
                <div class="score-bar-wrap">
                    <div class="score-bar" style="width:{bar_w}%;background:{bar_color};"></div>
                </div>
                <span class="score-val" style="color:{color};">{score:+.2f}</span>
            </div>
        </div>""")

    return "\n".join(cards)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

available_tickers = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "IBM", "INTC",
    "AMD", "NFLX", "SPOT", "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "AVGO",
    "PYPL", "SQ", "SHOP", "UBER", "LYFT", "ZM", "DOCU", "TWLO", "SNOW", "PLTR",
]

search_query = st.sidebar.text_input(
    "Rechercher un ticker",
    placeholder="Tapez un symbole (ex: NVDA, AAPL)...",
    help="Tapez pour filtrer la liste des tickers disponibles.",
)

filtered_tickers = [t for t in available_tickers if search_query.upper() in t] if search_query else available_tickers

selected_ticker = st.sidebar.selectbox(
    "Choisir un stock",
    options=filtered_tickers,
    index=0 if filtered_tickers else None,
    help="Sélectionnez un ticker dans la liste filtrée.",
)

custom_ticker = st.sidebar.text_input("Ou saisir un ticker personnalisé", value="")

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "ticker_override" not in st.session_state:
    st.session_state.ticker_override = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "ai_summary_ticker" not in st.session_state:
    st.session_state.ai_summary_ticker = None

# Ticker resolution: custom > override (from favorites) > selectbox
if custom_ticker.strip():
    ticker = custom_ticker.strip().upper()
    st.session_state.ticker_override = None
elif st.session_state.ticker_override:
    ticker = st.session_state.ticker_override
else:
    ticker = selected_ticker if selected_ticker else "NVDA"

if custom_ticker.strip():
    is_valid, suggestions = validate_ticker(ticker)
    if not is_valid:
        st.sidebar.error(
            f"Ticker '{ticker}' non trouvé. Suggestions : {', '.join(suggestions) if suggestions else 'Aucun'}"
        )
        ticker = selected_ticker if selected_ticker else "NVDA"

if st.sidebar.button("⭐ Ajouter aux favoris", key="add_fav"):
    if ticker not in st.session_state.favorites:
        st.session_state.favorites.append(ticker)
        st.sidebar.success(f"{ticker} ajouté aux favoris !")
    else:
        st.sidebar.info(f"{ticker} est déjà dans les favoris.")

if st.session_state.favorites:
    st.sidebar.subheader("⭐ Favoris")
    for fav in list(st.session_state.favorites):
        col1, col2 = st.sidebar.columns([3, 1])
        if col1.button(fav, key=f"select_{fav}"):
            st.session_state.ticker_override = fav
            st.rerun()
        if col2.button("❌", key=f"remove_{fav}"):
            st.session_state.favorites.remove(fav)
            if st.session_state.ticker_override == fav:
                st.session_state.ticker_override = None
            st.rerun()

start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("today"))
if start_date > end_date:
    st.sidebar.error("La date de début doit être antérieure à la date de fin.")

sentiment_filter = st.sidebar.selectbox(
    "Filtrer le sentiment",
    options=["Tous", "positive", "negative", "neutral"],
    help="Affiche uniquement les actualités du sentiment sélectionné.",
)

force_refresh = st.sidebar.checkbox(
    "Forcer le rafraîchissement des données",
    help="Télécharge à nouveau les données même si elles sont en cache.",
)

st.sidebar.subheader("Indicateurs techniques")
show_technical_indicators = st.sidebar.checkbox("SMA 20 / SMA 50", help="Moyennes mobiles simples.")
show_rsi = st.sidebar.checkbox("RSI (14)", help="Indice de Force Relative.")
show_bollinger = st.sidebar.checkbox("Bandes de Bollinger", help="Bandes de Bollinger 20-2.")
show_macd = st.sidebar.checkbox("MACD (12-26-9)", help="Moving Average Convergence Divergence.")
show_volume = st.sidebar.checkbox("Volume", help="Volume journalier des échanges.")
show_sentiment_overlay = st.sidebar.checkbox(
    "Sentiment overlay", help="Superpose le score de sentiment sur le graphique des prix (axe droit)."
)

run_button = st.sidebar.button("Lancer l'analyse IA", type="primary")

# ── Ticker header with live price ────────────────────────────────────────────

live_price, live_chg, live_chg_pct = get_live_price(ticker)

if live_price is not None and live_chg is not None:
    chg_class = "ticker-change-up" if live_chg >= 0 else "ticker-change-down"
    arrow = "▲" if live_chg >= 0 else "▼"
    chg_html = f'<span class="{chg_class}">{arrow} {live_chg:+.2f} ({live_chg_pct:+.2f}%)</span>'
else:
    chg_html = ""

price_html = f'<span class="ticker-price">${live_price:,.2f}</span>' if live_price else ""

st.markdown(
    f"""<div class="ticker-header">
        <span class="ticker-name">📈 {ticker}</span>
        {price_html}
        {chg_html}
        <span class="ticker-model">FinBERT · Yahoo Finance</span>
    </div>""",
    unsafe_allow_html=True,
)

# ── Analysis trigger ──────────────────────────────────────────────────────────

if run_button:
    with st.spinner(f"Analyse en cours pour {ticker}…"):
        fetch_stock_data(ticker, start=str(start_date), force_refresh=force_refresh)
        fetch_news(ticker, force_refresh=force_refresh)
        analyze_sentiment(ticker)
        load_price_data.clear()
        load_sentiment_data.clear()
        get_live_price.clear()
        st.session_state.ai_summary = None
        st.session_state.ai_summary_ticker = None
        st.sidebar.success("Analyse terminée !")

# ── Main dashboard ────────────────────────────────────────────────────────────

try:
    price_path = f"data/{ticker}_prices.csv"
    sent_path = f"data/{ticker}_sentiment.csv"

    if os.path.exists(price_path) and os.path.exists(sent_path):
        prices = load_price_data(price_path)
        raw_sentiment = load_sentiment_data(sent_path)

        if sentiment_filter != "Tous":
            sentiment = raw_sentiment[raw_sentiment["sentiment"] == sentiment_filter].copy()
        else:
            sentiment = raw_sentiment.copy()

        prices = prices.sort_index()
        prices_range = prices.loc[str(start_date) : str(end_date)]

        if "publishedAt" in sentiment.columns:
            sentiment["publishedAt"] = (
                pd.to_datetime(sentiment["publishedAt"], errors="coerce", utc=True)
                .dt.tz_convert(None)
            )
            mask = (sentiment["publishedAt"] >= pd.to_datetime(start_date)) & (
                sentiment["publishedAt"] <= pd.to_datetime(end_date)
            )
            sentiment = sentiment[mask].copy()

        # ── AI Signal banner ──────────────────────────────────────────────────

        if not prices.empty and not raw_sentiment.empty:
            signal_label, signal_reason, signal_color = compute_ai_signal(raw_sentiment, prices)
            st.markdown(
                f"""<div class="signal-banner" style="background:{signal_color}18;border-left:5px solid {signal_color};">
                <span class="signal-label" style="color:{signal_color};">Signal IA : {signal_label}</span>
                <span class="signal-reason">{signal_reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── KPI cards ─────────────────────────────────────────────────────────

        counts = raw_sentiment["sentiment"].value_counts().to_dict()
        total_news = len(raw_sentiment)
        pos_count = counts.get("positive", 0)
        neg_count = counts.get("negative", 0)
        neu_count = counts.get("neutral", 0)
        sent_score = float(raw_sentiment["positive"].mean() - raw_sentiment["negative"].mean())

        rsi_series = calculate_rsi(prices).dropna()
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        rsi_color = "#EF553B" if rsi_val and rsi_val > 70 else "#00CC96" if rsi_val and rsi_val < 30 else "#aaa"

        close_series = prices["Close"].dropna()
        if len(close_series) >= 2:
            lookback = min(22, len(close_series) - 1)
            month_chg = (float(close_series.iloc[-1]) - float(close_series.iloc[-lookback])) / float(close_series.iloc[-lookback]) * 100
            month_arrow = "▲" if month_chg >= 0 else "▼"
            month_sub = f"{month_arrow} {month_chg:+.1f}% (1m)"
        else:
            month_sub = "—"

        pos_pct_str = f"{int(100*pos_count/total_news) if total_news else 0}%"
        neg_pct_str = f"{int(100*neg_count/total_news) if total_news else 0}%"
        rsi_sub = "surachat" if rsi_val and rsi_val > 70 else "survente" if rsi_val and rsi_val < 30 else "neutre"

        kpi_data = [
            ("📰 Total News",  str(total_news),                              "#a0b8d8",   "—"),
            ("🟢 Positif",     str(pos_count),                               "#2ecc8f",   pos_pct_str),
            ("🔴 Négatif",     str(neg_count),                               "#e05555",   neg_pct_str),
            ("📊 RSI (14j)",   f"{rsi_val:.1f}" if rsi_val else "—",         rsi_color,   rsi_sub),
            ("🎯 Score Sent.", f"{sent_score:+.3f}",                         "#2ecc8f" if sent_score > 0 else "#e05555",   month_sub),
        ]

        cols = st.columns(len(kpi_data))
        for col, (label, value, color, sub) in zip(cols, kpi_data):
            col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="color:{color};">{value}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Claude AI summary ─────────────────────────────────────────────────

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        st.subheader("Résumé IA — Claude")
        if ANTHROPIC_AVAILABLE and api_key:
            gen_col, _ = st.columns([1, 5])
            if gen_col.button("🤖 Générer", key="gen_summary") or (
                st.session_state.ai_summary and st.session_state.ai_summary_ticker == ticker
            ):
                if not st.session_state.ai_summary or st.session_state.ai_summary_ticker != ticker:
                    with st.spinner("Claude analyse le marché…"):
                        st.session_state.ai_summary = generate_market_summary(ticker, raw_sentiment, prices)
                        st.session_state.ai_summary_ticker = ticker
                if st.session_state.ai_summary:
                    st.markdown(
                        f'<div class="claude-box">{st.session_state.ai_summary}</div>',
                        unsafe_allow_html=True,
                    )
        elif not ANTHROPIC_AVAILABLE:
            st.info("📦 Package `anthropic` non installé — `pip install anthropic`")
        else:
            st.info("🔑 Définissez `ANTHROPIC_API_KEY` pour activer le résumé IA Claude.")

        # ── Price chart ───────────────────────────────────────────────────────

        st.subheader(f"Évolution du cours : {ticker}")
        data_for_chart = (prices_range if not prices_range.empty else prices).copy()

        if show_technical_indicators:
            data_for_chart["SMA_20"] = data_for_chart["Close"].rolling(window=20).mean()
            data_for_chart["SMA_50"] = data_for_chart["Close"].rolling(window=50).mean()

        if show_bollinger:
            bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(data_for_chart)
            data_for_chart["BB_mid"] = bb_mid
            data_for_chart["BB_upper"] = bb_upper
            data_for_chart["BB_lower"] = bb_lower

        if show_rsi:
            data_for_chart["RSI"] = calculate_rsi(data_for_chart)

        if show_macd:
            macd, macd_sig, macd_hist = calculate_macd(data_for_chart)
            data_for_chart["MACD"] = macd
            data_for_chart["MACD_signal"] = macd_sig
            data_for_chart["MACD_hist"] = macd_hist

        sub_panels = []
        if show_rsi:
            sub_panels.append("RSI (14)")
        if show_macd:
            sub_panels.append("MACD")

        n_rows = 1 + len(sub_panels)
        if n_rows == 1:
            row_heights = [1.0]
        elif n_rows == 2:
            row_heights = [0.68, 0.32]
        else:
            row_heights = [0.5, 0.25, 0.25]

        panel_titles = [f"Prix — {ticker}"] + sub_panels
        specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * len(sub_panels)
        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            subplot_titles=panel_titles, row_heights=row_heights,
            vertical_spacing=0.04, specs=specs,
        )

        fig.add_trace(
            go.Scatter(
                x=data_for_chart.index, y=data_for_chart["Close"],
                mode="lines", name="Prix clôture",
                line=dict(color="#5B9BD5", width=1.5),
            ),
            row=1, col=1,
        )

        if show_technical_indicators:
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["SMA_20"],
                           mode="lines", name="SMA 20", line=dict(color="orange", width=1.5)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["SMA_50"],
                           mode="lines", name="SMA 50", line=dict(color="red", width=1.5)),
                row=1, col=1,
            )

        if show_bollinger:
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["BB_upper"],
                           mode="lines", name="BB Upper", line=dict(color="lightblue", width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data_for_chart.index, y=data_for_chart["BB_lower"],
                    mode="lines", name="BB Lower", line=dict(color="lightblue", width=1),
                    fill="tonexty", fillcolor="rgba(173,216,230,0.12)",
                ),
                row=1, col=1,
            )

        if show_sentiment_overlay and "publishedAt" in raw_sentiment.columns:
            _sol = raw_sentiment.copy()
            _sol["publishedAt"] = pd.to_datetime(_sol["publishedAt"], errors="coerce")
            _sol = _sol.dropna(subset=["publishedAt"])
            _sol["date"] = _sol["publishedAt"].dt.normalize()
            _sol["score"] = _sol["positive"] - _sol["negative"]
            _daily_sol = _sol.groupby("date")["score"].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=_daily_sol["date"], y=_daily_sol["score"],
                    mode="lines+markers", name="Sentiment",
                    line=dict(color="rgba(255,165,0,0.8)", width=1.5, dash="dot"),
                    marker=dict(size=4),
                ),
                row=1, col=1, secondary_y=True,
            )
            fig.update_yaxes(title_text="Sentiment", secondary_y=True, row=1, col=1,
                             range=[-1, 1], showgrid=False)

        current_row = 2
        if show_rsi:
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["RSI"],
                           mode="lines", name="RSI", line=dict(color="purple")),
                row=current_row, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, annotation_text="Surachat")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, annotation_text="Survente")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            current_row += 1

        if show_macd:
            hist_colors = ["#00CC96" if v >= 0 else "#EF553B" for v in data_for_chart["MACD_hist"].fillna(0)]
            fig.add_trace(
                go.Bar(x=data_for_chart.index, y=data_for_chart["MACD_hist"],
                       name="Histogramme", marker_color=hist_colors, opacity=0.6),
                row=current_row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["MACD"],
                           mode="lines", name="MACD", line=dict(color="#5B9BD5", width=1.2)),
                row=current_row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["MACD_signal"],
                           mode="lines", name="Signal", line=dict(color="orange", width=1.2)),
                row=current_row, col=1,
            )
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)

        fig.update_yaxes(title_text="Prix ($)", secondary_y=False, row=1, col=1)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=460 + 160 * len(sub_panels),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_volume and "Volume" in data_for_chart.columns:
            st.subheader("Volume des échanges")
            vol = pd.to_numeric(data_for_chart["Volume"], errors="coerce")
            fig_vol = go.Figure(
                go.Bar(x=data_for_chart.index, y=vol, marker_color="rgba(99,110,250,0.5)", name="Volume")
            )
            fig_vol.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, xaxis_title="Date", yaxis_title="Volume",
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        if prices_range.empty:
            st.warning("Aucune donnée de prix disponible pour la plage sélectionnée.")

        # ── Sentiment gauge + news cards ──────────────────────────────────────

        if sentiment.empty:
            st.warning("Aucun article ne correspond au filtre. Changez le filtre ou relancez l'analyse.")
        else:
            sentiment["score"] = sentiment["positive"] - sentiment["negative"]

            gauge_score = float(raw_sentiment["positive"].mean() - raw_sentiment["negative"].mean())
            gauge_val = (gauge_score + 1) / 2 * 100

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Humeur du Marché")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_val,
                    number={"suffix": "/100", "font": {"color": "#dde", "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#444", "tickfont": {"color": "#666"}},
                        "bar": {"color": "#636EFA", "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 33],  "color": "rgba(239,85,59,0.18)"},
                            {"range": [33, 66], "color": "rgba(99,110,250,0.10)"},
                            {"range": [66, 100],"color": "rgba(0,204,150,0.18)"},
                        ],
                        "threshold": {
                            "line": {"color": "#fff", "width": 2},
                            "thickness": 0.75,
                            "value": gauge_val,
                        },
                    },
                    title={"text": "Score sentiment global", "font": {"color": "#888", "size": 12}},
                ))
                fig_gauge.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=260,
                    margin=dict(t=50, b=10, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                fig_pie = px.pie(
                    raw_sentiment, names="sentiment", color="sentiment",
                    color_discrete_map={"positive": "#00CC96", "neutral": "#636EFA", "negative": "#EF553B"},
                    hole=0.5,
                )
                fig_pie.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True, height=220, margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("Dernières actualités")
                st.markdown(render_news_cards(sentiment), unsafe_allow_html=True)

            csv_export = sentiment.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Exporter les données de sentiment (CSV)",
                data=csv_export,
                file_name=f"{ticker}_sentiment_export.csv",
                mime="text/csv",
            )

            # ── Top news ──────────────────────────────────────────────────────

            top_positive = sentiment.nlargest(3, "positive")[["title", "positive", "negative", "score"]]
            top_negative = sentiment.nlargest(3, "negative")[["title", "positive", "negative", "score"]]
            st.subheader("Actualités clés")
            tp_col, tn_col = st.columns(2)
            with tp_col:
                st.markdown("**Top 3 positifs**")
                st.dataframe(top_positive, use_container_width=True, hide_index=True)
            with tn_col:
                st.markdown("**Top 3 négatifs**")
                st.dataframe(top_negative, use_container_width=True, hide_index=True)

            # ── Sentiment over time ───────────────────────────────────────────

            if "publishedAt" in sentiment.columns:
                sentiment_time = sentiment.dropna(subset=["publishedAt"]).copy()
                if not sentiment_time.empty:
                    sentiment_time = (
                        sentiment_time.set_index("publishedAt")
                        .resample("D")
                        .agg(score=("score", "mean"), articles=("title", "count"))
                        .reset_index()
                    )

                    st.subheader("Évolution du sentiment dans le temps")
                    fig_time = px.line(
                        sentiment_time, x="publishedAt", y="score", markers=True,
                        labels={"publishedAt": "Date", "score": "Score moyen"},
                        template="plotly_dark",
                    )
                    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_time, use_container_width=True)

                    st.subheader("Volume de news par jour")
                    fig_news_vol = px.bar(
                        sentiment_time, x="publishedAt", y="articles",
                        labels={"publishedAt": "Date", "articles": "Nombre d'articles"},
                        template="plotly_dark",
                    )
                    fig_news_vol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_news_vol, use_container_width=True)

            # ── Rolling correlation ───────────────────────────────────────────

            rolling_corr, actual_window = compute_rolling_correlation(prices, raw_sentiment)
            n_days = raw_sentiment["publishedAt"].apply(
                lambda x: pd.to_datetime(x, errors="coerce", utc=True)
            ).dropna().dt.tz_localize(None).dt.normalize().nunique() if "publishedAt" in raw_sentiment.columns else 0

            st.subheader(f"Corrélation Sentiment & Rendements (rolling {actual_window}j)")
            if rolling_corr is not None and not rolling_corr["rolling_corr"].dropna().empty:
                fig_corr = px.line(
                    rolling_corr.dropna(subset=["rolling_corr"]),
                    x="date", y="rolling_corr",
                    labels={"date": "Date", "rolling_corr": f"Corrélation ({actual_window}j)"},
                    template="plotly_dark",
                )
                fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_corr.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Corrélation forte +")
                fig_corr.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Corrélation forte −")
                fig_corr.update_yaxes(range=[-1, 1])
                st.plotly_chart(fig_corr, use_container_width=True)
                caption = (
                    f"Corrélation de Pearson glissante sur {actual_window} jours "
                    f"({n_days} jours de news disponibles — fenêtre s'élargit automatiquement à chaque analyse)."
                )
                if actual_window < 14:
                    caption += " Relancez l'analyse dans quelques jours pour enrichir l'historique."
                st.caption(caption)
            else:
                st.info(
                    f"📅 {n_days} jour(s) de news collectés — il en faut au moins 3. "
                    "Chaque analyse accumule de nouveaux articles : relancez dans quelques jours."
                )

    else:
        st.info("👋 Entrez un ticker dans la barre latérale et cliquez sur 'Lancer l'analyse IA'.")

except Exception as e:
    st.error(f"Erreur lors de l'affichage : {e}")
    import traceback
    st.code(traceback.format_exc())

st.divider()
st.caption("Projet Data & IA · Modèle : ProsusAI/FinBERT · Source : Yahoo Finance")
