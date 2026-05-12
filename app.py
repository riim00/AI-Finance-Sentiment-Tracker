import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf

from src.ingestion import fetch_stock_data, fetch_news
from src.processing import analyze_sentiment

st.set_page_config(page_title="AI Finance Sentiment Tracker", layout="wide")


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


def compute_rolling_correlation(prices_df, sentiment_df, window=14):
    """Rolling correlation (14j) entre score de sentiment journalier et rendements."""
    if "publishedAt" not in sentiment_df.columns:
        return None

    sent = sentiment_df.copy()
    sent["publishedAt"] = pd.to_datetime(sent["publishedAt"], errors="coerce")
    sent = sent.dropna(subset=["publishedAt"])
    sent["date"] = sent["publishedAt"].dt.normalize()
    sent["score"] = sent["positive"] - sent["negative"]
    daily_sent = sent.groupby("date")["score"].mean().reset_index()

    returns = prices_df["Close"].pct_change()
    returns.index = pd.to_datetime(returns.index).normalize()
    returns_df = returns.reset_index()
    returns_df.columns = ["date", "returns"]

    merged = pd.merge(daily_sent, returns_df, on="date", how="inner").dropna()
    if len(merged) < window:
        return None

    merged = merged.sort_values("date")
    merged["rolling_corr"] = merged["score"].rolling(window).corr(merged["returns"])
    return merged


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
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

# Session state init
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "ticker_override" not in st.session_state:
    st.session_state.ticker_override = None

# Ticker resolution: custom > override (from favorites) > selectbox
if custom_ticker.strip():
    ticker = custom_ticker.strip().upper()
    st.session_state.ticker_override = None
elif st.session_state.ticker_override:
    ticker = st.session_state.ticker_override
else:
    ticker = selected_ticker if selected_ticker else "NVDA"

# Validate custom ticker
if custom_ticker.strip():
    is_valid, suggestions = validate_ticker(ticker)
    if not is_valid:
        st.sidebar.error(
            f"Ticker '{ticker}' non trouvé. Suggestions : {', '.join(suggestions) if suggestions else 'Aucun'}"
        )
        ticker = selected_ticker if selected_ticker else "NVDA"

# Favorites
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

run_button = st.sidebar.button("Lancer l'analyse IA", type="primary")

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.title(f"📈 AI Finance Sentiment Tracker : {ticker}")
st.write(f"Analyse en temps réel pour **{ticker}** — Modèle : **FinBERT** (NLP financier)")

# ─── ANALYSIS ───────────────────────────────────────────────────────────────
if run_button:
    with st.spinner(f"Analyse en cours pour {ticker}…"):
        fetch_stock_data(ticker, start=str(start_date), force_refresh=force_refresh)
        fetch_news(ticker, force_refresh=force_refresh)
        analyze_sentiment(ticker)
        load_price_data.clear()
        load_sentiment_data.clear()
        st.sidebar.success("Analyse terminée !")

# ─── RESULTS ────────────────────────────────────────────────────────────────
try:
    price_path = f"data/{ticker}_prices.csv"
    sent_path = f"data/{ticker}_sentiment.csv"

    if os.path.exists(price_path) and os.path.exists(sent_path):
        prices = load_price_data(price_path)
        raw_sentiment = load_sentiment_data(sent_path)

        # Filter sentiment
        if sentiment_filter != "Tous":
            sentiment = raw_sentiment[raw_sentiment["sentiment"] == sentiment_filter].copy()
        else:
            sentiment = raw_sentiment.copy()

        # Filter prices by date range
        prices = prices.sort_index()
        prices_range = prices.loc[str(start_date) : str(end_date)]

        # Filter sentiment by date range
        if "publishedAt" in sentiment.columns:
            sentiment["publishedAt"] = pd.to_datetime(sentiment["publishedAt"], errors="coerce")
            mask = (sentiment["publishedAt"] >= pd.to_datetime(start_date)) & (
                sentiment["publishedAt"] <= pd.to_datetime(end_date)
            )
            sentiment = sentiment[mask].copy()

        # ── AI SIGNAL ────────────────────────────────────────────────────────
        if not prices.empty and not raw_sentiment.empty:
            signal_label, signal_reason, signal_color = compute_ai_signal(raw_sentiment, prices)
            st.markdown(
                f"""<div style="background:{signal_color}22;border-left:5px solid {signal_color};
                padding:12px 20px;border-radius:6px;margin-bottom:16px;">
                <span style="font-size:1.15em;font-weight:700;color:{signal_color};">
                Signal IA : {signal_label}</span>
                &nbsp;&nbsp;<span style="color:#aaa;font-size:0.9em;">{signal_reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── SUMMARY METRICS ──────────────────────────────────────────────────
        counts = raw_sentiment["sentiment"].value_counts().to_dict()
        st.subheader("Synthèse du sentiment")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total news", len(raw_sentiment))
        m2.metric("Positif", counts.get("positive", 0))
        m3.metric("Neutre", counts.get("neutral", 0))
        m4.metric("Négatif", counts.get("negative", 0))

        # ── PRICE CHART ──────────────────────────────────────────────────────
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

        # Build subplot layout dynamically
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
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=panel_titles,
            row_heights=row_heights,
            vertical_spacing=0.04,
        )

        fig.add_trace(
            go.Scatter(
                x=data_for_chart.index,
                y=data_for_chart["Close"],
                mode="lines",
                name="Prix clôture",
                line=dict(color="#5B9BD5", width=1.5),
            ),
            row=1,
            col=1,
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

        # Add RSI and MACD to their respective sub-panels
        current_row = 2
        if show_rsi:
            fig.add_trace(
                go.Scatter(x=data_for_chart.index, y=data_for_chart["RSI"],
                           mode="lines", name="RSI", line=dict(color="purple")),
                row=current_row, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1,
                          annotation_text="Surachat")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1,
                          annotation_text="Survente")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            current_row += 1

        if show_macd:
            hist_colors = [
                "#00CC96" if v >= 0 else "#EF553B"
                for v in data_for_chart["MACD_hist"].fillna(0)
            ]
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

        fig.update_yaxes(title_text="Prix ($)", row=1, col=1)
        fig.update_layout(
            template="plotly_dark",
            height=460 + 160 * len(sub_panels),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        if show_volume and "Volume" in data_for_chart.columns:
            st.subheader("Volume des échanges")
            vol = pd.to_numeric(data_for_chart["Volume"], errors="coerce")
            fig_vol = go.Figure(
                go.Bar(x=data_for_chart.index, y=vol, marker_color="rgba(99,110,250,0.5)", name="Volume")
            )
            fig_vol.update_layout(template="plotly_dark", height=250,
                                  xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True)

        if prices_range.empty:
            st.warning("Aucune donnée de prix disponible pour la plage sélectionnée.")

        # ── SENTIMENT ────────────────────────────────────────────────────────
        if sentiment.empty:
            st.warning("Aucun article ne correspond au filtre. Changez le filtre ou relancez l'analyse.")
        else:
            sentiment["score"] = sentiment["positive"] - sentiment["negative"]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Humeur du Marché")
                fig_pie = px.pie(
                    raw_sentiment,
                    names="sentiment",
                    color="sentiment",
                    color_discrete_map={"positive": "#00CC96", "neutral": "#636EFA", "negative": "#EF553B"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("Dernières actualités")
                display_cols = ["title", "sentiment", "positive", "negative", "score"]
                col_config = {
                    "title": st.column_config.TextColumn("Titre"),
                    "positive": st.column_config.NumberColumn("Positif", format="%.3f"),
                    "negative": st.column_config.NumberColumn("Négatif", format="%.3f"),
                    "score": st.column_config.NumberColumn("Score", format="%.3f"),
                }
                if "link" in sentiment.columns:
                    display_cols.insert(1, "link")
                    col_config["link"] = st.column_config.LinkColumn("Lien")

                st.data_editor(
                    sentiment[display_cols],
                    column_config=col_config,
                    hide_index=True,
                    use_container_width=True,
                    disabled=True,
                    key="news_table",
                )

            # Export button
            csv_export = sentiment.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Exporter les données de sentiment (CSV)",
                data=csv_export,
                file_name=f"{ticker}_sentiment_export.csv",
                mime="text/csv",
            )

            # Top news
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

            # Sentiment over time
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
                    fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_time, use_container_width=True)

                    st.subheader("Volume de news par jour")
                    fig_news_vol = px.bar(
                        sentiment_time, x="publishedAt", y="articles",
                        labels={"publishedAt": "Date", "articles": "Nombre d'articles"},
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_news_vol, use_container_width=True)

            # ── ROLLING CORRELATION ──────────────────────────────────────────
            st.subheader("Corrélation Sentiment & Rendements (rolling 14j)")
            rolling_corr = compute_rolling_correlation(prices, raw_sentiment)
            if rolling_corr is not None and not rolling_corr["rolling_corr"].dropna().empty:
                fig_corr = px.line(
                    rolling_corr.dropna(subset=["rolling_corr"]),
                    x="date", y="rolling_corr",
                    labels={"date": "Date", "rolling_corr": "Corrélation (14j)"},
                    template="plotly_dark",
                )
                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_corr.add_hline(y=0.5, line_dash="dot", line_color="green",
                                   annotation_text="Corrélation forte +")
                fig_corr.add_hline(y=-0.5, line_dash="dot", line_color="red",
                                   annotation_text="Corrélation forte −")
                fig_corr.update_yaxes(range=[-1, 1])
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption(
                    "Corrélation de Pearson glissante sur 14 jours entre le score de sentiment "
                    "FinBERT et les rendements journaliers du cours."
                )
            else:
                avg_sent = float(raw_sentiment["positive"].mean() - raw_sentiment["negative"].mean())
                avg_price = (
                    float(prices_range["Close"].mean())
                    if not prices_range.empty
                    else float(prices["Close"].mean())
                )
                st.write(f"Score sentiment moyen : **{avg_sent:.3f}** | Prix moyen : **{avg_price:.2f} $**")
                st.info(
                    "Données temporelles insuffisantes pour la corrélation glissante. "
                    "Relancez l'analyse avec un historique plus large."
                )

    else:
        st.info("👋 Entrez un ticker dans la barre latérale et cliquez sur 'Lancer l'analyse IA'.")

except Exception as e:
    st.error(f"Erreur lors de l'affichage : {e}")
    import traceback
    st.code(traceback.format_exc())

st.divider()
st.caption("Projet Data & IA · Modèle : ProsusAI/FinBERT · Source : Yahoo Finance")
