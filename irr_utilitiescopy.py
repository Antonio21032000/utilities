import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# Tenta importar numpy_financial
try:
    import numpy_financial as npf
except Exception:
    npf = None

# Configurar página do Streamlit
st.set_page_config(
    page_title="IRR Real - Dashboard Completo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Definir as cores da STK
STK_AZUL = "#102E46"
STK_DOURADO = "#C98C2E"
STK_BRANCO = "#FFFFFF"

# Estilo CSS personalizado
st.markdown("""
    <style>
        .stApp {
            background-color: #102E46;
            color: white;
        }
        
        .title-container {
            background-color: #C98C2E;
            padding: 1.5rem;
            margin: 0;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .title-text {
            color: white !important;
            font-size: 2.8em;
            font-weight: 600;
            margin: 0;
            padding: 0;
            font-family: sans-serif;
            letter-spacing: 1px;
        }
        
        /* Forçar texto branco em todos os elementos */
        .stMarkdown, .stText, p, div, span {
            color: white !important;
        }
        
        /* Spinner com texto branco */
        .stSpinner > div {
            color: white !important;
        }
        
        /* Texto de loading */
        .stSpinner > div > div {
            color: white !important;
        }
        
        /* Melhorar visibilidade de todos os textos */
        .st-emotion-cache-1y4p8pa {
            color: white !important;
        }
        
        .st-emotion-cache-16txtl3 {
            color: white !important;
        }
        
        /* Texto de erro */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* Forçar contraste alto */
        * {
            color: white !important;
        }
        
        /* Exceções para elementos que devem manter cor original */
        .title-text, .stPlotlyChart {
            color: inherit !important;
        }
    </style>
""", unsafe_allow_html=True)

# Título personalizado
st.markdown("""
    <div class="title-container">
        <h1 class="title-text" style="color: white !important;">IRR Real - Dashboard Completo</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

def compute_irr(cashflows: np.ndarray) -> float:
    """Calcula IRR usando numpy_financial ou método de bisseção como fallback"""
    values = np.asarray(cashflows, dtype=float)
    if npf is not None:
        try:
            irr_val = float(npf.irr(values))
            return irr_val
        except Exception:
            pass

    def npv(rate: float) -> float:
        periods = np.arange(values.shape[0], dtype=float)
        return float(np.sum(values / (1.0 + rate) ** periods))

    low, high = -0.99, 10.0
    f_low, f_high = npv(low), npv(high)
    if np.sign(f_low) == np.sign(f_high):
        return np.nan

    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if np.sign(f_low) * np.sign(f_mid) <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return mid

def compute_xirr(cashflows: np.ndarray, dates: list, guess: float = 0.1) -> float:
    """
    Calcula XIRR considerando datas específicas (similar ao Excel)
    """
    if len(cashflows) != len(dates):
        return np.nan

    values = np.asarray(cashflows, dtype=float)
    dates = pd.to_datetime(dates)
    base_date = dates[0]
    years = [(d - base_date).days / 365.25 for d in dates]

    def xnpv(rate):
        return sum(cf / (1 + rate) ** year for cf, year in zip(values, years))

    rate = guess
    for _ in range(100):
        npv_val = xnpv(rate)
        if abs(npv_val) < 1e-6:
            return rate
        delta = 1e-6
        d_npv = (xnpv(rate + delta) - xnpv(rate - delta)) / (2 * delta)
        if abs(d_npv) < 1e-10:
            return np.nan
        rate = rate - npv_val / d_npv
        if rate < -0.99 or rate > 100:
            return np.nan
    return rate

def cap_to_first_digits_mln(value, digits=6):
    """Converte market cap para milhões e mantém apenas os primeiros dígitos"""
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])

# Executar o código principal
try:
    with st.spinner("🔄 Carregando dados e calculando XIRR..."):
        
        # Tickets a baixar do Yahoo (apenas os necessários para preços)
        tickers_for_prices = [
            # Consolidações por classes
            "CPLE3", "CPLE6",      # Copel
            "IGTI3", "IGTI4",      # Iguatemi
            "ENGI3", "ENGI4",      # Energisa
            # Demais nomes individuais
            "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3"
        ]

        tickers_sa = [f"{t}.SA" for t in tickers_for_prices]
        closes = yf.download(tickers_sa, period="5d", progress=False)["Close"].ffill()
        last_close = closes.iloc[-1]
        last_close.index = [t.replace(".SA", "") for t in last_close.index]
        prices = last_close.astype(float)

        # Quantidade de ações por classe (fornecidas)
        shares_classes = {
            # Copel
            "CPLE3": 1_300_347_300,
            "CPLE6": 1_679_335_290,
            # Iguatemi
            "IGTI3": 770_992_429,
            "IGTI4": 435_368_756,
            # Energisa
            "ENGI3": 887_231_247,
            "ENGI4": 1_402_193_416,
            # Demais (1 classe)
            "EQTL3": 1_255_510_000,
            "SBSP3": 683_510_000,
            "NEOE3": 1_213_800_000,
            "ENEV3": 1_936_970_000,
            "ELET3": 2_308_630_000,
            "EGIE3": 815_928_000,
            "MULT3": 513_164_000,
            "ALOS3": 542_937_000,
        }

        shares_series = pd.Series(shares_classes).reindex(prices.index)
        mc_raw = prices * shares_series

        # --- Consolidações de market cap ---
        # Copel -> representante: CPLE6
        if {"CPLE3", "CPLE6"}.issubset(mc_raw.index):
            cple_total = mc_raw["CPLE3"] + mc_raw["CPLE6"]
        else:
            st.error("❌ Preços de CPLE3/CPLE6 não encontrados.")
            st.stop()

        # Iguatemi -> alimentar IGTI11
        if {"IGTI3", "IGTI4"}.issubset(mc_raw.index):
            igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
        else:
            st.error("❌ Preços de IGTI3/IGTI4 não encontrados.")
            st.stop()

        # Energisa -> alimentar ENGI11
        if {"ENGI3", "ENGI4"}.issubset(mc_raw.index):
            engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
        else:
            st.error("❌ Preços de ENGI3/ENGI4 não encontrados.")
            st.stop()

        # --- Monta a tabela final 'resultado' com os tickers da planilha ---
        final_tickers = [
            "CPLE6", "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3",
            "MULT3", "ALOS3", "IGTI11", "ENGI11"
        ]

        rows = []
        for t in final_tickers:
            if t == "CPLE6":
                price = prices.get("CPLE6", np.nan)
                shares = shares_classes["CPLE3"] + shares_classes["CPLE6"]
                mc = cple_total
            elif t == "IGTI11":
                price = np.nan  # não usamos preço direto do Yahoo
                shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]  # apenas referência
                mc = igti_total
            elif t == "ENGI11":
                price = np.nan
                shares = shares_classes["ENGI3"] + shares_classes["ENGI4"]
                mc = engi_total
            else:
                price = prices.get(t, np.nan)
                shares = shares_classes.get(t, np.nan)
                mc = price * shares if (pd.notna(price) and pd.notna(shares)) else np.nan

            rows.append({"ticker": t, "price": price, "shares": shares, "market_cap": mc})

        resultado = pd.DataFrame(rows).set_index("ticker")
        # Reduz market_cap para milhões e mantém 6 primeiros dígitos
        resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

        st.markdown("### 📊 Dados de Market Cap (em milhões, 6 primeiros dígitos)")
        st.dataframe(resultado, use_container_width=True)

        # === Carrega Excel com fluxos de caixa ===
        try:
            df = pd.read_excel('irrdash3.xlsx')
        except FileNotFoundError:
            st.error("❌ Arquivo 'irrdash3.xlsx' não encontrado. Coloque-o no diretório de execução.")
            st.stop()

        # Primeira linha como cabeçalho
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # Garante colunas para todos os tickers de 'resultado'
        for t in resultado.index:
            if t not in df.columns:
                df[t] = pd.NA

        # Escreve market caps (negativos) na primeira linha de dados
        target_row = df.index[0]
        today = datetime.now().date()
        for ticker in resultado.index:
            mc_val = resultado.loc[ticker, 'market_cap']
            df.loc[target_row, ticker] = -abs(mc_val)

        # Garante numéricos nas colunas dos tickers
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors='coerce')

        st.markdown("### 💰 Fluxos de Caixa por Empresa")
        cashflow_display = df[resultado.index].copy()
        st.dataframe(cashflow_display, use_container_width=True)

        # === XIRR por ticker (sem zeragem em 2025-12-31) ===
        st.markdown("### 🧮 Calculando XIRR para cada empresa (com datas específicas)")
        
        irr_results = {}
        details_container = st.container()
        
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue

            cashflows = series_cf.values.astype(float).copy()
            n_periods = len(cashflows)

            # Datas: hoje (market cap) + 31/12 de cada ano subsequente
            dates_list = [today]
            for i in range(1, n_periods):
                future_dt = date(today.year + i - 1, 12, 31)
                dates_list.append(future_dt)

            cf_with_dates = pd.DataFrame({
                'Data': dates_list,
                'Fluxo_de_Caixa': cashflows,
                'Ano': [d.year for d in dates_list]
            })

            with details_container:
                st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
                st.markdown(f"**📊 {t} - Fluxos de caixa com datas (para XIRR):**")
                st.dataframe(cf_with_dates, use_container_width=True)

            irr_results[t] = compute_xirr(cashflows, dates_list)
            
            with details_container:
                if pd.isna(irr_results[t]):
                    st.markdown("  ➤ **XIRR calculado:** N/A")
                else:
                    st.markdown(f"  ➤ **XIRR calculado:** {irr_results[t]:.4f} ({irr_results[t]*100:.2f}%)")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")

        ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])
        ytm_df['irr_aj'] = ytm_df['irr']  # inicial

        # Ajuste de IRR real SOMENTE para MULT3, ALOS3 e IGTI11 (deflator 4,5% a.a.)
        st.markdown("### 🔧 Aplicando ajuste de IRR real (deflator 4,5% a.a.)")
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Ajuste aplicado em:** MULT3, ALOS3 e IGTI11")
        
        for ticker_adj in ['MULT3', 'ALOS3', 'IGTI11']:
            if ticker_adj in ytm_df.index and not pd.isna(ytm_df.loc[ticker_adj, 'irr']):
                ytm_df.loc[ticker_adj, 'irr_aj'] = ((1 + ytm_df.loc[ticker_adj, 'irr']) / (1 + 0.045)) - 1
                st.markdown(f"  **{ticker_adj}:** {ytm_df.loc[ticker_adj, 'irr_aj']*100:.2f}% (real)")
        
        st.markdown("</div>", unsafe_allow_html=True)

        ytm_clean = ytm_df[['irr_aj']].dropna().sort_values('irr_aj', ascending=True)

        st.markdown("### 📈 Resultados Finais - IRR (ajustada onde aplicável)")
        
        if len(ytm_clean) > 0:
            # Criar colunas para layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Criar paleta de cores
                colors = [
                    "#708090", "#A9A9A9", "#BC987E", "#87A96B", "#8FBC8F",
                    "#D2B48C", "#DEB887", "#B0C4DE", "#C0C0C0", "#98A2B3", "#8B9DC3"
                ]
                
                plot_data = pd.DataFrame({
                    'empresa': ytm_clean.index,
                    'irr': ytm_clean['irr_aj'] * 100,
                })

                fig_irr = px.bar(
                    plot_data,
                    x='empresa',
                    y='irr',
                    title="IRR por Empresa (ajustada onde aplicável)",
                    color='empresa',
                    color_discrete_sequence=colors,
                    text='irr'
                )

                fig_irr.update_traces(
                    texttemplate='%{text:.2f}%',
                    textposition='outside',
                    textfont=dict(color='white', size=12)
                )

                fig_irr.update_layout(
                    plot_bgcolor=STK_AZUL,
                    paper_bgcolor=STK_AZUL,
                    font_color='white',
                    xaxis_title="Empresas",
                    yaxis_title="IRR (%)",
                    height=600,
                    showlegend=False,
                    margin=dict(t=50, b=50, l=50, r=50),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        tickfont=dict(color='white', size=12),
                        title_font=dict(color='white', size=14)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        tickfont=dict(color='white', size=12),
                        tickformat='.2f',
                        title_font=dict(color='white', size=14)
                    )
                )

                st.plotly_chart(fig_irr, use_container_width=True)
            
            with col2:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("#### 🏆 Ranking IRR")
                
                for i, (empresa, row) in enumerate(ytm_clean.iterrows(), 1):
                    emoji = "🥇" if i == len(ytm_clean) else "🥈" if i == len(ytm_clean)-1 else "🥉" if i == len(ytm_clean)-2 else "📊"
                    st.markdown(f"{emoji} **{empresa}:** {row['irr_aj']*100:.2f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Resumo estatístico
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("#### 📊 Estatísticas")
                st.markdown(f"**Maior IRR:** {ytm_clean.iloc[-1].name} ({ytm_clean.iloc[-1]['irr_aj']*100:.2f}%)")
                st.markdown(f"**Menor IRR:** {ytm_clean.iloc[0].name} ({ytm_clean.iloc[0]['irr_aj']*100:.2f}%)")
                st.markdown(f"**IRR Média:** {ytm_clean['irr_aj'].mean()*100:.2f}%")
                st.markdown(f"**IRR Mediana:** {ytm_clean['irr_aj'].median()*100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("⚠️ Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

    # Rodapé com aviso para refresh
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: bold;
                    background-color: rgba(201, 140, 46, 0.1); padding: 15px; border-radius: 8px; margin-top: 20px;'>
            💡 <strong>Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página</strong>
        </div>
        """, 
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("📁 Arquivo 'irrdash3.xlsx' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
except Exception as e:
    st.error(f"❌ Erro durante a execução: {str(e)}")


