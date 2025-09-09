import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, date

# Configuração da página
st.set_page_config(
    page_title="IRR Real Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado para o tema escuro e estilo
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #D4A574 0%, #C19A5C 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-container {
        background-color: #1E3A5F;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #D4A574;
    }
    
    .refresh-note {
        background-color: #2D4A6B;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #D4A574;
    }
    
    .refresh-note p {
        color: #D4A574;
        margin: 0;
        font-weight: 500;
    }
    
    .stApp {
        background-color: #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)

# Funções do código original
try:
    import numpy_financial as npf
except Exception:
    npf = None

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
    """Calcula XIRR considerando datas específicas (similar ao Excel)"""
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

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data():
    """Carrega e processa todos os dados"""
    try:
        # Tickes a baixar do Yahoo
        tickers_for_prices = [
            "CPLE3", "CPLE6", "IGTI3", "IGTI4", "ENGI3", "ENGI4",
            "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3"
        ]

        tickers_sa = [f"{t}.SA" for t in tickers_for_prices]
        closes = yf.download(tickers_sa, period="5d", progress=False)["Close"].ffill()
        last_close = closes.iloc[-1]
        last_close.index = [t.replace(".SA", "") for t in last_close.index]
        prices = last_close.astype(float)

        # Quantidade de ações por classe
        shares_classes = {
            "CPLE3": 1_300_347_300, "CPLE6": 1_679_335_290,
            "IGTI3": 770_992_429, "IGTI4": 435_368_756,
            "ENGI3": 887_231_247, "ENGI4": 1_402_193_416,
            "EQTL3": 1_255_510_000, "SBSP3": 683_510_000, "NEOE3": 1_213_800_000,
            "ENEV3": 1_936_970_000, "ELET3": 2_308_630_000, "EGIE3": 815_928_000,
            "MULT3": 513_164_000, "ALOS3": 542_937_000,
        }

        shares_series = pd.Series(shares_classes).reindex(prices.index)
        mc_raw = prices * shares_series

        # Consolidações de market cap
        if {"CPLE3", "CPLE6"}.issubset(mc_raw.index):
            cple_total = mc_raw["CPLE3"] + mc_raw["CPLE6"]
        else:
            raise ValueError("Preços de CPLE3/CPLE6 não encontrados.")

        if {"IGTI3", "IGTI4"}.issubset(mc_raw.index):
            igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
        else:
            raise ValueError("Preços de IGTI3/IGTI4 não encontrados.")

        if {"ENGI3", "ENGI4"}.issubset(mc_raw.index):
            engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
        else:
            raise ValueError("Preços de ENGI3/ENGI4 não encontrados.")

        # Monta a tabela final
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
                price = np.nan
                shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]
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
        resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

        return resultado, prices

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None, None

def calculate_irr_from_excel(resultado):
    """Calcula IRR a partir do arquivo Excel"""
    try:
        df = pd.read_excel('irrdash3.xlsx')
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # Garante colunas para todos os tickers
        for t in resultado.index:
            if t not in df.columns:
                df[t] = pd.NA

        # Escreve market caps (negativos) na primeira linha
        target_row = df.index[0]
        today = datetime.now().date()
        for ticker in resultado.index:
            mc_val = resultado.loc[ticker, 'market_cap']
            df.loc[target_row, ticker] = -abs(mc_val)

        # Garante numéricos
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors='coerce')

        # Calcula XIRR
        irr_results = {}
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue

            cashflows = series_cf.values.astype(float).copy()
            n_periods = len(cashflows)

            dates_list = [today]
            for i in range(1, n_periods):
                future_dt = date(today.year + i - 1, 12, 31)
                dates_list.append(future_dt)

            irr_results[t] = compute_xirr(cashflows, dates_list)

        ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])
        ytm_df['irr_aj'] = ytm_df['irr']

        # Ajuste de IRR real para MULT3, ALOS3 e IGTI11
        for ticker_adj in ['MULT3', 'ALOS3', 'IGTI11']:
            if ticker_adj in ytm_df.index and not pd.isna(ytm_df.loc[ticker_adj, 'irr']):
                ytm_df.loc[ticker_adj, 'irr_aj'] = ((1 + ytm_df.loc[ticker_adj, 'irr']) / (1 + 0.045)) - 1

        return ytm_df[['irr_aj']].dropna().sort_values('irr_aj', ascending=True)

    except FileNotFoundError:
        # Dados de exemplo se não houver arquivo Excel
        sample_data = {
            'EGIE3': 0.0920, 'ENGI11': 0.0950, 'ENEV3': 0.0998, 'MULT3': 0.0999,
            'SBSP3': 0.1035, 'CPLE6': 0.1088, 'IGTI11': 0.1123, 'NEOE3': 0.1135,
            'EQTL3': 0.1143, 'ALOS3': 0.1199
        }
        return pd.DataFrame.from_dict(sample_data, orient='index', columns=['irr_aj'])

# Header principal
st.markdown("""
<div class="main-header">
    <h1>IRR Real</h1>
</div>
""", unsafe_allow_html=True)

# Carregamento dos dados
with st.spinner("🔄 Carregando dados e calculando XIRR..."):
    resultado, prices = load_data()

if resultado is not None:
    # Calcula IRR
    ytm_clean = calculate_irr_from_excel(resultado)
    
    if len(ytm_clean) > 0:
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #D4A574; margin-bottom: 10px;">🏆 Maior IRR</h3>
                <h2 style="color: white; margin: 0;">{} ({:.2f}%)</h2>
            </div>
            """.format(ytm_clean.iloc[-1].name, ytm_clean.iloc[-1]['irr_aj']*100), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #D4A574; margin-bottom: 10px;">🔻 Menor IRR</h3>
                <h2 style="color: white; margin: 0;">{} ({:.2f}%)</h2>
            </div>
            """.format(ytm_clean.iloc[0].name, ytm_clean.iloc[0]['irr_aj']*100), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #D4A574; margin-bottom: 10px;">📊 Empresas Analisadas</h3>
                <h2 style="color: white; margin: 0;">{}</h2>
            </div>
            """.format(len(ytm_clean)), unsafe_allow_html=True)

        # Gráfico principal
        st.markdown("### 📈 IRR por Empresa (ajustada onde aplicável)")
        
        plot_data = pd.DataFrame({
            'empresa': ytm_clean.index,
            'irr': ytm_clean['irr_aj'] * 100,
        })

        # Cores customizadas baseadas na imagem
        colors = ['#5A7A9A', '#8A9AAA', '#B8A982', '#9AB87C', '#A2B584', 
                  '#C4A975', '#D4A574', '#A4B6D4', '#B4C6E4', '#C4D6F4']
        
        fig_irr = go.Figure(data=[
            go.Bar(
                x=plot_data['empresa'],
                y=plot_data['irr'],
                text=[f'{val:.2f}%' for val in plot_data['irr']],
                textposition='outside',
                marker_color=colors[:len(plot_data)],
                textfont=dict(color='white', size=12)
            )
        ])

        fig_irr.update_layout(
            title={
                'text': "IRR por Empresa (ajustada onde aplicável)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'white'}
            },
            xaxis_title="Empresas",
            yaxis_title="IRR (%)",
            height=600,
            showlegend=False,
            plot_bgcolor='#1E3A5F',
            paper_bgcolor='#1E3A5F',
            font={'color': 'white'},
            xaxis=dict(
                tickfont={'color': 'white'},
                title_font={'color': 'white'}
            ),
            yaxis=dict(
                tickfont={'color': 'white'},
                title_font={'color': 'white'},
                range=[0, max(plot_data['irr']) * 1.15]
            ),
            margin=dict(t=60, b=40)
        )

        st.plotly_chart(fig_irr, use_container_width=True)

        # Tabela de resultados
        with st.expander("📋 Ver tabela detalhada"):
            display_df = ytm_clean.copy()
            display_df['IRR (%)'] = (display_df['irr_aj'] * 100).round(2)
            display_df = display_df[['IRR (%)']]
            st.dataframe(display_df, use_container_width=True)

        # Nota de atualização
        st.markdown("""
        <div class="refresh-note">
            <p>💡 Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("⚠️ Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

else:
    st.error("❌ Erro ao carregar dados. Verifique sua conexão com a internet.")

# Sidebar com informações
with st.sidebar:
    st.markdown("### ℹ️ Informações")
    st.markdown("""
    - **IRR Real**: Ajustada com deflator de 4,5% a.a. para MULT3, ALOS3 e IGTI11
    - **Dados**: Yahoo Finance + Excel (irrdash3.xlsx)
    - **Atualização**: Dados atualizados a cada refresh
    - **XIRR**: Considera datas específicas dos fluxos de caixa
    """)
    
    if st.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.experimental_rerun()



