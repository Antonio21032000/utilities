import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Tenta importar numpy_financial
try:
    import numpy_financial as npf
except Exception:
    npf = None

# Configurar página do Streamlit
st.set_page_config(
    page_title="Análise de IRR - Empresas B3",
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
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #102E46;
        }

        .stTabs [data-baseweb="tab"] {
            color: #C98C2E;
            background-color: rgba(201, 140, 46, 0.1);
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(201, 140, 46, 0.2);
            color: white;
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
    </style>
""", unsafe_allow_html=True)

# Título personalizado
st.markdown("""
    <div class="title-container">
        <h1 class="title-text" style="color: white !important;">Análise de IRR - Empresas B3</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Função para calcular IRR
def compute_irr(cashflows: np.ndarray) -> float:
    values = np.asarray(cashflows, dtype=float)
    
    # Tenta numpy_financial primeiro
    if npf is not None:
        try:
            irr_val = float(npf.irr(values))
            return irr_val
        except Exception:
            pass

    # Fallback: método de bisseção
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

# Função para converter market cap
def cap_to_first_digits_mln(value, digits=6):
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])

# Executar o código principal
try:
    with st.spinner("Carregando dados e calculando IRR..."):
        
        # Buscar preços de fechamento para múltiplos tickers da B3
        tickers = [
            "CPLE6", "EQTL3", "ENGI11", "SBSP3", "NEOE3",
            "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3",
            "IGTI11"
        ]

        # Adiciona sufixo da B3 e baixa os dados
        tickers_sa = [f"{t}.SA" for t in tickers]
        closes = yf.download(tickers_sa, period="5d")["Close"]

        # Pega o último preço de fechamento disponível
        last_close = closes.iloc[-1]

        # Renomeia os índices para remover o sufixo ".SA"
        last_close.index = [t.replace(".SA", "") for t in last_close.index]

        # Converte preços para float (Series)
        prices = last_close.astype(float)

        # Quantidade de ações por ticker (inteiros)
        shares_dict = {
            "CPLE6": 2982810000,
            "EQTL3": 1255510000,
            "ENGI11": 457130458,
            "SBSP3": 683510000,
            "NEOE3": 1213800000,
            "ENEV3": 1936970000,
            "ELET3": 2308630000,
            "EGIE3": 815928000,
            "MULT3": 513164000,
            "ALOS3": 542937000,
            "IGTI11": 296728385,
        }

        shares = pd.Series(shares_dict).reindex(prices.index)

        # Calcula market cap (preço x ações)
        market_cap = prices * shares

        resultado = pd.DataFrame({
            "price": prices,
            "shares": shares,
            "market_cap": market_cap,
        })

        # Reduz market_cap para milhões e mantém apenas os 6 primeiros dígitos
        resultado['market_cap'] = resultado['market_cap'].apply(cap_to_first_digits_mln)

        # PEGA O dataframe do excel irrdash3.xlsx
        df = pd.read_excel('irrdash3.xlsx')

        # vamos subir a linha 0 como titulo
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # garante que as colunas dos tickers existam
        for t in resultado.index:
            if t not in df.columns:
                df[t] = pd.NA

        # escreve os valores na primeira linha de dados usando rótulos
        target_row = df.index[0]
        df.loc[target_row, list(resultado.index)] = list(resultado['market_cap'].values)

        # multiplica por -1 a linha do 1 do df
        df.iloc[0] = df.iloc[0] * -1

        # garante numéricos nas colunas de tickers
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors='coerce')

        irr_results = {}
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue
            # espera-se que a primeira linha seja saída (negativa)
            irr_results[t] = compute_irr(series_cf.values)

        ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])

    # Criar tabs
    tab1, tab2 = st.tabs(["📊 Market Cap & Dados", "💰 Análise IRR"])

    with tab1:
        st.subheader("📊 Dados das Empresas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Cap e Preços:**")
            display_df = resultado.copy()
            display_df['market_cap_original'] = display_df['market_cap'] * 1e6  # Volta para valor original
            display_df['market_cap_formatted'] = display_df['market_cap_original'].apply(lambda x: f"R$ {x:,.0f}" if not pd.isna(x) else "N/A")
            display_df['price_formatted'] = display_df['price'].apply(lambda x: f"R$ {x:.2f}" if not pd.isna(x) else "N/A")
            display_df['shares_formatted'] = display_df['shares'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            
            st.dataframe(display_df[['price_formatted', 'shares_formatted', 'market_cap_formatted']], use_container_width=True)
        
        with col2:
            # Gráfico de market cap
            valid_mc = resultado.dropna(subset=['market_cap'])
            if len(valid_mc) > 0:
                fig_mc = px.bar(
                    x=valid_mc.index,
                    y=valid_mc['market_cap'],
                    title="Market Cap por Empresa (Milhões R$)",
                    color_discrete_sequence=[STK_DOURADO]
                )
                fig_mc.update_layout(
                    plot_bgcolor=STK_AZUL,
                    paper_bgcolor=STK_AZUL,
                    font_color='white',
                    xaxis_title="Empresas",
                    yaxis_title="Market Cap (Milhões R$)"
                )
                st.plotly_chart(fig_mc, use_container_width=True)

    with tab2:
        st.subheader("💰 Análise de IRR (Yield to Maturity)")
        
        # Limpar dados inválidos e ordenar
        ytm_clean = ytm_df.dropna().sort_values('irr', ascending=True)
        
        if len(ytm_clean) > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Resultados de IRR:**")
                display_irr = ytm_clean.copy()
                display_irr['irr_percent'] = display_irr['irr'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(display_irr[['irr_percent']], use_container_width=True)
                
                # Estatísticas
                st.write("**Estatísticas:**")
                st.metric("Média", f"{ytm_clean['irr'].mean()*100:.2f}%")
                st.metric("Mediana", f"{ytm_clean['irr'].median()*100:.2f}%")
                st.metric("Desvio Padrão", f"{ytm_clean['irr'].std()*100:.2f}%")
                st.metric("Melhor IRR", f"{ytm_clean['irr'].max()*100:.2f}%")
                st.metric("Pior IRR", f"{ytm_clean['irr'].min()*100:.2f}%")
            
            with col2:
                # Gráfico principal do ytm_df
                fig_irr = px.bar(
                    x=ytm_clean.index,
                    y=ytm_clean['irr'] * 100,
                    title="IRR (Yield to Maturity) por Empresa",
                    color_discrete_sequence=[STK_DOURADO],
                    text=ytm_clean['irr'].apply(lambda x: f"{x*100:.2f}%")
                )
                
                fig_irr.update_traces(
                    textposition='outside',
                    textfont=dict(color='white', size=12)
                )
                
                fig_irr.update_layout(
                    plot_bgcolor=STK_AZUL,
                    paper_bgcolor=STK_AZUL,
                    font_color='white',
                    xaxis_title="Empresas",
                    yaxis_title="IRR (%)",
                    height=500,
                    showlegend=False,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                    )
                )
                
                st.plotly_chart(fig_irr, use_container_width=True)
            
            # Gráfico adicional: Distribuição de IRR
            st.subheader("📈 Distribuição de IRR")
            fig_hist = px.histogram(
                x=ytm_clean['irr'] * 100,
                nbins=10,
                title="Distribuição dos Valores de IRR",
                color_discrete_sequence=[STK_DOURADO]
            )
            fig_hist.update_layout(
                plot_bgcolor=STK_AZUL,
                paper_bgcolor=STK_AZUL,
                font_color='white',
                xaxis_title="IRR (%)",
                yaxis_title="Frequência"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        else:
            st.error("Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

except FileNotFoundError:
    st.error("❌ Arquivo 'irrdash3.xlsx' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
except Exception as e:
    st.error(f"❌ Erro durante a execução: {str(e)}")

# Rodapé
st.markdown("---")
st.markdown(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
