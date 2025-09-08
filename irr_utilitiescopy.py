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
    page_title="Calculadora de IRR - Empresas B3",
    layout="wide",
    initial_sidebar_state="expanded"
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
        <h1 class="title-text" style="color: white !important;">Calculadora de IRR - Empresas B3</h1>
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

# Função para converter market cap para primeiros dígitos em milhões
def cap_to_first_digits_mln(value, digits=6):
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])

# Sidebar para configurações
st.sidebar.header("Configurações")

# Lista de tickers
default_tickers = [
    "CPLE6", "EQTL3", "ENGI11", "SBSP3", "NEOE3",
    "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3", "IGTI11"
]

# Quantidades de ações por ticker
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

# Seleção de tickers na sidebar
selected_tickers = st.sidebar.multiselect(
    "Selecione os tickers:",
    default_tickers,
    default=default_tickers[:8]
)

# Upload do arquivo Excel
uploaded_file = st.sidebar.file_uploader(
    "Upload do arquivo Excel (irrdash3.xlsx)", 
    type=['xlsx', 'xls']
)

# Botão para executar análise
run_analysis = st.sidebar.button("🚀 Executar Análise", type="primary")

# Criar abas
tab1, tab2, tab3 = st.tabs(["📊 Market Cap", "💰 Análise IRR", "📈 Visualizações"])

if run_analysis and selected_tickers:
    
    with st.spinner("Baixando dados do Yahoo Finance..."):
        try:
            # Adiciona sufixo da B3 e baixa os dados
            tickers_sa = [f"{t}.SA" for t in selected_tickers]
            closes = yf.download(tickers_sa, period="5d")["Close"]
            
            # Pega o último preço de fechamento disponível
            if len(selected_tickers) == 1:
                last_close = pd.Series([closes.iloc[-1]], index=[selected_tickers[0]])
            else:
                last_close = closes.iloc[-1]
                last_close.index = [t.replace(".SA", "") for t in last_close.index]
            
            # Converte preços para float
            prices = last_close.astype(float)
            
            # Quantidade de ações por ticker
            shares = pd.Series({k: v for k, v in shares_dict.items() if k in selected_tickers}).reindex(prices.index)
            
            # Calcula market cap
            market_cap = prices * shares
            
            # Cria DataFrame resultado
            resultado = pd.DataFrame({
                "price": prices,
                "shares": shares,
                "market_cap": market_cap,
            })
            
            # Converte market cap para formato adequado
            resultado['market_cap_formatted'] = resultado['market_cap'].apply(cap_to_first_digits_mln)
            
            # Exibe na aba Market Cap
            with tab1:
                st.subheader("📊 Market Cap das Empresas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dados Atuais:**")
                    display_df = resultado.copy()
                    display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"R$ {x:,.0f}" if not pd.isna(x) else "N/A")
                    display_df['price'] = display_df['price'].apply(lambda x: f"R$ {x:.2f}" if not pd.isna(x) else "N/A")
                    display_df['shares'] = display_df['shares'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
                    st.dataframe(display_df, use_container_width=True)
                
                with col2:
                    # Gráfico de market cap
                    fig_mc = px.bar(
                        x=resultado.index,
                        y=resultado['market_cap']/1e9,  # Em bilhões
                        title="Market Cap por Empresa (R$ Bilhões)",
                        color_discrete_sequence=[STK_DOURADO]
                    )
                    fig_mc.update_layout(
                        plot_bgcolor=STK_AZUL,
                        paper_bgcolor=STK_AZUL,
                        font_color='white',
                        xaxis_title="Empresas",
                        yaxis_title="Market Cap (R$ Bilhões)"
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
            
            # Processa arquivo Excel se fornecido
            if uploaded_file is not None:
                try:
                    # Lê o arquivo Excel
                    df = pd.read_excel(uploaded_file)
                    
                    # Move a linha 0 como título
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    
                    # Garante que as colunas dos tickers existam
                    for t in resultado.index:
                        if t not in df.columns:
                            df[t] = pd.NA
                    
                    # Escreve os valores na primeira linha usando os market caps
                    target_row = df.index[0]
                    df.loc[target_row, list(resultado.index)] = list(resultado['market_cap_formatted'].values)
                    
                    # Multiplica por -1 a primeira linha (saída de caixa)
                    df.iloc[0] = df.iloc[0] * -1
                    
                    # Garante que as colunas sejam numéricas
                    for t in resultado.index:
                        df[t] = pd.to_numeric(df[t], errors='coerce')
                    
                    # Calcula IRR para cada ticker
                    irr_results = {}
                    cashflow_details = {}
                    
                    for t in resultado.index:
                        series_cf = df[t].dropna()
                        if series_cf.empty:
                            irr_results[t] = np.nan
                            cashflow_details[t] = []
                        else:
                            irr_results[t] = compute_irr(series_cf.values)
                            cashflow_details[t] = series_cf.values.tolist()
                    
                    # Cria DataFrame com resultados
                    ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])
                    ytm_df = ytm_df.sort_values('irr', ascending=False)
                    
                    # Exibe na aba IRR
                    with tab2:
                        st.subheader("💰 Análise de IRR")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Resultados de IRR:**")
                            display_irr = ytm_df.copy()
                            display_irr['irr_percent'] = display_irr['irr'].apply(
                                lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
                            )
                            st.dataframe(display_irr[['irr_percent']], use_container_width=True)
                            
                            # Estatísticas
                            valid_irrs = ytm_df['irr'].dropna()
                            if len(valid_irrs) > 0:
                                st.write("**Estatísticas:**")
                                st.write(f"- Média: {valid_irrs.mean()*100:.2f}%")
                                st.write(f"- Mediana: {valid_irrs.median()*100:.2f}%")
                                st.write(f"- Desvio Padrão: {valid_irrs.std()*100:.2f}%")
                        
                        with col2:
                            # Seletor para exibir fluxo de caixa
                            selected_company = st.selectbox(
                                "Selecione empresa para ver fluxo de caixa:",
                                options=list(cashflow_details.keys())
                            )
                            
                            if selected_company and cashflow_details[selected_company]:
                                cf_data = cashflow_details[selected_company]
                                periods = list(range(len(cf_data)))
                                
                                fig_cf = go.Figure(data=go.Bar(
                                    x=[f"Período {i}" for i in periods],
                                    y=cf_data,
                                    marker_color=[STK_DOURADO if x >= 0 else 'red' for x in cf_data]
                                ))
                                fig_cf.update_layout(
                                    title=f"Fluxo de Caixa - {selected_company}",
                                    plot_bgcolor=STK_AZUL,
                                    paper_bgcolor=STK_AZUL,
                                    font_color='white'
                                )
                                st.plotly_chart(fig_cf, use_container_width=True)
                    
                    # Exibe visualizações na aba 3
                    with tab3:
                        st.subheader("📈 Visualizações Comparativas")
                        
                        # Gráfico de IRR
                        valid_data = ytm_df.dropna()
                        if len(valid_data) > 0:
                            fig_irr = px.bar(
                                x=valid_data.index,
                                y=valid_data['irr'] * 100,
                                title="IRR por Empresa (%)",
                                color_discrete_sequence=[STK_DOURADO]
                            )
                            fig_irr.update_layout(
                                plot_bgcolor=STK_AZUL,
                                paper_bgcolor=STK_AZUL,
                                font_color='white',
                                xaxis_title="Empresas",
                                yaxis_title="IRR (%)"
                            )
                            st.plotly_chart(fig_irr, use_container_width=True)
                            
                            # Gráfico de dispersão IRR vs Market Cap
                            scatter_data = pd.merge(
                                resultado[['market_cap']], 
                                ytm_df[['irr']], 
                                left_index=True, 
                                right_index=True
                            ).dropna()
                            
                            if len(scatter_data) > 0:
                                fig_scatter = px.scatter(
                                    scatter_data,
                                    x='market_cap',
                                    y='irr',
                                    title="IRR vs Market Cap",
                                    labels={'market_cap': 'Market Cap (R$)', 'irr': 'IRR'},
                                    color_discrete_sequence=[STK_DOURADO]
                                )
                                fig_scatter.update_layout(
                                    plot_bgcolor=STK_AZUL,
                                    paper_bgcolor=STK_AZUL,
                                    font_color='white'
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Erro ao processar arquivo Excel: {str(e)}")
            else:
                with tab2:
                    st.info("📁 Faça upload do arquivo Excel para calcular o IRR")
                with tab3:
                    st.info("📁 Faça upload do arquivo Excel para ver as visualizações")
                    
        except Exception as e:
            st.error(f"Erro ao baixar dados: {str(e)}")

else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Executar Análise'")

# Rodapé
st.markdown("---")
st.markdown(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


