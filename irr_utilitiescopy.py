import pandas as pd
import numpy as np
import yfinance as yf
import numpy_financial as npf
import streamlit as st
import plotly.express as px

# Configurar página do Streamlit
st.set_page_config(
    page_title="Análise de IRR - Setor Elétrico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Definir as cores da STK
STK_AZUL = "#102E46"
STK_DOURADO = "#C98C2E"
STK_BRANCO = "#FFFFFF"

# Estilo CSS personalizado com fundo azul escuro
st.markdown("""
    <style>
        /* Fundo principal */
        .stApp {
            background-color: #102E46;
        }
        
        /* Container do título personalizado */
        .title-container {
            background-color: #102E46;
            padding: 2rem;
            margin: 2rem 0;
            text-align: center;
        }
        
        /* Título principal */
        .title-text {
            color: #FFFFFF;
            font-size: 3.5em;
            font-weight: bold;
            margin: 0;
            padding: 0;
            font-family: sans-serif;
        }
        
        /* Container do título */
        div[data-testid="stHeader"] {
            background-color: #102E46;
        }
        
        /* Customização de textos */
        .st-emotion-cache-1y4p8pa {
            color: #FFFFFF;
        }
        
        /* Remover marca d'água do Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Estilo para o container do gráfico */
        .plot-container {
            background-color: #102E46;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }

        /* Centralizar o conteúdo */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        
        /* Esconder o título padrão do Streamlit */
        .st-emotion-cache-1629p8f {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Título personalizado com HTML/CSS
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">Análise de IRR - Setor Elétrico</h1>
    </div>
""", unsafe_allow_html=True)

# Ler o arquivo Excel definindo a linha 1 como cabeçalho
irr_dash = pd.read_excel('irrdash3.xlsx', header=1)

# Remover a coluna 'Unnamed: 0' se ela existir
if 'Unnamed: 0' in irr_dash.columns:
    irr_dash = irr_dash.drop('Unnamed: 0', axis=1)

# Definir o ticker da empresa
ticker_symbol = "CPLE6.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_cple6 = market_cap / 1e6*-1

# EQTL3
ticker_symbol = "EQTL3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_eqtl3 = market_cap / 1e6*-1

# ENGI11
ticker_symbol = "ENGI11.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_engi11 = market_cap / 1e6*-1

# SBSP3
ticker_symbol = "SBSP3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_sbsp3 = market_cap / 1e6*-1

# NEOE3
ticker_symbol = "NEOE3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_neoe3 = market_cap / 1e6*-1

# ENEV3
ticker_symbol = "ENEV3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_enev3 = market_cap / 1e6*-1

# ELET3
ticker_symbol = "ELET3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_elet3 = market_cap / 1e6*-1

# EGIE3
ticker_symbol = "EGIE3.SA"
ticker = yf.Ticker(ticker_symbol)
market_cap = ticker.info.get("marketCap")
market_cap_egie3 = market_cap / 1e6*-1

# Transformar irr_dash em dataframe
irr_dash = pd.DataFrame(irr_dash)

# Preenche a linha que está com NaN com o valor da variavel market_cap
irr_dash.at[0, "CPLE6"] = market_cap_cple6
irr_dash.at[0, "EQTL3"] = market_cap_eqtl3
irr_dash.at[0, "ENGI11"] = market_cap_engi11
irr_dash.at[0, "SBSP3"] = market_cap_sbsp3
irr_dash.at[0, "NEOE3"] = market_cap_neoe3
irr_dash.at[0, "ENEV3"] = market_cap_enev3
irr_dash.at[0, "ELET3"] = market_cap_elet3
irr_dash.at[0, "EGIE3"] = market_cap_egie3

# Data atual
data_atual = pd.Timestamp.now().strftime('%d/%m/%Y')
irr_dash.at[0, "Year"] = data_atual

# Calcular IRR para todas as empresas
irr_dict = {}
for empresa in ["CPLE6", "EQTL3", "ENGI11", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3"]:
    cashflows = irr_dash[empresa].dropna().astype(float).tolist()
    irr_value = npf.irr(cashflows)
    irr_dict[empresa] = irr_value

# Criar DataFrame com os valores de IRR
irr_df = pd.DataFrame(list(irr_dict.items()), columns=['Empresa', 'IRR'])
irr_df_plot = irr_df.copy()
irr_df_plot['IRR'] = irr_df_plot['IRR'] * 100

# Criar gráfico de barras com Plotly
fig = px.bar(
    irr_df_plot,
    x='Empresa',
    y='IRR',
    color_discrete_sequence=[STK_DOURADO]
)

# Adicionar os valores em cima das barras
fig.update_traces(
    text=irr_df_plot['IRR'].apply(lambda x: f'{x:.2f}%'),
    textposition='outside',
    textfont=dict(color=STK_DOURADO)
)

# Personalizar o layout do gráfico
fig.update_layout(
    plot_bgcolor=STK_AZUL,
    paper_bgcolor=STK_AZUL,
    font_color=STK_DOURADO,
    showlegend=False,
    height=700,
    margin=dict(t=50, b=50, l=50, r=50),
    xaxis=dict(
        title="Empresas",
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        tickfont=dict(color=STK_DOURADO, size=14),
        title_font=dict(color=STK_DOURADO, size=16)
    ),
    yaxis=dict(
        title="IRR (%)",
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        tickfont=dict(color=STK_DOURADO, size=14),
        tickformat='.2f',
        title_font=dict(color=STK_DOURADO, size=16)
    )
)

# Exibir o gráfico no Streamlit com margens personalizadas
st.plotly_chart(fig, use_container_width=True)
