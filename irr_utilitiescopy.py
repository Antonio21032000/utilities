import pandas as pd
import numpy as np
import yfinance as yf
import numpy_financial as npf
import streamlit as st
import plotly.express as px

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

# Criar versão não formatada do DataFrame para o gráfico
irr_df_plot = irr_df.copy()
irr_df_plot['IRR'] = irr_df_plot['IRR'] * 100  # Converter para percentagem

# Formatar a coluna IRR como percentagem para exibição na tabela
irr_df['IRR'] = irr_df['IRR'].apply(lambda x: f"{x:.2%}")

# Configurar página do Streamlit
st.set_page_config(
    page_title="Análise de IRR - Setor Elétrico",
    layout="wide"
)

# Definir as cores da STK
STK_AZUL = "#102E46"
STK_DOURADO = "#C98C2E"

# Estilo CSS personalizado
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .stHeader {
        background-color: #102E46;
    }
    .st-emotion-cache-1629p8f h1 {
        color: #102E46;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("Análise de IRR - Setor Elétrico")

# Criar gráfico de barras com Plotly usando o DataFrame não formatado
fig = px.bar(
    irr_df_plot,
    x='Empresa',
    y='IRR',
    title='IRR por Empresa',
    color_discrete_sequence=[STK_AZUL]
)

# Personalizar o layout do gráfico
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color=STK_AZUL,
    title_font_color=STK_AZUL,
    title_font_size=24,
    showlegend=False,
    height=600
)

# Personalizar os eixos
fig.update_xaxes(
    title_text='Empresas',
    title_font_color=STK_AZUL,
    tickfont_color=STK_AZUL,
    gridcolor='lightgray'
)

fig.update_yaxes(
    title_text='IRR (%)',
    title_font_color=STK_AZUL,
    tickfont_color=STK_AZUL,
    gridcolor='lightgray',
    tickformat='.2f'  # Mostrar 2 casas decimais
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Exibir o DataFrame abaixo do gráfico
st.subheader("Dados Detalhados")
st.dataframe(
    irr_df.style.set_properties(**{
        'background-color': 'white',
        'color': STK_AZUL
    })
)
