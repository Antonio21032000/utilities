import pandas as pd
import numpy as np
import yfinance as yf
import numpy_financial as npf
import streamlit as st

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

# faz a mesma coisa para EQTL3, ENGI11, SBSP3, NEOE3, ENEV3, ELET3, EGIE3

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


#transformar irr_dash em dataframe
irr_dash = pd.DataFrame(irr_dash)

# preenche a linha que está com NaN com o valor da variavel market_ca

irr_dash.at[0, "CPLE6"] = market_cap_cple6
irr_dash.at[0, "EQTL3"] = market_cap_eqtl3
irr_dash.at[0, "ENGI11"] = market_cap_engi11
irr_dash.at[0, "SBSP3"] = market_cap_sbsp3
irr_dash.at[0, "NEOE3"] = market_cap_neoe3
irr_dash.at[0, "ENEV3"] = market_cap_enev3
irr_dash.at[0, "ELET3"] = market_cap_elet3
irr_dash.at[0, "EGIE3"] = market_cap_egie3

# data atual
data_atual = pd.Timestamp.now().strftime('%d/%m/%Y')

irr_dash.at[0, "Year"] = data_atual

# calcula a TIR na coluna CPLE6

# Filtrar os fluxos de caixa da coluna 'CPLE6' (removendo possíveis NaN)
cashflows_cple6 = irr_dash["CPLE6"].dropna().astype(float).tolist()
# Calcular a TIR (IRR) usando numpy_financial
irr_value_cple6 = npf.irr(cashflows_cple6)

# EQTL3
cashflows_eqtl3 = irr_dash["EQTL3"].dropna().astype(float).tolist()
irr_value_eqtl3 = npf.irr(cashflows_eqtl3)

# ENGI11
cashflows_engi11 = irr_dash["ENGI11"].dropna().astype(float).tolist()
irr_value_engi11 = npf.irr(cashflows_engi11)

# SBSP3
cashflows_sbsp3 = irr_dash["SBSP3"].dropna().astype(float).tolist()
irr_value_sbsp3 = npf.irr(cashflows_sbsp3)

# NEOE3
cashflows_neoe3 = irr_dash["NEOE3"].dropna().astype(float).tolist()
irr_value_neoe3 = npf.irr(cashflows_neoe3)

# ENEV3
cashflows_enev3 = irr_dash["ENEV3"].dropna().astype(float).tolist()
irr_value_enev3 = npf.irr(cashflows_enev3)

# ELET3
cashflows_elet3 = irr_dash["ELET3"].dropna().astype(float).tolist()
irr_value_elet3 = npf.irr(cashflows_elet3)

# EGIE3
cashflows_egie3 = irr_dash["EGIE3"].dropna().astype(float).tolist()
irr_value_egie3 = npf.irr(cashflows_egie3)

# Imprimir os resultados
print(f"TIR CPLE6: {irr_value_cple6:.2%}")
print(f"TIR EQTL3: {irr_value_eqtl3:.2%}")
print(f"TIR ENGI11: {irr_value_engi11:.2%}")
print(f"TIR SBSP3: {irr_value_sbsp3:.2%}")
print(f"TIR NEOE3: {irr_value_neoe3:.2%}")
print(f"TIR ENEV3: {irr_value_enev3:.2%}")
print(f"TIR ELET3: {irr_value_elet3:.2%}")
print(f"TIR EGIE3: {irr_value_egie3:.2%}")

# Criar dicionário com os valores de IRR
irr_dict = {
    'CPLE6': irr_value_cple6,
    'EQTL3': irr_value_eqtl3,
    'ENGI11': irr_value_engi11,
    'SBSP3': irr_value_sbsp3,
    'NEOE3': irr_value_neoe3,
    'ENEV3': irr_value_enev3,
    'ELET3': irr_value_elet3,
    'EGIE3': irr_value_egie3
}

# Criar DataFrame com os valores de IRR
irr_df = pd.DataFrame(list(irr_dict.items()), columns=['Empresa', 'IRR'])

# Formatar a coluna IRR como percentagem
irr_df['IRR'] = irr_df['IRR'].apply(lambda x: f"{x:.2%}")

# Ordenar o DataFrame pelo valor do IRR (opcional)
# irr_df = irr_df.sort_values('IRR', ascending=False)

# Exibir o DataFrame
print("\nDataFrame com valores de IRR:")
print(irr_df)


# streamlit



import streamlit as st
import plotly.express as px
import pandas as pd

# Configurar página do Streamlit com as cores da STK
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

# Criar gráfico de barras com Plotly
fig = px.bar(
    irr_df,
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
    tickformat='.2%'
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
