import streamlit as st
import pandas as pd
import plotly.express as px

# Exemplo de dados simulados baseados no ytm_clean
# Substitua por seus dados reais
ytm_clean = pd.DataFrame({
    'irr_aj': [0.0524, 0.0687, 0.0789, 0.0856, 0.0923, 0.1045, 0.1123, 0.1234, 0.1345, 0.1456, 0.1567]
}, index=['ALOS3', 'MULT3', 'EGIE3', 'ELET3', 'ENEV3', 'NEOE3', 'SBSP3', 'EQTL3', 'CPLE6', 'IGTI11', 'ENGI11'])

# Criar o gráfico
plot_data = pd.DataFrame({
    'empresa': ytm_clean.index,
    'irr': ytm_clean['irr_aj'] * 100,  # Convertendo para percentual
})

fig_irr = px.bar(
    plot_data,
    x='empresa',
    y='irr',
    title="IRR por Empresa (ajustada onde aplicável)",
    text='irr',
    color='irr',
    color_continuous_scale='RdYlGn'
)

fig_irr.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

fig_irr.update_layout(
    xaxis_title="Empresas",
    yaxis_title="IRR (%)",
    height=600,
    showlegend=False,
    xaxis_tickangle=-45
)

# Exibir no Streamlit
st.plotly_chart(fig_irr, use_container_width=True)
