import streamlit as st
import pandas as pd
import plotly.express as px

# Assumindo que ytm_clean já foi calculado pelo código principal
# ytm_clean vem do processo de cálculo do IRR/XIRR com dados reais

# Criar o gráfico usando os dados reais do ytm_clean
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

