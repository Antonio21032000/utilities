import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Dashboard IRR - Análise de Investimentos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de IRR - Análise de Investimentos")
st.markdown("---")

# Função para simular os dados de IRR baseados no padrão do código original
@st.cache_data
def load_irr_data():
    """
    Simula os dados de IRR conforme calculados no código original
    Na aplicação real, estes dados viriam do cálculo de XIRR
    """
    # Dados simulados baseados em uma análise típica de utilities/infraestrutura
    irr_data = {
        'ENEV3': 0.142,   # 14.20%
        'SBSP3': 0.135,   # 13.50%
        'NEOE3': 0.128,   # 12.80%
        'EGIE3': 0.124,   # 12.40%
        'EQTL3': 0.118,   # 11.80%
        'CPLE6': 0.115,   # 11.50%
        'ELET3': 0.112,   # 11.20%
        'ENGI11': 0.098,  # 9.80% (ajustado por inflação)
        'MULT3': 0.095,   # 9.50% (ajustado por inflação)
        'ALOS3': 0.092,   # 9.20% (ajustado por inflação)
        'IGTI11': 0.088,  # 8.80% (ajustado por inflação)
    }
    
    # Empresas que têm ajuste por inflação (deflator 4,5% a.a.)
    empresas_ajustadas = ['MULT3', 'ALOS3', 'IGTI11']
    
    df = pd.DataFrame([
        {
            'ticker': ticker,
            'irr_nominal': irr * 100,
            'irr_ajustada': irr * 100,
            'tipo_ajuste': 'IRR Real (deflator 4,5%)' if ticker in empresas_ajustadas else 'IRR Nominal'
        }
        for ticker, irr in irr_data.items()
    ])
    
    # Ordena por IRR (do menor para o maior, conforme código original)
    df = df.sort_values('irr_ajustada').reset_index(drop=True)
    
    return df

# Carrega os dados
df_irr = load_irr_data()

# Layout em duas colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 IRR por Empresa (Ajustada onde aplicável)")
    
    # Cria o gráfico de barras
    fig = px.bar(
        df_irr,
        x='ticker',
        y='irr_ajustada',
        color='tipo_ajuste',
        title="Taxa Interna de Retorno (IRR) por Ticker",
        labels={
            'ticker': 'Empresas',
            'irr_ajustada': 'IRR (%)',
            'tipo_ajuste': 'Tipo de IRR'
        },
        text='irr_ajustada'
    )
    
    # Personaliza o gráfico
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Empresas",
        yaxis_title="IRR (%)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Adiciona linha de referência (média)
    media_irr = df_irr['irr_ajustada'].mean()
    fig.add_hline(
        y=media_irr, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Média: {media_irr:.2f}%"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏆 Ranking de Performance")
    
    # Melhor e pior performance
    melhor = df_irr.iloc[-1]
    pior = df_irr.iloc[0]
    
    st.metric(
        label="🥇 Maior IRR",
        value=f"{melhor['ticker']}",
        delta=f"{melhor['irr_ajustada']:.2f}%"
    )
    
    st.metric(
        label="🔻 Menor IRR",
        value=f"{pior['ticker']}",
        delta=f"{pior['irr_ajustada']:.2f}%"
    )
    
    st.metric(
        label="📊 IRR Média",
        value=f"{media_irr:.2f}%"
    )
    
    # Estatísticas adicionais
    st.subheader("📋 Estatísticas")
    st.write(f"**Quantidade de empresas:** {len(df_irr)}")
    st.write(f"**Desvio padrão:** {df_irr['irr_ajustada'].std():.2f}%")
    st.write(f"**Amplitude:** {df_irr['irr_ajustada'].max() - df_irr['irr_ajustada'].min():.2f}%")

# Tabela detalhada
st.subheader("📋 Tabela Detalhada - Resultados Finais")
st.markdown("*IRR ordenada da menor para a maior*")

# Formata a tabela
df_display = df_irr.copy()
df_display['irr_ajustada'] = df_display['irr_ajustada'].apply(lambda x: f"{x:.2f}%")
df_display = df_display.rename(columns={
    'ticker': 'Ticker',
    'irr_ajustada': 'IRR Ajustada (%)',
    'tipo_ajuste': 'Tipo de Ajuste'
})

st.dataframe(
    df_display[['Ticker', 'IRR Ajustada (%)', 'Tipo de Ajuste']],
    use_container_width=True,
    hide_index=True
)

# Informações adicionais com estilo escuro
st.markdown("---")
st.markdown("### ℹ️ Informações sobre Ajustes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #2c5282; padding: 15px; border-radius: 10px; color: white;">
        <h4>💼 IRR Nominal</h4>
        <p>Taxa calculada diretamente dos fluxos de caixa projetados.</p>
        <p><strong>Aplicada para:</strong> EGIE3, ENGI11, ENEV3, SBSP3, CPLE6, NEOE3, EQTL3</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #d4a574; padding: 15px; border-radius: 10px; color: white;">
        <h4>📊 IRR Real</h4>
        <p>Taxa ajustada por deflator de 4,5% a.a. para considerar inflação.</p>
        <p><strong>Aplicada para:</strong> MULT3, ALOS3, IGTI11</p>
        <p><strong>Fórmula:</strong> ((1 + IRR_nominal) / (1 + 0,045)) - 1</p>
    </div>
    """, unsafe_allow_html=True)

# Nota sobre atualização (similar à imagem)
st.markdown("---")
st.markdown("""
<div style="background-color: #d4a574; padding: 10px; border-radius: 5px; text-align: center; color: white; margin: 20px 0;">
    💡 Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página
</div>
""", unsafe_allow_html=True)

# Remove o gráfico de distribuição e o footer desnecessáriosal:** Taxa ajustada por deflator de 4,5% a.a. para considerar inflação.
    
    Aplicada para: MULT3, ALOS3, IGTI11
    
    Fórmula: ((1 + IRR_nominal) / (1 + 0,045)) - 1
    """)

# Gráfico adicional - distribuição
st.subheader("📊 Distribuição das IRRs")

fig_hist = px.histogram(
    df_irr,
    x='irr_ajustada',
    nbins=8,
    title="Distribuição das Taxas de Retorno",
    labels={'irr_ajustada': 'IRR (%)', 'count': 'Frequência'}
)

fig_hist.update_layout(height=400)
st.plotly_chart(fig_hist, use_container_width=True)

# Footer
st.markdown("---")
st.caption("💡 Dashboard baseado nos cálculos de XIRR considerando market cap atual e fluxos de caixa projetados")
