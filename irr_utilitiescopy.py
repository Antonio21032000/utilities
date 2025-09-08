import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Tenta importar numpy_financial
try:
    import numpy_financial as npf
except Exception:
    npf = None

# Configurar página do Streamlit
st.set_page_config(
    page_title="IRR Real",
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
        <h1 class="title-text" style="color: white !important;">IRR Real</h1>
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

# Função para calcular XIRR (como Excel) com datas específicas
def compute_xirr(cashflows: np.ndarray, dates: list, guess: float = 0.1) -> float:
    """
    Calcula XIRR considerando datas específicas para cada fluxo de caixa
    Similar à função XIRR do Excel
    """
    if len(cashflows) != len(dates):
        return np.nan
        
    values = np.asarray(cashflows, dtype=float)
    dates = pd.to_datetime(dates)
    
    # Data base (primeira data)
    base_date = dates[0]
    
    # Converter datas para anos decimais desde a data base
    years = [(d - base_date).days / 365.25 for d in dates]
    
    def xnpv(rate):
        """Calcula NPV considerando datas específicas"""
        return sum(cf / (1 + rate) ** year for cf, year in zip(values, years))
    
    # Método de Newton-Raphson para encontrar a taxa
    rate = guess
    for _ in range(100):
        npv = xnpv(rate)
        if abs(npv) < 1e-6:
            return rate
            
        # Derivada numérica
        delta = 1e-6
        d_npv = (xnpv(rate + delta) - xnpv(rate - delta)) / (2 * delta)
        
        if abs(d_npv) < 1e-10:
            return np.nan
            
        rate = rate - npv / d_npv
        
        # Verificar convergência
        if rate < -0.99 or rate > 100:
            return np.nan
            
    return rate

# Função para converter market cap
def cap_to_first_digits_mln(value, digits=6):
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])

# Executar o código principal
try:
    with st.spinner("🔄 Carregando dados e calculando IRR..."):
        
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
        
        # Data atual para o investimento inicial (market cap)
        today = datetime.now().date()
        
        # Market cap como saída de caixa na data atual
        for ticker in resultado.index:
            if ticker in df.columns:
                market_cap_value = resultado.loc[ticker, 'market_cap']
                # Aplica como saída de caixa (negativo) na data atual
                df.loc[target_row, ticker] = -abs(market_cap_value)

        # garante numéricos nas colunas de tickers
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors='coerce')

        irr_results = {}
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue
                
            # Criar datas correspondentes aos fluxos de caixa
            cashflows = series_cf.values
            n_periods = len(cashflows)
            
            # Data atual para investimento inicial, depois anos subsequentes
            dates = [today]
            for i in range(1, n_periods):
                future_date = today.replace(year=today.year + i)
                dates.append(future_date)
            
            # Calcular XIRR com datas específicas
            irr_results[t] = compute_xirr(cashflows, dates)

        ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])

        # Cria uma nova coluna no ytm_df com os mesmos valores da coluna irr
        ytm_df['irr_aj'] = ytm_df['irr']

        # Agora nessa irr_aj coluna, faz a conta em IGTI11, MULT3, ALOS3 considerando (1+irr) / (1+4,5%) - 1 para pegar a irr real deles
        if 'IGTI11' in ytm_df.index and not pd.isna(ytm_df.loc['IGTI11', 'irr']):
            ytm_df.loc['IGTI11', 'irr_aj'] = ((1 + ytm_df.loc['IGTI11', 'irr']) / (1 + 0.045)) - 1
        
        if 'MULT3' in ytm_df.index and not pd.isna(ytm_df.loc['MULT3', 'irr']):
            ytm_df.loc['MULT3', 'irr_aj'] = ((1 + ytm_df.loc['MULT3', 'irr']) / (1 + 0.045)) - 1
        
        if 'ALOS3' in ytm_df.index and not pd.isna(ytm_df.loc['ALOS3', 'irr']):
            ytm_df.loc['ALOS3', 'irr_aj'] = ((1 + ytm_df.loc['ALOS3', 'irr']) / (1 + 0.045)) - 1

    # Limpar dados inválidos e ordenar usando a coluna irr_aj
    ytm_clean = ytm_df[['irr_aj']].dropna().sort_values('irr_aj', ascending=True)
    
    if len(ytm_clean) > 0:
        # Criar paleta de cores neutras e elegantes
        colors = [
            "#708090", "#A9A9A9", "#BC987E", "#87A96B", "#8FBC8F",
            "#D2B48C", "#DEB887", "#B0C4DE", "#C0C0C0", "#98A2B3", "#8B9DC3"
        ]
        
        # Criar DataFrame para o gráfico com cores específicas
        plot_data = pd.DataFrame({
            'empresa': ytm_clean.index,
            'irr': ytm_clean['irr_aj'] * 100,
            'cor': colors[:len(ytm_clean)]
        })
        
        # Gráfico principal usando irr_aj com cores vibrantes
        fig_irr = px.bar(
            plot_data,
            x='empresa',
            y='irr',
            title="IRR Real por Empresa",
            color='empresa',
            color_discrete_sequence=colors,
            text='irr'
        )
        
        # Formatar texto nas barras
        fig_irr.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
            textfont=dict(color='white', size=14)
        )
        
        fig_irr.update_layout(
            plot_bgcolor=STK_AZUL,
            paper_bgcolor=STK_AZUL,
            font_color='white',
            xaxis_title="Empresas",
            yaxis_title="IRR Real (%)",
            height=600,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='white', size=14),
                title_font=dict(color='white', size=16)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='white', size=14),
                tickformat='.2f',
                title_font=dict(color='white', size=16)
            )
        )
        
        st.plotly_chart(fig_irr, use_container_width=True)
        
    else:
        st.error("⚠️ Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

    # Rodapé com aviso para refresh (sempre aparece)
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: bold;
                    background-color: rgba(201, 140, 46, 0.1); padding: 15px; border-radius: 8px; margin-top: 20px;'>
            💡 <strong>Para pegar os preços mais recentes e a YTM mais atualizada, dê refresh na página</strong>
        </div>
        """, 
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("📁 Arquivo 'irrdash3.xlsx' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
except Exception as e:
    st.error(f"❌ Erro durante a execução: {str(e)}")





