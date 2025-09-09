import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date

# Tenta importar numpy_financial
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
    """
    Calcula XIRR considerando datas específicas (similar ao Excel)
    """
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

def main():
    # Configuração da página Streamlit
    st.set_page_config(
        page_title="IRR Real Dashboard", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Título principal
    st.markdown("""
        <div style="background: linear-gradient(90deg, #D2691E 0%, #CD853F 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; text-align: center; margin: 0;">IRR Real</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Botão de refresh
    if st.button("🔄 Atualizar Dados", type="primary"):
        st.rerun()
    
    with st.spinner("🔄 Carregando dados e calculando XIRR..."):
        try:
            # Tickes a baixar do Yahoo (apenas os necessários para preços)
            tickers_for_prices = [
                # Consolidações por classes
                "CPLE3", "CPLE6",      # Copel
                "IGTI3", "IGTI4",      # Iguatemi
                "ENGI3", "ENGI4",      # Energisa
                # Demais nomes individuais
                "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3"
            ]

            tickers_sa = [f"{t}.SA" for t in tickers_for_prices]
            closes = yf.download(tickers_sa, period="5d", progress=False)["Close"].ffill()
            last_close = closes.iloc[-1]
            last_close.index = [t.replace(".SA", "") for t in last_close.index]
            prices = last_close.astype(float)

            # Quantidade de ações por classe (fornecidas)
            shares_classes = {
                # Copel
                "CPLE3": 1_300_347_300,
                "CPLE6": 1_679_335_290,
                # Iguatemi
                "IGTI3": 770_992_429,
                "IGTI4": 435_368_756,
                # Energisa
                "ENGI3": 887_231_247,
                "ENGI4": 1_402_193_416,
                # Demais (1 classe)
                "EQTL3": 1_255_510_000,
                "SBSP3": 683_510_000,
                "NEOE3": 1_213_800_000,
                "ENEV3": 1_936_970_000,
                "ELET3": 2_308_630_000,
                "EGIE3": 815_928_000,
                "MULT3": 513_164_000,
                "ALOS3": 542_937_000,
            }

            shares_series = pd.Series(shares_classes).reindex(prices.index)
            mc_raw = prices * shares_series

            # --- Consolidações de market cap ---
            # Copel -> representante: CPLE6
            if {"CPLE3", "CPLE6"}.issubset(mc_raw.index):
                cple_total = mc_raw["CPLE3"] + mc_raw["CPLE6"]
            else:
                st.error("❌ Preços de CPLE3/CPLE6 não encontrados.")
                return

            # Iguatemi -> alimentar IGTI11
            if {"IGTI3", "IGTI4"}.issubset(mc_raw.index):
                igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
            else:
                st.error("❌ Preços de IGTI3/IGTI4 não encontrados.")
                return

            # Energisa -> alimentar ENGI11
            if {"ENGI3", "ENGI4"}.issubset(mc_raw.index):
                engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
            else:
                st.error("❌ Preços de ENGI3/ENGI4 não encontrados.")
                return

            # --- Monta a tabela final 'resultado' com os tickers da planilha ---
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
                    price = np.nan  # não usamos preço direto do Yahoo
                    shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]  # apenas referência
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
            # Reduz market_cap para milhões e mantém 6 primeiros dígitos (conforme sua lógica)
            resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

            # === Carrega Excel com fluxos de caixa ===
            try:
                df = pd.read_excel('irrdash3.xlsx')
            except FileNotFoundError:
                st.error("❌ Arquivo 'irrdash3.xlsx' não encontrado. Coloque-o no diretório de execução.")
                return

            # Primeira linha como cabeçalho
            df.columns = df.iloc[0]
            df = df.iloc[1:]

            # Garante colunas para todos os tickers de 'resultado'
            for t in resultado.index:
                if t not in df.columns:
                    df[t] = pd.NA

            # Escreve market caps (negativos) na primeira linha de dados
            target_row = df.index[0]
            today = datetime.now().date()
            for ticker in resultado.index:
                mc_val = resultado.loc[ticker, 'market_cap']
                df.loc[target_row, ticker] = -abs(mc_val)

            # Garante numéricos nas colunas dos tickers
            for t in resultado.index:
                df[t] = pd.to_numeric(df[t], errors='coerce')

            # === XIRR por ticker (sem zeragem em 2025-12-31) ===
            irr_results = {}
            for t in resultado.index:
                series_cf = df[t].dropna()
                if series_cf.empty:
                    irr_results[t] = np.nan
                    continue

                cashflows = series_cf.values.astype(float).copy()
                n_periods = len(cashflows)

                # Datas: hoje (market cap) + 31/12 de cada ano subsequente
                dates_list = [today]
                for i in range(1, n_periods):
                    future_dt = date(today.year + i - 1, 12, 31)
                    dates_list.append(future_dt)

                irr_results[t] = compute_xirr(cashflows, dates_list)

            ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])
            ytm_df['irr_aj'] = ytm_df['irr']  # inicial

            # Ajuste de IRR real SOMENTE para MULT3, ALOS3 e IGTI11 (deflator 4,5% a.a.)
            for ticker_adj in ['MULT3', 'ALOS3', 'IGTI11']:
                if ticker_adj in ytm_df.index and not pd.isna(ytm_df.loc[ticker_adj, 'irr']):
                    ytm_df.loc[ticker_adj, 'irr_aj'] = ((1 + ytm_df.loc[ticker_adj, 'irr']) / (1 + 0.045)) - 1

            ytm_clean = ytm_df[['irr_aj']].dropna().sort_values('irr_aj', ascending=True)

            if len(ytm_clean) > 0:
                # Preparar dados para o gráfico
                chart_data = pd.DataFrame({
                    'Empresas': ytm_clean.index,
                    'IRR Real (%)': ytm_clean['irr_aj'] * 100
                })

                # Criar o gráfico de barras com Streamlit
                st.bar_chart(
                    data=chart_data.set_index('Empresas'),
                    height=600,
                    use_container_width=True
                )

                # Mostrar estatísticas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Maior IRR", 
                        f"{ytm_clean.iloc[-1].name}",
                        f"{ytm_clean.iloc[-1]['irr_aj']*100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "🔻 Menor IRR", 
                        f"{ytm_clean.iloc[0].name}",
                        f"{ytm_clean.iloc[0]['irr_aj']*100:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "📊 Média IRR", 
                        "",
                        f"{ytm_clean['irr_aj'].mean()*100:.2f}%"
                    )

                # Tabela detalhada
                st.subheader("📈 Resultados Detalhados")
                results_display = pd.DataFrame({
                    'Empresa': ytm_clean.index,
                    'IRR Real (%)': [f"{irr*100:.2f}%" for irr in ytm_clean['irr_aj']]
                })
                st.dataframe(results_display, use_container_width=True, hide_index=True)
                
            else:
                st.warning("⚠️ Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

        except Exception as e:
            st.error(f"❌ Erro durante a execução: {str(e)}")

    # Nota no rodapé
    st.markdown("""
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                    background-color: rgba(0,0,0,0.1); padding: 10px; border-radius: 5px;">
            <p style="margin: 0; text-align: center;">
                💡 Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


