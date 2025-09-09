import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    st.title("📊 Dashboard IRR por Empresa")
    st.write("🔄 Carregando dados e calculando XIRR...")

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
            raise ValueError("Preços de CPLE3/CPLE6 não encontrados.")

        # Iguatemi -> alimentar IGTI11
        if {"IGTI3", "IGTI4"}.issubset(mc_raw.index):
            igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
        else:
            raise ValueError("Preços de IGTI3/IGTI4 não encontrados.")

        # Energisa -> alimentar ENGI11
        if {"ENGI3", "ENGI4"}.issubset(mc_raw.index):
            engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
        else:
            raise ValueError("Preços de ENGI3/ENGI4 não encontrados.")

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

        st.subheader("📊 Dados (market cap em milhões, 6 primeiros dígitos)")
        st.dataframe(resultado)

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

        st.subheader("💰 Fluxos de caixa por empresa")
        cashflow_display = df[resultado.index].copy()
        st.dataframe(cashflow_display)

        # === XIRR por ticker (sem zeragem em 2025-12-31) ===
        st.subheader("🧮 Calculando XIRR para cada empresa (com datas específicas)")
        irr_results = {}
        
        progress_bar = st.progress(0)
        for i, t in enumerate(resultado.index):
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue

            cashflows = series_cf.values.astype(float).copy()
            n_periods = len(cashflows)

            # Datas: hoje (market cap) + 31/12 de cada ano subsequente
            dates_list = [today]
            for j in range(1, n_periods):
                future_dt = date(today.year + j - 1, 12, 31)
                dates_list.append(future_dt)

            cf_with_dates = pd.DataFrame({
                'Data': dates_list,
                'Fluxo_de_Caixa': cashflows,
                'Ano': [d.year for d in dates_list]
            })

            with st.expander(f"📊 {t} - Fluxos de caixa com datas (para XIRR)"):
                st.dataframe(cf_with_dates)

            irr_results[t] = compute_xirr(cashflows, dates_list)
            if pd.isna(irr_results[t]):
                st.write(f"  ➤ {t} XIRR calculado: N/A")
            else:
                st.write(f"  ➤ {t} XIRR calculado: {irr_results[t]:.4f} ({irr_results[t]*100:.2f}%)")
            
            progress_bar.progress((i + 1) / len(resultado.index))

        ytm_df = pd.DataFrame.from_dict(irr_results, orient='index', columns=['irr'])
        ytm_df['irr_aj'] = ytm_df['irr']  # inicial

        # Ajuste de IRR real SOMENTE para MULT3, ALOS3 e IGTI11 (deflator 4,5% a.a.)
        st.subheader("🔧 Aplicando ajuste de IRR real (deflator 4,5% a.a.) para MULT3, ALOS3 e IGTI11")
        for ticker_adj in ['MULT3', 'ALOS3', 'IGTI11']:
            if ticker_adj in ytm_df.index and not pd.isna(ytm_df.loc[ticker_adj, 'irr']):
                ytm_df.loc[ticker_adj, 'irr_aj'] = ((1 + ytm_df.loc[ticker_adj, 'irr']) / (1 + 0.045)) - 1
                st.write(f"  {ticker_adj}: {ytm_df.loc[ticker_adj, 'irr_aj']*100:.2f}% (real)")

        ytm_clean = ytm_df[['irr_aj']].dropna().sort_values('irr_aj', ascending=True)

        st.subheader("📈 Resultados finais - IRR (ajustada onde aplicável) ordenada")
        
        # Mostra tabela de resultados
        results_display = ytm_clean.copy()
        results_display['irr_aj'] = results_display['irr_aj'] * 100
        results_display.columns = ['IRR (%)']
        st.dataframe(results_display.round(2))

        if len(ytm_clean) > 0:
            st.success(f"🏆 Maior IRR: {ytm_clean.iloc[-1].name} ({ytm_clean.iloc[-1]['irr_aj']*100:.2f}%)")
            st.info(f"🔻 Menor IRR: {ytm_clean.iloc[0].name} ({ytm_clean.iloc[0]['irr_aj']*100:.2f}%)")

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
        else:
            st.warning("⚠️ Não foi possível calcular IRR para nenhuma empresa. Verifique os dados do arquivo Excel.")

    except Exception as e:
        st.error(f"❌ Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()
