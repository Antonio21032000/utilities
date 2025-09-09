import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date

# Tenta importar numpy_financial
try:
    import numpy_financial as npf
except Exception:
    npf = None


def compute_irr(cashflows: np.ndarray) -> float:
    """Calcula IRR usando numpy_financial ou método de bisseção como fallback."""
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
    """Calcula XIRR considerando datas específicas (similar ao Excel)."""
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
    """Converte market cap para milhões e mantém apenas os primeiros dígitos."""
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])


def main():
    # ===== Configuração da página =====
    st.set_page_config(
        page_title="IRR Real Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ===== CSS / Tema =====
    st.markdown(
        """
<style>
/* ===== TIPOGRAFIA =====
   -> Se tiver as fontes, coloque os .woff2 em ./fonts/
   -> Fallback para Inter se não existirem
*/
@font-face{
  font-family:"STK Display";
  src:url("fonts/STK-Display.woff2") format("woff2");
  font-weight:600 800; font-style:normal; font-display:swap;
}
@font-face{
  font-family:"STK Text";
  src:url("fonts/STK-Text.woff2") format("woff2");
  font-weight:300 700; font-style:normal; font-display:swap;
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root{
  --stk-bg:#0e314a;         /* azul de fundo */
  --stk-gold:#BD8A25;       /* cabeçalho mostarda */
  --stk-grid:rgba(255,255,255,.12);
  --stk-note-bg:rgba(255,209,84,.06);
  --stk-note-bd:rgba(255,209,84,.25);
  --stk-note-fg:#FFD14F;
}

html, body, [class^="css"] {
  font-family:"STK Text", Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
}

/* ===== Pinte o app todo (inclui topo/toolbar) ===== */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stDecoration"],
[data-testid="stHeader"]{
  background:var(--stk-bg) !important;
}
[data-testid="stHeader"]{ box-shadow:none !important; }

/* Container padrão com menos respiro */
.block-container{ padding-top:.75rem; padding-bottom:.75rem; }

/* Cabeçalho chapado */
.app-header{
  background:var(--stk-gold);
  padding:28px 24px; border-radius:12px; text-align:center;
  margin:16px 0 24px;
  box-shadow:0 1px 0 rgba(255,255,255,.05) inset, 0 6px 20px rgba(0,0,0,.15);
}
.app-header h1{
  font-family:"STK Display", Inter, system-ui, sans-serif;
  font-weight:800; letter-spacing:.4px; color:#fff; margin:0;
}

/* Nota de rodapé — maior e mais legível */
.footer-note{
  background:var(--stk-note-bg); border:1px solid var(--stk-note-bd);
  border-radius:10px; padding:12px 16px; color:var(--stk-note-fg);
  text-align:center; margin-top:16px; font-size:1.1rem; font-weight:600;
}

/* Garantir tipografia também no SVG dos gráficos */
svg text{ font-family:"STK Text", Inter, system-ui, sans-serif !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    # ===== Header =====
    st.markdown('<div class="app-header"><h1>IRR Real</h1></div>', unsafe_allow_html=True)

    try:
        # ===== Preços Yahoo (último fechamento disponível) =====
        tickers_for_prices = [
            # Consolidações por classes
            "CPLE3", "CPLE6",      # Copel
            "IGTI3", "IGTI4",      # Iguatemi
            "ENGI3", "ENGI4",      # Energisa
            # Demais nomes individuais
            "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3", "MULT3", "ALOS3",
        ]

        tickers_sa = [f"{t}.SA" for t in tickers_for_prices]
        closes = yf.download(tickers_sa, period="5d", progress=False)["Close"].ffill()
        last_close = closes.iloc[-1]
        last_close.index = [t.replace(".SA", "") for t in last_close.index]
        prices = last_close.astype(float)

        # ===== Quantidade de ações por classe =====
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

        # ===== Consolidações de market cap =====
        if {"CPLE3", "CPLE6"}.issubset(mc_raw.index):
            cple_total = mc_raw["CPLE3"] + mc_raw["CPLE6"]
        else:
            raise ValueError("Preços de CPLE3/CPLE6 não encontrados.")

        if {"IGTI3", "IGTI4"}.issubset(mc_raw.index):
            igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
        else:
            raise ValueError("Preços de IGTI3/IGTI4 não encontrados.")

        if {"ENGI3", "ENGI4"}.issubset(mc_raw.index):
            engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
        else:
            raise ValueError("Preços de ENGI3/ENGI4 não encontrados.")

        # ===== Tabela final com tickers-alvo =====
        final_tickers = [
            "CPLE6", "EQTL3", "SBSP3", "NEOE3", "ENEV3", "ELET3", "EGIE3",
            "MULT3", "ALOS3", "IGTI11", "ENGI11",
        ]

        rows = []
        for t in final_tickers:
            if t == "CPLE6":
                price = prices.get("CPLE6", np.nan)
                shares = shares_classes["CPLE3"] + shares_classes["CPLE6"]
                mc = cple_total
            elif t == "IGTI11":
                price = np.nan  # não usamos preço direto do Yahoo
                shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]
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
        # Reduz market_cap para milhões e mantém 6 primeiros dígitos
        resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

        # ===== Carrega Excel com fluxos de caixa =====
        try:
            df = pd.read_excel("irrdash3.xlsx")
        except FileNotFoundError:
            st.error("❌ Arquivo 'irrdash3.xlsx' não encontrado.")
            return

        # Primeira linha vira cabeçalho
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
            mc_val = resultado.loc[ticker, "market_cap"]
            df.loc[target_row, ticker] = -abs(mc_val)

        # Converte colunas para numérico
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors="coerce")

        # ===== XIRR por ticker (sem zeragem em 2025-12-31) =====
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
            for j in range(1, n_periods):
                future_dt = date(today.year + j - 1, 12, 31)
                dates_list.append(future_dt)

            irr_results[t] = compute_xirr(cashflows, dates_list)

        ytm_df = pd.DataFrame.from_dict(irr_results, orient="index", columns=["irr"])
        ytm_df["irr_aj"] = ytm_df["irr"]  # inicial

        # Ajuste de IRR real SOMENTE para MULT3, ALOS3 e IGTI11 (deflator 4,5% a.a.)
        for ticker_adj in ["MULT3", "ALOS3", "IGTI11"]:
            if ticker_adj in ytm_df.index and not pd.isna(ytm_df.loc[ticker_adj, "irr"]):
                ytm_df.loc[ticker_adj, "irr_aj"] = ((1 + ytm_df.loc[ticker_adj, "irr"]) / (1 + 0.045)) - 1

        ytm_clean = ytm_df[["irr_aj"]].dropna().sort_values("irr_aj", ascending=True)

        # ===== Gráfico =====
        if len(ytm_clean) > 0:
            # Paleta suave (ordem acompanha o sort crescente)
            soft_palette = [
                "#9AA7B2", "#AEB8C2", "#C7B79A", "#A7C5A4", "#BFD7B5",
                "#BBD0E6", "#D4DBF0", "#E2DBC5", "#E8E1CF", "#D3D9E6",
            ]

            plot_data = pd.DataFrame({
                "empresa": ytm_clean.index,
                "irr": (ytm_clean["irr_aj"] * 100).round(2),
            }).reset_index(drop=True)

            bar_colors = soft_palette[: len(plot_data)]

            fig_irr = go.Figure(
                go.Bar(
                    x=plot_data["empresa"],
                    y=plot_data["irr"],
                    text=[f"{v:.2f}%" for v in plot_data["irr"]],
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
                )
            )

            fig_irr.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont=dict(color="white", size=14),
            )

            ymax = max(12.0, float(plot_data["irr"].max()) * 1.10)

            fig_irr.update_layout(
                bargap=0.2,
                plot_bgcolor="#0e314a",
                paper_bgcolor="#0e314a",
                font=dict(
                    family="STK Text, Inter, system-ui, sans-serif",
                    color="white",
                    size=14,
                ),
                xaxis=dict(
                    title="Empresas",
                    tickfont=dict(size=12, color="white"),
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    title="IRR Real (%)",
                    range=[0, ymax],
                    gridcolor="rgba(255,255,255,.12)",
                    zeroline=False,
                    tickfont=dict(color="white"),
                ),
                margin=dict(l=30, r=30, t=10, b=70),
                showlegend=False,
            )

            st.plotly_chart(fig_irr, use_container_width=True)

            # Nota de rodapé
            st.markdown(
                """
<div class="footer-note">
  💡 Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.error("⚠️ Não foi possível calcular IRR para nenhuma empresa.")

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")


if __name__ == "__main__":
    main()



