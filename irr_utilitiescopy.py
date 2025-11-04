import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import base64, os, re

# ---------- Finance helpers ----------
try:
    import numpy_financial as npf
except Exception:
    npf = None

# --- Helpers seguros contra NA/NaT ---
def _isna(x):
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def sblank(x: object) -> str:
    return "" if _isna(x) else str(x)

def sfloat(x: object, nd: int = 2) -> str:
    if _isna(x):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return sblank(x)

def compute_irr(cashflows: np.ndarray) -> float:
    values = np.asarray(cashflows, dtype=float)
    if npf is not None:
        try:
            return float(npf.irr(values))
        except Exception:
            pass

    def npv(rate: float) -> float:
        periods = np.arange(values.shape[0], dtype=float)
        return float(np.sum(values / (1.0 + rate) ** periods))

    low, high = -0.99, 10.0
    f_low, f_high = npv(low), npv(high)
    if np.sign(f_low) == np.sign(f_high):
        return np.nan
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
    if len(cashflows) != len(dates):
        return np.nan
    values = np.asarray(cashflows, dtype=float)
    dates = pd.to_datetime(dates)
    base_date = dates[0]
    years = [(d - base_date).days / 365.25 for d in dates]

    def xnpv(rate):
        return sum(cf / (1 + rate) ** yr for cf, yr in zip(values, years))

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
    if pd.isna(value):
        return pd.NA
    total_mln = round(value / 1e6)
    return int(str(int(total_mln))[:digits])

# ---------- Prices (intraday + fallback) ----------
def fetch_latest_prices_intraday_with_fallback(tickers):
    tickers_sa = [f"{t}.SA" for t in tickers]
    prices, source, ts_used = {}, {}, {}

    try:
        intraday = yf.download(tickers_sa, period="1d", interval="1m", progress=False)["Close"]
        if isinstance(intraday, pd.Series):
            intraday = intraday.to_frame()
        intraday = intraday.ffill()
        ts1m = intraday.dropna(how="all").index.max()
    except Exception:
        intraday = pd.DataFrame(); ts1m = None

    try:
        daily = yf.download(tickers_sa, period="5d", progress=False)["Close"].ffill()
        tsd = daily.index[-1] if len(daily.index) else None
    except Exception:
        daily = pd.DataFrame(); tsd = None

    for t, tsa in zip(tickers, tickers_sa):
        val, used_ts, used_src = np.nan, None, None
        if ts1m is not None and tsa in getattr(intraday, "columns", []):
            v = intraday.loc[ts1m, tsa]
            if pd.notna(v):
                val = float(v); used_ts = ts1m; used_src = "intraday 1m"
        if (pd.isna(val)) and (tsa in getattr(daily, "columns", [])) and len(daily):
            v = daily.iloc[-1][tsa]
            if pd.notna(v):
                val = float(v); used_ts = tsd; used_src = "daily close"
        prices[t] = val; source[t] = used_src if used_src else "N/A"; ts_used[t] = used_ts

    return pd.Series(prices, name="preco"), pd.DataFrame({"Fonte": pd.Series(source), "Timestamp": pd.Series(ts_used)})

# ---------- Duration loader (aba 'duration') ----------
def load_duration_map(excel_path="irrdash3.xlsx", sheet="duration") -> pd.Series:
    try:
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    except Exception:
        return pd.Series(dtype="float64")

    header_row = None
    for i, row in raw.iterrows():
        if any(isinstance(v, str) and "duration" in v.strip().lower() for v in row):
            header_row = i; break
    if header_row is None:
        return pd.Series(dtype="float64")

    header_vals = raw.iloc[header_row].tolist()
    dur_idx = None
    for j, v in enumerate(header_vals):
        if isinstance(v, str) and "duration" in v.strip().lower():
            dur_idx = j; break
    if dur_idx is None:
        return pd.Series(dtype="float64")

    df = raw.iloc[header_row+1:].reset_index(drop=True)

    ticker_idx, best = None, -1
    for j in range(df.shape[1]):
        if j == dur_idx: continue
        s = df.iloc[:, j].dropna(); cnt = 0
        for x in s:
            if isinstance(x, str):
                token = x.strip().upper()
                if re.fullmatch(r"[A-Z]{3,5}\d{0,2}", token):
                    cnt += 1
        if cnt > best: best, ticker_idx = cnt, j
    if ticker_idx is None:
        return pd.Series(dtype="float64")

    tickers = df.iloc[:, ticker_idx].astype(str).str.strip().str.upper()
    durations = pd.to_numeric(df.iloc[:, dur_idx], errors="coerce")
    out = pd.Series(durations.values, index=tickers.values)
    out = out[~out.index.isin(["", "NAN", "NONE"])].dropna()
    return out

# ---------- Helpers de formatação ----------
def format_ts_brt(ts) -> str:
    t = pd.to_datetime(ts, errors="coerce")
    if _isna(t): return ""
    try:
        if t.tzinfo is None: t = t.tz_localize("UTC")
        t = t.tz_convert("America/Sao_Paulo")
    except Exception:
        pass
    return t.strftime("%Y-%m-%d %H:%M")

def build_price_table_html(df: pd.DataFrame) -> str:
    rows_html = []
    for _, r in df.iterrows():
        fonte = sblank(r.get("Fonte"))
        badge_class = "badge-live" if "intraday" in fonte.lower() else "badge-daily"
        rows_html.append(
            "<tr>"
            f"<td>{sblank(r.get('Ticker'))}</td>"
            f"<td class='num'>{sfloat(r.get('Preço'))}</td>"
            f"<td><span class='badge {badge_class}'>{fonte}</span></td>"
            f"<td>{sblank(r.get('Timestamp'))}</td>"
            f"<td class='num'>{sfloat(r.get('Duration'))}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'>"
        "<div class='table-title'>🕒 Preços usados (Yahoo Finance)</div>"
        "<table class='styled-table'>"
        "<thead><tr><th>Ticker</th><th>Preço</th><th>Fonte</th><th>Timestamp</th><th>Duration</th></tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table>"
        "<div class='table-note'>intraday 1m pode ter atraso de ~15 min • Timestamp em horário de Brasília.</div>"
        "</div>"
    )

# ---------- App ----------
def main():
    st.set_page_config(page_title="IRR Real Dashboard", page_icon="📈",
                       layout="wide", initial_sidebar_state="collapsed")

    # ====== THEME / CSS ======
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root{
  --stk-bg:#0e314a; --stk-gold:#BD8A25; --stk-grid:rgba(255,255,255,.12);
  --stk-note-bg:rgba(255,209,84,.06); --stk-note-bd:rgba(255,209,84,.25); --stk-note-fg:#FFD14F;
  --stk-header-bg:#ffffff; --stk-header-fg:#0e314a;
}

html, body, [class^="css"]{font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;}
html, body, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stDecoration"],
header[data-testid="stHeader"], header[data-testid="stHeader"] *,
[data-testid="stToolbar"], [data-testid="stToolbar"] *{background:var(--stk-bg) !important;}
header[data-testid="stHeader"]{box-shadow:none !important;}

.block-container{padding-top:.75rem; padding-bottom:.75rem; max-width:none !important; padding-left:1.25rem; padding-right:1.25rem;}

/* Header */
.app-header{background:var(--stk-header-bg); padding:18px 20px; border-radius:12px;
            margin:16px 0 16px; box-shadow:0 1px 0 rgba(0,0,0,.04) inset, 0 6px 20px rgba(0,0,0,.10);}
.header-inner{position:relative; height:48px; display:flex; align-items:center; justify-content:center;}
.stk-logo{position:absolute; left:16px; top:50%; transform:translateY(-50%); height:44px; width:auto; filter:drop-shadow(0 1px 0 rgba(0,0,0,.10));}
.app-header h1{margin:0; color:var(--stk-header-fg); font-weight:800; letter-spacing:.4px;}

/* Nota */
.footer-note{background:var(--stk-note-bg); border:1px solid var(--stk-note-bd); border-radius:10px;
             padding:16px 18px; color:var(--stk-note-fg); text-align:center; margin:18px 0 8px; font-size:1.1rem; font-weight:600; width:100%;}

/* Tabela dark */
.table-wrap{margin:14px 0 8px;}
.table-title{color:#cfe8ff; font-weight:700; margin:0 0 8px; font-size:1.1rem;}
.styled-table{width:100%; border-collapse:separate; border-spacing:0; background:rgba(255,255,255,.03);
              border:1px solid rgba(255,255,255,.08); border-radius:12px; overflow:hidden;}
.styled-table thead th{background:rgba(255,255,255,.06); color:#fff; text-align:left; padding:12px 14px; font-weight:600;
                       border-bottom:1px solid rgba(255,255,255,.08);}
.styled-table tbody td{color:#fff; padding:12px 14px; border-bottom:1px solid rgba(255,255,255,.06);}
.styled-table tbody tr:nth-child(even){background:rgba(255,255,255,.02);}
.styled-table tbody tr:last-child td{border-bottom:none;}
.styled-table td.num{text-align:right; font-variant-numeric:tabular-nums;}
.badge{display:inline-block; padding:4px 8px; border-radius:999px; font-size:.85rem; border:1px solid transparent;}
.badge-live{background:#1f6f5f; color:#d3fff1; border-color:rgba(211,255,241,.25);}
.badge-daily{background:#6f5f1f; color:#fff3c2; border-color:rgba(255,243,194,.25);}
.table-note{color:#cfe8ff; opacity:.8; font-size:.85rem; margin-top:8px;}
svg text{font-family:Inter, system-ui, sans-serif !important;}
</style>
""", unsafe_allow_html=True)

    # Header
    LOGO_PATH = "STKGRAFICO.png"
    logo_b64 = None
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    if logo_b64:
        st.markdown(
            "<div class='app-header'><div class='header-inner'>"
            f"<img class='stk-logo' src='data:image/png;base64,{logo_b64}' alt='STK'/>"
            "<h1>IRR Real</h1>"
            "</div></div>", unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='app-header'><div class='header-inner'><h1>IRR Real</h1></div></div>", unsafe_allow_html=True)

    try:
        # ====== Tickers ======
        tickers_for_prices = [
            "CPLE3","CPLE6","IGTI3","IGTI4","ENGI3","ENGI4","ENGI11",
            "EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3","MULT3","ALOS3",
        ]

        # ====== Preços ======
        prices_series, meta = fetch_latest_prices_intraday_with_fallback(tickers_for_prices)
        prices = prices_series.astype(float)

        # ====== Shares ======
        shares_classes = {
            "CPLE3": 1_300_347_300, "CPLE6": 1_679_335_290,
            "IGTI3": 770_992_429, "IGTI4": 435_368_756,
            "ENGI3": 887_231_247, "ENGI4": 1_402_193_416,
            "ENGI11": 2_289_420_000,
            "EQTL3": 1_255_510_000, "SBSP3": 683_510_000,
            "NEOE3": 1_213_800_000, "ENEV3": 1_936_970_000,
            "ELET3": 2_308_630_000, "EGIE3": 815_928_000,
            "MULT3": 513_164_000, "ALOS3": 542_937_000,
        }
        shares_series = pd.Series(shares_classes).reindex(prices.index)
        mc_raw = prices * shares_series

        # ====== Consolidações ======
        engi11_price = prices.get("ENGI11", np.nan)
        engi11_shares = shares_series.get("ENGI11", np.nan)
        if pd.notna(engi11_price) and engi11_price > 0:
            engi_total = engi11_price * engi11_shares
            engi_calc_source = "ENGI11 price × ENGI11 shares (fixo)"
        else:
            if {"ENGI3","ENGI4"}.issubset(mc_raw.index):
                engi_total = mc_raw["ENGI3"] + mc_raw["ENGI4"]
                engi_calc_source = "fallback: ENGI3×shares + ENGI4×shares"
                st.info("ENGI11: preço não disponível; usando fallback (ENGI3+ENGI4).")
            else:
                raise ValueError("Não foi possível calcular o market cap de ENGI.")

        if {"CPLE3","CPLE6"}.issubset(mc_raw.index):
            cple_total = mc_raw["CPLE3"] + mc_raw["CPLE6"]
        else:
            raise ValueError("Preços/Ações de CPLE3/CPLE6 não encontrados.")
        if {"IGTI3","IGTI4"}.issubset(mc_raw.index):
            igti_total = mc_raw["IGTI3"] + mc_raw["IGTI4"]
        else:
            raise ValueError("Preços/Ações de IGTI3/IGTI4 não encontrados.")

        # ====== Tabela final (para XIRR)
        final_tickers = ["CPLE6","EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3","MULT3","ALOS3","IGTI11","ENGI11"]
        rows = []
        for t in final_tickers:
            if t == "CPLE6":
                price = prices.get("CPLE6", np.nan)
                shares = shares_classes["CPLE3"] + shares_classes["CPLE6"]
                mc = cple_total
            elif t == "IGTI11":
                price = np.nan
                shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]
                mc = igti_total
            elif t == "ENGI11":
                price = prices.get("ENGI11", np.nan)
                shares = shares_classes["ENGI11"]
                mc = engi_total
            else:
                price = prices.get(t, np.nan)
                shares = shares_series.get(t, np.nan)
                mc = price * shares if (pd.notna(price) and pd.notna(shares)) else np.nan
            rows.append({"ticker": t, "price": price, "shares": shares, "market_cap": mc})

        resultado = pd.DataFrame(rows).set_index("ticker")
        resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

        # ====== Excel dos fluxos ======
        try:
            df = pd.read_excel("irrdash3.xlsx")
        except FileNotFoundError:
            st.error("❌ Arquivo 'irrdash3.xlsx' não encontrado.")
            return

        # Header = primeira linha
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        # 🔧 Normaliza nomes de colunas para garantir match com os tickers
        df.columns = [str(c).strip().upper() for c in df.columns]

        # Garante colunas para todos os tickers finais
        for t in resultado.index:
            if t not in df.columns:
                df[t] = pd.NA

        # Injeta CF inicial na primeira linha de dados
        target_row = 0
        today = datetime.now().date()
        for t in resultado.index:
            mc_val = resultado.loc[t, "market_cap"]
            if pd.isna(mc_val):
                continue
            df.loc[target_row, t] = -abs(mc_val)

        # Converte para numérico
        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors="coerce")

        # ====== XIRR ======
        irr_results = {}
        missing = []
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.size < 2:
                irr_results[t] = np.nan
                missing.append(t)
                continue
            cashflows = series_cf.values.astype(float)
            n = len(cashflows)
            dates_list = [today] + [date(today.year + j - 1, 12, 31) for j in range(1, n)]
            irr_results[t] = compute_xirr(cashflows, dates_list)

        if missing:
            st.caption("Sem XIRR (faltam fluxos futuros) para: " + ", ".join(missing))

        ytm_df = pd.DataFrame.from_dict(irr_results, orient="index", columns=["irr"])
        ytm_df["irr_aj"] = ytm_df["irr"]
        for t in ["MULT3","ALOS3","IGTI11"]:
            if t in ytm_df.index and not pd.isna(ytm_df.loc[t, "irr"]):
                ytm_df.loc[t, "irr_aj"] = ((1 + ytm_df.loc[t, "irr"]) / (1 + 0.045)) - 1

        ytm_clean = ytm_df[["irr_aj"]].dropna().sort_values("irr_aj", ascending=True)
        ytm_plot = ytm_clean[~ytm_clean.index.isin(["ELET3", "ELET6"])]

        # ====== Gráfico ======
        if len(ytm_plot) == 0:
            st.warning("Nenhum ticker disponível para o gráfico de IRR após os filtros.")
        else:
            plot_data = pd.DataFrame({
                "empresa": ytm_plot.index,
                "irr": (ytm_plot["irr_aj"] * 100).round(2),
            }).reset_index(drop=True)

            destaque = {"EQTL3", "EGIE3", "IGTI11", "SBSP3", "CPLE6"}
            cor_ouro = "rgb(201,140,46)"
            cor_azul = "rgb(16,144,178)"
            bar_colors = [cor_ouro if e in destaque else cor_azul for e in plot_data["empresa"]]

            fig = go.Figure(go.Bar(
                x=plot_data["empresa"],
                y=plot_data["irr"],
                text=[f"{v:.2f}%" for v in plot_data["irr"]],
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
            ))
            fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="white", size=14))

            ymax = max(12.0, float(plot_data["irr"].max()) * 1.10)
            fig.update_layout(
                bargap=0.12, plot_bgcolor="#0e314a", paper_bgcolor="#0e314a",
                uniformtext_minsize=10,
                font=dict(family="Inter, system-ui, sans-serif", color="white", size=14),
                xaxis=dict(title="Empresas", showgrid=False, showline=False, zeroline=False),
                yaxis=dict(title="IRR Real (%)", range=[4.0, ymax], dtick=1,
                           gridcolor="rgba(255,255,255,.12)", zeroline=False),
                margin=dict(l=10, r=10, t=6, b=62), showlegend=False, height=560,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ====== Duration ======
        duration_map = load_duration_map("irrdash3.xlsx", "duration").copy()
        def set_if_missing(label, value):
            if (label not in duration_map.index) or pd.isna(duration_map.loc[label]):
                duration_map.loc[label] = value
        if "IGTI11" in duration_map.index:
            v = duration_map.loc["IGTI11"]; set_if_missing("IGTI3", v); set_if_missing("IGTI4", v)
        if "ENGI11" in duration_map.index:
            v = duration_map.loc["ENGI11"]
            if pd.notna(v): set_if_missing("ENGI3", v); set_if_missing("ENGI4", v)

        order = ["CPLE3","CPLE6","IGTI3","IGTI4","ENGI3","ENGI4","ENGI11",
                 "EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3","MULT3","ALOS3"]
        tbl = pd.DataFrame({"Preço": prices.reindex(order)})
        tbl["Fonte"] = meta["Fonte"].reindex(order)
        tbl["Timestamp"] = meta["Timestamp"].reindex(order).map(format_ts_brt)
        tbl = tbl.rename_axis("Ticker").reset_index()
        tbl["Duration"] = tbl["Ticker"].map(duration_map)
        tbl["__dur_num"] = pd.to_numeric(tbl["Duration"], errors="coerce")
        tbl = tbl.sort_values(by="__dur_num", ascending=False, na_position="last").drop(columns="__dur_num")

        st.markdown(build_price_table_html(tbl), unsafe_allow_html=True)
        st.markdown("<div class='footer-note'>💡 Para pegar os preços mais recentes e a XIRR mais atualizada, dê refresh na página</div>", unsafe_allow_html=True)
        st.caption(f"ENGI total calculado via: {engi_calc_source}")

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

if __name__ == "__main__":
    main()





















