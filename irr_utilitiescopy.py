import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import base64, os, re, tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# =========================================================
# yfinance: força tz-cache para um lugar seguro no Cloud
# =========================================================
try:
    from yfinance.utils import set_tz_cache_location
    _tz_dir = os.path.join(tempfile.gettempdir(), "py-yfinance-tzcache")
    os.makedirs(_tz_dir, exist_ok=True)
    set_tz_cache_location(_tz_dir)
except Exception:
    pass

# ---------- Finance helpers ----------
try:
    import numpy_financial as npf
except Exception:
    npf = None

# ---------- Paths (cache local no Streamlit Cloud) ----------
CACHE_DIR = tempfile.gettempdir()
LAST_PRICES_PATH = os.path.join(CACHE_DIR, "last_prices.pkl")
LAST_META_PATH = os.path.join(CACHE_DIR, "last_prices_meta.pkl")

# =========================================================
# Helpers seguros contra NA/NaT
# =========================================================
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

# =========================================================
# IRR helpers
# =========================================================
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
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        return np.nan

    base_date = dates[0]
    years = [(d - base_date).days / 365.25 for d in dates]

    def xnpv(rate):
        return sum(cf / (1 + rate) ** year for cf, year in zip(values, years))

    rate = guess
    for _ in range(80):
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
    try:
        if pd.isna(value):
            return np.nan
        total_mln = int(round(float(value) / 1e6))
        return int(str(abs(total_mln))[:digits])
    except Exception:
        return np.nan

# =========================================================
# Cache helpers (pickle – sem pyarrow)
# =========================================================
def load_last_prices():
    if os.path.exists(LAST_PRICES_PATH) and os.path.exists(LAST_META_PATH):
        try:
            prices = pd.read_pickle(LAST_PRICES_PATH)
            meta = pd.read_pickle(LAST_META_PATH)
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze()
            prices = pd.Series(prices, name="preco")
            if not isinstance(meta, pd.DataFrame):
                meta = pd.DataFrame(meta)
            return prices, meta
        except Exception:
            return None, None
    return None, None

def save_last_prices(prices: pd.Series, meta: pd.DataFrame):
    try:
        pd.Series(prices, name="preco").to_pickle(LAST_PRICES_PATH)
        meta.to_pickle(LAST_META_PATH)
    except Exception:
        pass

# =========================================================
# Timestamp format
# =========================================================
def format_ts_brt(ts) -> str:
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return ""
    try:
        if getattr(t, "tzinfo", None) is None:
            t = t.tz_localize("UTC")
        t = t.tz_convert("America/Sao_Paulo")
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""

# =========================================================
# Duration loader
# =========================================================
def load_duration_map(excel_path="irrdash3.xlsx", sheet="duration") -> pd.Series:
    try:
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    except Exception:
        return pd.Series(dtype="float64")

    header_row = None
    for i, row in raw.iterrows():
        if any(isinstance(v, str) and "duration" in v.strip().lower() for v in row):
            header_row = i
            break
    if header_row is None:
        return pd.Series(dtype="float64")

    header_vals = raw.iloc[header_row].tolist()
    dur_idx = None
    for j, v in enumerate(header_vals):
        if isinstance(v, str) and "duration" in v.strip().lower():
            dur_idx = j
            break
    if dur_idx is None:
        return pd.Series(dtype="float64")

    df = raw.iloc[header_row + 1 :].reset_index(drop=True)

    ticker_idx, best_score = None, -1
    for j in range(df.shape[1]):
        if j == dur_idx:
            continue
        s = df.iloc[:, j].dropna()
        cnt = 0
        for x in s:
            if isinstance(x, str):
                token = x.strip().upper()
                if re.fullmatch(r"[A-Z]{3,5}\d{0,2}", token):
                    cnt += 1
        if cnt > best_score:
            best_score, ticker_idx = cnt, j

    if ticker_idx is None:
        return pd.Series(dtype="float64")

    tickers = df.iloc[:, ticker_idx].astype(str).str.strip().str.upper()
    durations = pd.to_numeric(df.iloc[:, dur_idx], errors="coerce")

    out = pd.Series(durations.values, index=tickers.values)
    out = out[~out.index.isin(["", "NAN", "NONE"])].dropna()
    return out

# =========================================================
# UI table HTML
# =========================================================
def build_price_table_html(df: pd.DataFrame) -> str:
    rows_html = []
    for _, r in df.iterrows():
        fonte = sblank(r.get("Fonte"))
        badge_class = "badge-live" if "intraday" in fonte.lower() else "badge-daily"
        preco = sfloat(r.get("Preço"))
        ts = sblank(r.get("Timestamp"))
        dur = sfloat(r.get("Duration"))
        rows_html.append(
            "<tr>"
            f"<td>{sblank(r.get('Ticker'))}</td>"
            f"<td class='num'>{preco}</td>"
            f"<td><span class='badge {badge_class}'>{fonte}</span></td>"
            f"<td>{ts}</td>"
            f"<td class='num'>{dur}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'>"
        "<div class='table-title'>🕒 Preços usados (Yahoo Finance)</div>"
        "<table class='styled-table'>"
        "<thead><tr><th>Ticker</th><th>Preço</th><th>Fonte</th><th>Timestamp</th><th>Duration</th></tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table>"
        "<div class='table-note'>intraday pode ter atraso • Timestamp em horário de Brasília.</div>"
        "</div>"
    )

# =========================================================
# Yahoo fetch (robusto p/ 1 ticker e muitos tickers)
# =========================================================
def _yf_download_close(tickers_sa, period, interval, timeout_s=8):
    """
    Retorna um DF apenas com Close, colunas = tickers_sa.
    Funciona mesmo quando tickers_sa tem 1 item (quando o yfinance devolve colunas 'Close').
    """
    df = yf.download(
        tickers_sa,
        period=period,
        interval=interval,
        progress=False,
        group_by="column",
        auto_adjust=False,
        actions=False,
        threads=False,      # importantíssimo no Cloud
        timeout=timeout_s,  # timeout do requests interno
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Caso multiindex (vários tickers)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"].copy()
        return close.ffill()

    # Caso 1 ticker
    if "Close" in df.columns:
        close = df[["Close"]].copy()
        ticker_name = tickers_sa[0] if len(tickers_sa) >= 1 else "Close"
        close = close.rename(columns={"Close": ticker_name})
        return close.ffill()

    return pd.DataFrame()

def fetch_prices_yahoo_safe(tickers, max_total_seconds=10):
    """
    - Probe rápido para detectar bloqueio geral (sem travar UX).
    - Depois baixa tudo em 1 chamada (intraday) e fallback (daily).
    - Timeout duro total via ThreadPoolExecutor.
    """
    tickers_sa = [f"{t}.SA" for t in tickers]
    probe_ticker = tickers_sa[0] if tickers_sa else "VALE3.SA"

    def _job():
        probe = _yf_download_close([probe_ticker], period="5d", interval="1d", timeout_s=6)
        if probe.empty:
            return None, None, "probe_failed"

        intraday = _yf_download_close(tickers_sa, period="1d", interval="1m", timeout_s=8)
        ts_intraday = intraday.dropna(how="all").index.max() if len(intraday) else None

        daily = _yf_download_close(tickers_sa, period="5d", interval="1d", timeout_s=8)
        ts_daily = daily.dropna(how="all").index.max() if len(daily) else None

        prices, source, ts_used = {}, {}, {}

        for t, tsa in zip(tickers, tickers_sa):
            val, used_ts, used_src = np.nan, None, None

            if ts_intraday is not None and tsa in intraday.columns:
                v = intraday.loc[ts_intraday, tsa]
                if pd.notna(v):
                    val = float(v)
                    used_ts = ts_intraday
                    used_src = "intraday 1m"

            if pd.isna(val) and ts_daily is not None and tsa in daily.columns:
                v = daily.loc[ts_daily, tsa]
                if pd.notna(v):
                    val = float(v)
                    used_ts = ts_daily
                    used_src = "daily close"

            prices[t] = val
            source[t] = used_src if used_src is not None else "N/A"
            ts_used[t] = used_ts

        price_series = pd.Series(prices, name="preco")
        meta = pd.DataFrame({"Fonte": pd.Series(source), "Timestamp": pd.Series(ts_used)})
        return price_series, meta, "ok"

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_job)
        try:
            prices, meta, status = fut.result(timeout=max_total_seconds)
        except FuturesTimeout:
            return None, None, "timeout"
        except Exception:
            return None, None, "error"

    if status != "ok" or prices is None:
        return None, None, status

    return prices, meta, "ok"

# =========================================================
# App
# =========================================================
def main():
    st.set_page_config(
        page_title="IRR Real Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root{
  --stk-bg:#0e314a; --stk-gold:#BD8A25; --stk-grid:rgba(255,255,255,.12);
  --stk-note-bg:rgba(255,209,84,.06); --stk-note-bd:rgba(255,209,84,.25); --stk-note-fg:#FFD14F;
  --stk-header-bg:#ffffff; --stk-header-fg:#0e314a;
}
html, body, [class^="css"]{font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;}
html, body, .stApp,[data-testid="stAppViewContainer"],[data-testid="stDecoration"],
header[data-testid="stHeader"], header[data-testid="stHeader"] *,
[data-testid="stToolbar"], [data-testid="stToolbar"] *{background:var(--stk-bg) !important;}
header[data-testid="stHeader"]{box-shadow:none !important;}
.block-container{padding-top:.75rem; padding-bottom:.75rem; max-width:none !important; padding-left:1.25rem; padding-right:1.25rem;}

.app-header{background:var(--stk-header-bg); padding:18px 20px; border-radius:12px; margin:16px 0 16px;
  box-shadow:0 1px 0 rgba(0,0,0,.04) inset, 0 6px 20px rgba(0,0,0,.10);}
.header-inner{position:relative; height:48px; display:flex; align-items:center; justify-content:center;}
.stk-logo{position:absolute; left:16px; top:50%; transform:translateY(-50%); height:44px; width:auto;
  filter:drop-shadow(0 1px 0 rgba(0,0,0,.10));}
.app-header h1{margin:0; color:var(--stk-header-fg); font-weight:800; letter-spacing:.4px;}

.footer-note{background:var(--stk-note-bg); border:1px solid var(--stk-note-bd); border-radius:10px; padding:16px 18px;
  color:var(--stk-note-fg); text-align:center; margin:18px 0 8px; font-size:1.1rem; font-weight:600; width:100%;}

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

/* ===== Banner de cache (substitui st.info, melhora contraste) ===== */
.cache-banner{
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.22);
  color: #ffffff;
  padding: 12px 14px;
  border-radius: 12px;
  font-weight: 600;
  margin: 10px 0 12px;
}
.cache-banner b{ color:#FFD14F; }
</style>
""", unsafe_allow_html=True)

    # Header com logo
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
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='app-header'><div class='header-inner'><h1>IRR Real</h1></div></div>",
            unsafe_allow_html=True,
        )

    # ====== Tickers ======
    tickers_for_prices = [
        "CPLE3","IGTI3","IGTI4","ENGI3","ENGI4","ENGI11",
        "EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3",
        "MULT3","ALOS3","AXIA3","AXIA6","AXIA7",
    ]

    # ====== UX: mostra cache IMEDIATO (se existir) ======
    cached_prices, cached_meta = load_last_prices()
    has_cache = cached_prices is not None and cached_meta is not None

    colA, colB = st.columns([1.2, 4.8])
    with colA:
        refresh = st.button("🔄 Atualizar preços agora", use_container_width=True)
    with colB:
        st.caption("Se o Yahoo bloquear, o app não trava: mantém o último cache salvo.")

    # Default: usar cache
    if has_cache:
        prices_series = cached_prices.reindex(tickers_for_prices)
        meta = cached_meta.reindex(tickers_for_prices)

        # (AJUSTE) banner com contraste bom
        st.markdown(
            "<div class='cache-banner'>"
            "Mostrando <b>últimos preços salvos</b>. Clique em <b>“Atualizar preços agora”</b> para tentar pegar os mais recentes."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        prices_series = pd.Series(index=tickers_for_prices, dtype="float64", name="preco")
        meta = pd.DataFrame(index=tickers_for_prices, data={"Fonte": "N/A", "Timestamp": pd.NaT})

    # Se não tem cache, faz 1 tentativa automática. Se tem cache, só atualiza se clicar.
    need_fetch = refresh or (not has_cache)

    status_box = st.empty()
    if need_fetch:
        status_box.info("Baixando preços do Yahoo Finance (timeout ~10s)...")
        with st.spinner("Carregando..."):
            prices_new, meta_new, stt = fetch_prices_yahoo_safe(tickers_for_prices, max_total_seconds=10)

        if stt == "ok" and prices_new is not None and prices_new.notna().sum() > 0:
            prices_series = prices_new
            meta = meta_new
            save_last_prices(prices_series, meta)
            status_box.success("✅ Preços atualizados com sucesso (Yahoo).")
        else:
            msg = "⚠️ Não consegui atualizar pelo Yahoo agora."
            if stt == "timeout":
                msg += " (timeout)"
            elif stt == "probe_failed":
                msg += " (bloqueio/rate-limit provável)"
            else:
                msg += " (erro)"
            status_box.warning(msg + " Mantendo o último cache disponível.")

    # A partir daqui, o app sempre segue (cache ou novo)
    try:
        prices = pd.to_numeric(prices_series, errors="coerce").astype(float)

        # ====== Shares ======
        shares_classes = {
            "CPLE3": 2_982_810_590,
            "IGTI3": 770_992_429,
            "IGTI4": 435_368_756,
            "ENGI3": 887_231_247,
            "ENGI4": 1_402_193_416,
            "ENGI11": 502_800_000,
            "EQTL3": 1_255_510_000,
            "SBSP3": 683_510_000,
            "NEOE3": 1_213_800_000,
            "ENEV3": 1_936_970_000,
            "ELET3": 2_308_630_000,
            "EGIE3": 1_142_300_000,
            "MULT3": 513_164_000,
            "ALOS3": 542_937_000,
            "AXIA3": 2_028_500_000,
            "AXIA6": 279_000_000,
            "AXIA7": 606_750_000,
        }
        shares_series = pd.Series(shares_classes).reindex(prices.index)
        mc_raw = prices * shares_series

        # ====== Consolidações (ENGI11) ======
        THRESH_ENGI_MIN_IRR_PCT = 4.0

        engi11_price  = prices.get("ENGI11", np.nan)
        engi11_shares = shares_series.get("ENGI11", np.nan)
        cap_11 = (
            engi11_price * engi11_shares
            if (pd.notna(engi11_price) and pd.notna(engi11_shares) and engi11_price > 0)
            else np.nan
        )

        cap_34 = np.nan
        if {"ENGI3","ENGI4"}.issubset(mc_raw.index):
            cap_34 = mc_raw["ENGI3"] + mc_raw["ENGI4"]

        if pd.notna(cap_11):
            engi_total = cap_11
            engi_calc_source = "ENGI11 price × ENGI11 shares (cap_11)"
            engi_method_cap11 = True
        else:
            engi_total = cap_34 if pd.notna(cap_34) else np.nan
            engi_calc_source = "sem cap_11 (fallback apenas p/ tabela)"
            engi_method_cap11 = False

        # IGTI total (ticker sintético IGTI11)
        igti_total = mc_raw.get("IGTI3", np.nan) + mc_raw.get("IGTI4", np.nan)
        if pd.isna(igti_total):
            igti_total = np.nan

        # ====== Tabela final (para XIRR) ======
        final_tickers = [
            "CPLE3","EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3",
            "MULT3","ALOS3","AXIA6","IGTI11","ENGI11",
        ]

        rows = []
        for t in final_tickers:
            if t == "IGTI11":
                price = np.nan
                shares = shares_classes["IGTI3"] + shares_classes["IGTI4"]
                mc = igti_total

            elif t == "ENGI11":
                price = prices.get("ENGI11", np.nan)
                shares = shares_classes["ENGI11"]
                mc = engi_total

            elif t == "AXIA6":
                price_axia6 = prices.get("AXIA6", np.nan)
                shares_axia6 = shares_series.get("AXIA6", np.nan)
                price_axia3 = prices.get("AXIA3", np.nan)
                shares_axia3 = shares_series.get("AXIA3", np.nan)
                price_axia7 = prices.get("AXIA7", np.nan)
                shares_axia7 = shares_series.get("AXIA7", np.nan)

                price = price_axia6
                shares = (
                    shares_axia6 + shares_axia3 + shares_axia7
                    if all(pd.notna(v) for v in [shares_axia6, shares_axia3, shares_axia7])
                    else np.nan
                )

                parts = []
                if pd.notna(price_axia6) and pd.notna(shares_axia6):
                    parts.append(price_axia6 * shares_axia6)
                if pd.notna(price_axia3) and pd.notna(shares_axia3):
                    parts.append(price_axia3 * shares_axia3)
                if pd.notna(price_axia7) and pd.notna(shares_axia7):
                    parts.append(price_axia7 * shares_axia7)
                mc = sum(parts) if parts else np.nan

            else:
                price = prices.get(t, np.nan)
                shares = shares_series.get(t, np.nan)
                mc = price * shares if (pd.notna(price) and pd.notna(shares)) else np.nan

            rows.append({"ticker": t, "price": price, "shares": shares, "market_cap": mc})

        resultado = pd.DataFrame(rows).set_index("ticker")
        resultado["market_cap"] = resultado["market_cap"].apply(cap_to_first_digits_mln)

        # ====== Excel dos fluxos ======
        df = pd.read_excel("irrdash3.xlsx")
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        for t in resultado.index:
            if t not in df.columns:
                df[t] = np.nan

        target_row = df.index[0]
        today = datetime.now().date()
        for t in resultado.index:
            v = resultado.loc[t, "market_cap"]
            df.loc[target_row, t] = -abs(float(v)) if pd.notna(v) else np.nan

        for t in resultado.index:
            df[t] = pd.to_numeric(df[t], errors="coerce")

        # ====== XIRR ======
        irr_results = {}
        for t in resultado.index:
            series_cf = df[t].dropna()
            if series_cf.empty:
                irr_results[t] = np.nan
                continue
            cashflows = series_cf.values.astype(float).copy()
            n_periods = len(cashflows)
            dates_list = [today] + [date(today.year + j - 1, 12, 31) for j in range(1, n_periods)]
            irr_results[t] = compute_xirr(cashflows, dates_list)

        ytm_df = pd.DataFrame.from_dict(irr_results, orient="index", columns=["irr"])
        ytm_df["irr_aj"] = ytm_df["irr"]

        # Ajuste para IRR real (shopping / real estate, sem AXIA6)
        for t in ["MULT3","ALOS3","IGTI11"]:
            if t in ytm_df.index and not pd.isna(ytm_df.loc[t, "irr"]):
                ytm_df.loc[t, "irr_aj"] = ((1 + ytm_df.loc[t, "irr"]) / (1 + 0.045)) - 1

        ytm_clean = ytm_df[["irr_aj"]].dropna().sort_values("irr_aj", ascending=True)

        # ====== Regras de exibição no gráfico ======
        drop_list = ["ELET3", "ELET6"]

        if "ENGI11" in ytm_clean.index:
            engi_irr_pct = float(ytm_clean.loc["ENGI11", "irr_aj"] * 100.0)
        else:
            engi_irr_pct = np.nan

        show_engi11 = (engi_method_cap11 is True) and (pd.notna(engi_irr_pct) and engi_irr_pct >= THRESH_ENGI_MIN_IRR_PCT)
        if not show_engi11:
            drop_list.append("ENGI11")

        ytm_plot = ytm_clean[~ytm_clean.index.isin(drop_list)]

        # ====== Gráfico ======
        if len(ytm_plot) == 0:
            st.warning("Nenhum ticker disponível para o gráfico de IRR após os filtros.")
        else:
            plot_data = pd.DataFrame({
                "empresa": ytm_plot.index,
                "irr": (ytm_plot["irr_aj"] * 100).round(2),
            }).reset_index(drop=True)

            cor_ouro = "rgb(201,140,46)"
            bar_colors = [cor_ouro for _ in plot_data["empresa"]]

            fig = go.Figure(go.Bar(
                x=plot_data["empresa"], y=plot_data["irr"],
                text=[f"{v:.2f}%" for v in plot_data["irr"]],
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
            ))
            fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="white", size=14))

            irr_min = float(plot_data["irr"].min()) if len(plot_data) else 0.0
            irr_max = float(plot_data["irr"].max()) if len(plot_data) else 0.0
            ymin = min(4.0, irr_min - 1.0)
            ymax = max(12.0, irr_max * 1.10)

            fig.update_layout(
                bargap=0.12, plot_bgcolor="#0e314a", paper_bgcolor="#0e314a",
                uniformtext_minsize=10,
                font=dict(family="Inter, system-ui, sans-serif", color="white", size=14),
                xaxis=dict(title="Empresas", tickfont=dict(size=12, color="white"),
                           showgrid=False, showline=False, zeroline=False),
                yaxis=dict(title="IRR Real (%)", range=[ymin, ymax], dtick=1,
                           gridcolor="rgba(255,255,255,.12)", zeroline=False,
                           tickfont=dict(color="white")),
                margin=dict(l=10, r=10, t=6, b=62), showlegend=False, height=560,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ====== Duration ======
        duration_map = load_duration_map("irrdash3.xlsx", "duration").copy()

        def set_if_missing(label, value):
            if (label not in duration_map.index) or pd.isna(duration_map.loc[label]):
                duration_map.loc[label] = value

        if "IGTI11" in duration_map.index:
            v = duration_map.loc["IGTI11"]
            set_if_missing("IGTI3", v)
            set_if_missing("IGTI4", v)

        if "ENGI11" in duration_map.index:
            v = duration_map.loc["ENGI11"]
            if pd.notna(v):
                set_if_missing("ENGI3", v)
                set_if_missing("ENGI4", v)

        # ====== Tabela de preços + Duration ======
        order = [
            "CPLE3","IGTI3","IGTI4","ENGI3","ENGI4","ENGI11",
            "EQTL3","SBSP3","NEOE3","ENEV3","ELET3","EGIE3",
            "MULT3","ALOS3","AXIA3","AXIA6","AXIA7"
        ]

        tbl = pd.DataFrame({"Preço": prices.reindex(order)})
        tbl["Fonte"] = meta["Fonte"].reindex(order) if "Fonte" in meta.columns else "N/A"
        tbl["Timestamp"] = meta["Timestamp"].reindex(order).map(format_ts_brt) if "Timestamp" in meta.columns else ""
        tbl = tbl.rename_axis("Ticker").reset_index()
        tbl["Duration"] = tbl["Ticker"].map(duration_map)
        tbl["__dur_num"] = pd.to_numeric(tbl["Duration"], errors="coerce")
        tbl = tbl.sort_values(by="__dur_num", ascending=False, na_position="last").drop(columns="__dur_num")

        st.markdown(build_price_table_html(tbl), unsafe_allow_html=True)

        engi_status = "ENGI11 exibida (cap_11 e IRR ≥ 4%)" if show_engi11 else "ENGI11 ocultada (sem cap_11 ou IRR < 4%)"
        st.markdown(
            "<div class='footer-note'>💡 Se o Yahoo bloquear, o app mantém o último cache salvo. Use “Atualizar preços” para tentar novamente.</div>",
            unsafe_allow_html=True
        )
        st.caption(f"ENGI total calculado via: {engi_calc_source} • {engi_status}")

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

if __name__ == "__main__":
    main()

















