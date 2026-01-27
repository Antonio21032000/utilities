import os
import time
import random
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------
# UX: st.status (novo) com fallback p/ st.spinner (antigo)
# ------------------------------
@contextmanager
def status_or_spinner(label: str):
    try:
        with st.status(label, expanded=False) as s:
            yield s
    except Exception:
        with st.spinner(label):
            yield None

# ------------------------------
# yfinance: reduzir erros de timezone cache (race condition no cloud)
# ------------------------------
def _init_yf_cache_dir():
    cache_dir = os.environ.get("YF_TZ_CACHE", "/tmp/py-yfinance-tz")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # disponível em versões recentes
        yf.set_tz_cache_location(cache_dir)
    except Exception:
        pass

# ------------------------------
# Sessão HTTP com retry (429/5xx) + pool maior
# ------------------------------
def _build_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
        "Connection": "keep-alive",
    })

    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=30, pool_maxsize=30)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _normalize_b3(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return t
    if t.endswith(".SA"):
        return t
    return f"{t}.SA"

# ------------------------------
# Download em LOTE (1 chamada), SEM threads
# ------------------------------
def _download_batch_last_close(tickers: list[str], session: requests.Session) -> tuple[dict, dict]:
    """
    Retorna:
      prices[ticker] = float|None
      stamps[ticker] = pd.Timestamp|None
    """
    tickers = [_normalize_b3(t) for t in tickers if t and str(t).strip()]
    prices, stamps = {}, {}

    if not tickers:
        return prices, stamps

    # yfinance: caminho mais "barato" é pegar diário recente
    # (intraday aumenta chance de rate-limit)
    tick_str = " ".join(tickers)

    df = yf.download(
        tick_str,
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=False,       # MUITO importante no Streamlit Cloud
        progress=False,
        session=session,     # usa a sessão com retry
    )

    # MultiIndex quando vem mais de 1 ticker
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                if (t, "Close") in df.columns:
                    s = df[(t, "Close")].dropna()
                    prices[t] = float(s.iloc[-1]) if len(s) else None
                    stamps[t] = s.index[-1] if len(s) else None
                else:
                    prices[t], stamps[t] = None, None
            except Exception:
                prices[t], stamps[t] = None, None
    else:
        # 1 ticker só
        s = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
        t = tickers[0]
        prices[t] = float(s.iloc[-1]) if len(s) else None
        stamps[t] = s.index[-1] if len(s) else None

    return prices, stamps

# ------------------------------
# Cache: evita martelar o Yahoo a cada rerun
# ------------------------------
@st.cache_data(ttl=300, show_spinner=False)  # 5 minutos
def fetch_prices_yahoo_cached(tickers: tuple[str, ...]) -> tuple[dict, dict, str]:
    _init_yf_cache_dir()
    session = _build_requests_session()

    # 3 tentativas com backoff (caso venha HTML/vazio)
    last_err = ""
    for attempt in range(3):
        try:
            prices, stamps = _download_batch_last_close(list(tickers), session=session)

            # se ao menos 1 preço veio, consideramos sucesso parcial
            if any(v is not None for v in prices.values()):
                return prices, stamps, ""

            last_err = "Sem preços válidos (provável rate-limit/bloqueio)."
        except Exception as e:
            last_err = repr(e)

        # backoff + jitter
        time.sleep((1.2 * (2 ** attempt)) + random.random())

    return {}, {}, last_err

# ------------------------------
# Helper p/ manter última leitura boa na tela (UX)
# ------------------------------
def get_prices_with_session_state(tickers: list[str]) -> tuple[dict, dict, str]:
    prices, stamps, err = fetch_prices_yahoo_cached(tuple(tickers))

    if prices and any(v is not None for v in prices.values()):
        st.session_state["last_prices"] = prices
        st.session_state["last_stamps"] = stamps
        st.session_state["last_err"] = ""
        return prices, stamps, ""
    else:
        # fallback: mostra último resultado bom em vez de tela vazia
        lp = st.session_state.get("last_prices", {})
        ls = st.session_state.get("last_stamps", {})
        le = st.session_state.get("last_err", err or "Falha ao buscar preços.")
        return lp, ls, (err or le)

# ------------------------------
# EXEMPLO de uso no seu main()
# ------------------------------
def build_price_table(tickers_raw: list[str], durations_map: dict[str, float]) -> pd.DataFrame:
    prices, stamps, err = get_prices_with_session_state(tickers_raw)

    rows = []
    for raw in tickers_raw:
        t_yf = _normalize_b3(raw)
        p = prices.get(t_yf)
        ts = stamps.get(t_yf)

        rows.append({
            "Ticker": raw,
            "Preço": None if p is None else float(p),
            "Fonte": "Yahoo Finance",
            "Timestamp": "" if ts is None or pd.isna(ts) else pd.to_datetime(ts).strftime("%Y-%m-%d"),
            "Duration": durations_map.get(raw),
        })

    df = pd.DataFrame(rows)

    if err:
        st.warning(f"Yahoo/yfinance falhou agora (provável rate-limit/bloqueio). "
                   f"Estou exibindo o último resultado bom em cache. Detalhe: {err}")

    return df

def main():
    st.set_page_config(page_title="IRR Real", layout="wide")

    # Botão p/ forçar refresh (limpa cache)
    if st.button("Atualizar"):
        st.cache_data.clear()

    # Exemplo: seus tickers (substitua pela sua lista)
    tickers_raw = ["ENGI11", "NEOE3", "EQTL3", "SBSP3", "ALOS3", "MULT3"]

    # Exemplo: durations (substitua pelo seu dicionário)
    durations_map = {"ENGI11": 14.43, "NEOE3": 12.53, "EQTL3": 12.52, "SBSP3": 12.16, "ALOS3": 11.20, "MULT3": 11.01}

    with status_or_spinner("Baixando preços do Yahoo Finance (com retry + cache)..."):
        df_prices = build_price_table(tickers_raw, durations_map)

    st.subheader("Preços usados (Yahoo Finance)")
    st.dataframe(df_prices, use_container_width=True)

if __name__ == "__main__":
    main()

















