import math
import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Thai Short-Term Stock Scanner", layout="wide")
st.title("Thai Short-Term Stock Scanner (Yahoo Finance - Free)")

st.caption(
    "โหมดเล่นสั้น: BUY/SELL/WAIT + TP/SL/Trailing ตามเทรนด์ "
    "(เทรนด์ดี TP10 SL5 | เทรนด์กลาง TP7 SL4 | เทรนด์อ่อน TP5 SL3) "
    "ถึง TP → ขาย 50% แล้วปล่อยที่เหลือด้วย Trailing"
)

# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def atr(df: pd.DataFrame, period: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# Trend / risk-reward (ตามที่สรุป)
# -----------------------------
def trend_score(last: pd.Series) -> int:
    score = 0
    score += 1 if last["EMA20"] > last["EMA50"] else 0
    score += 1 if last["Close"] > last["EMA20"] else 0
    score += 1 if last["RSI14"] >= 50 else 0
    score += 1 if last["MACD"] > last["MACDsig"] else 0
    return int(score)

def rr_by_trend(score: int) -> dict:
    if score >= 3:
        return {"zone": "TREND_GOOD", "tp": 0.10, "sl": 0.05, "trail_start": 0.05}
    elif score == 2:
        return {"zone": "TREND_MID", "tp": 0.07, "sl": 0.04, "trail_start": 0.04}
    else:
        return {"zone": "TREND_WEAK", "tp": 0.05, "sl": 0.03, "trail_start": 0.03}

def trailing_from_peak(entry: float, peak: float, rr: dict):
    start = entry * (1 + rr["trail_start"])
    if peak < start:
        return None
    return peak * (1 - rr["sl"])

# -----------------------------
# Load symbols (OFFICIAL SET xls first)
# -----------------------------
@st.cache_data(ttl=24 * 60 * 60)
def load_set_symbols() -> pd.DataFrame:
    """
    ดึงรายชื่อหุ้นจากไฟล์ทางการ SET (xls) เพื่อลดปัญหา 403
    """
    url = "https://www.set.or.th/dat/eod/listedcompany/static/listedCompanies_en_US.xls"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_excel(BytesIO(r.content), engine="xlrd")
    # หา column ที่เป็น Symbol แบบยืดหยุ่น
    sym_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["symbol", "security symbol", "ticker", "ticker symbol"]:
            sym_col = c
            break
    if sym_col is None:
        sym_col = next((c for c in df.columns if "symbol" in str(c).lower()), None)
    if sym_col is None:
        raise RuntimeError("อ่านไฟล์ SET ได้ แต่หา column Symbol ไม่เจอ")

    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.strip()})
    out = out[out["Symbol"].str.match(r"^[A-Z0-9\.\-]+$")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out

@st.cache_data(ttl=24 * 60 * 60)
def validate_yahoo(symbols: list[str], max_check: int = 300) -> list[str]:
    ok = []
    for s in symbols[:max_check]:
        t = f"{s}.BK"
        try:
            hist = yf.download(t, period="5d", interval="1d", progress=False, threads=False)
            if hist is not None and len(hist) >= 1 and not hist["Close"].isna().all():
                ok.append(t)
        except Exception:
            pass
        time.sleep(0.02)
    return ok

# -----------------------------
# Signal short
# -----------------------------
def signal_short(df: pd.DataFrame) -> dict:
    df = df.copy()
    close = df["Close"]

    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI14"] = rsi(close, 14)
    df["MACD"], df["MACDsig"] = macd(close)
    df["ATR14"] = atr(df, 14)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    trend_up = (last["EMA20"] > last["EMA50"]) and (last["Close"] > last["EMA20"])
    macd_up = (prev["MACD"] <= prev["MACDsig"]) and (last["MACD"] > last["MACDsig"])
    macd_down = (prev["MACD"] >= prev["MACDsig"]) and (last["MACD"] < last["MACDsig"])
    rsi_buy = (prev["RSI14"] <= 40) and (last["RSI14"] > 40)
    rsi_over = (last["RSI14"] >= 70)

    score_sig = 0
    score_sig += 2 if trend_up else -1
    score_sig += 2 if macd_up else 0
    score_sig += 1 if rsi_buy else 0
    score_sig += -2 if (macd_down or rsi_over) else 0

    if score_sig >= 3:
        action = "BUY"
        reason = "Trend up + Momentum up"
    elif score_sig <= -2:
        action = "SELL"
        reason = "Momentum weak / Overbought"
    else:
        action = "WAIT"
        reason = "No clear edge"

    tscore = trend_score(last)
    rr = rr_by_trend(tscore)

    peak60 = float(close.tail(60).max())

    return {
        "Action": action,
        "Reason": reason,
        "Score": int(score_sig),
        "Close": float(last["Close"]),
        "RSI14": float(last["RSI14"]) if not math.isnan(last["RSI14"]) else np.nan,
        "EMA20": float(last["EMA20"]) if not math.isnan(last["EMA20"]) else np.nan,
        "EMA50": float(last["EMA50"]) if not math.isnan(last["EMA50"]) else np.nan,
        "TrendScore": int(tscore),
        "TrendZone": rr["zone"],
        "TP%": rr["tp"] * 100,
        "SL%": rr["sl"] * 100,
        "TrailStart%": rr["trail_start"] * 100,
        "Peak60": peak60,
    }

def parse_portfolio(text: str) -> pd.DataFrame:
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("ticker"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        t = parts[0]
        try:
            cost = float(parts[1])
        except Exception:
            continue
        qty = 0.0
        if len(parts) >= 3:
            try:
                qty = float(parts[2])
            except Exception:
                qty = 0.0
        rows.append({"Ticker": t, "Cost": cost, "Qty": qty})
    return pd.DataFrame(rows)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Settings")
    validate_n = st.slider("จำนวนที่จะ validate (.BK)", 50, 800, 250, 50)
    scan_n = st.slider("จำนวนหุ้นที่จะสแกน", 20, 500, 100, 10)
    period = st.selectbox("ช่วงข้อมูล", ["6mo", "1y", "2y"], index=1)
    st.divider()
    st.subheader("My Portfolio (optional)")
    st.caption("บรรทัดละตัว: Ticker,Cost,Qty  (Ticker แบบ PTT.BK)")
    portfolio_text = st.text_area("ตัวอย่าง:\nPTT.BK,34.50,1000\nAOT.BK,62.00,200", height=120)
    run = st.button("Run Scan", type="primary")

# =========================
# Run
# =========================
if run:
    with st.spinner("โหลดรายชื่อหุ้นจาก SET (ทางการ)..."):
        sym_df = load_set_symbols()

    st.write(f"รายชื่อจาก SET: **{len(sym_df):,}** ตัว")

    base_symbols = sym_df["Symbol"].tolist()

    with st.spinner("validate Yahoo (.BK) ..."):
        ok = validate_yahoo(base_symbols, max_check=validate_n)

    st.write(f"ผ่าน validate: **{len(ok):,}** ตัว | จะสแกน: **{min(scan_n, len(ok)):,}** ตัว")

    if len(ok) == 0:
        st.error("ไม่ผ่าน validate เลย (อาจโดนจำกัดจาก Yahoo หรือเน็ต) ลองลด validate/เปลี่ยนเน็ต")
        st.stop()

    tickers = ok[:scan_n]

    results = []
    chunk_size = 80
    total_chunks = max(1, (len(tickers) + chunk_size - 1) // chunk_size)
    prog = st.progress(0)
    done = 0

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]

        try:
            data = yf.download(
                " ".join(chunk),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            data = None

        for t in chunk:
            try:
                if data is None:
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[t].dropna()
                else:
                    df = data.dropna()

                if df is None or len(df) < 80:
                    continue

                sig = signal_short(df)
                results.append({"Ticker": t, **sig})
            except Exception:
                continue

        done += 1
        prog.progress(min(1.0, done / total_chunks))
        time.sleep(0.05)

    prog.empty()

    out = pd.DataFrame(results)
    if len(out) == 0:
        st.error("สแกนไม่สำเร็จ (อาจโดน rate limit) ลองลดจำนวนหุ้นที่จะสแกน")
        st.stop()

    # Ranking
    order_map = {"BUY": 0, "WAIT": 1, "SELL": 2}
    out["Order"] = out["Action"].map(order_map).fillna(9).astype(int)
    out = out.sort_values(["Order", "Score"], ascending=[True, False]).drop(columns=["Order"])

    # No position advice
    st.subheader("คำแนะนำกรณี 'ไม่มีของ' (No Position)")
    b1, b2, b3 = st.columns(3)
    b1.metric("BUY", int((out["Action"] == "BUY").sum()))
    b2.metric("WAIT", int((out["Action"] == "WAIT").sum()))
    b3.metric("SELL", int((out["Action"] == "SELL").sum()))
    st.write(
        "- ถ้า BUY น้อย/ไม่มี: **ไม่ต้องฝืนเทรด** → เก็บ Watchlist จาก WAIT ที่คะแนนสูง ๆ แล้วรอ\n"
        "- ถ้า BUY มี: เข้าได้ตามระบบ พร้อม TP/SL ตาม TrendZone\n"
        "- ถ้า SELL: ไม่เข้า รอจนกลับเป็น BUY"
    )

    # Top tables
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top BUY")
        st.dataframe(out[out["Action"] == "BUY"].head(30), use_container_width=True)
    with colB:
        st.subheader("Top SELL / ระวัง")
        st.dataframe(out[out["Action"] == "SELL"].head(30), use_container_width=True)

    st.subheader("ทั้งหมด")
    st.dataframe(out, use_container_width=True)

    # Portfolio advice
    port = parse_portfolio(portfolio_text)
    if len(port) > 0:
        st.subheader("คำแนะนำสำหรับ 'ของที่ถืออยู่'")
        merged = port.merge(out, on="Ticker", how="left")

        rows = []
        for _, r in merged.iterrows():
            t = r["Ticker"]
            cost = float(r["Cost"])
            qty = float(r.get("Qty", 0.0) or 0.0)

            if pd.isna(r.get("Close", np.nan)):
                rows.append({
                    "Ticker": t, "Cost": cost, "Qty": qty,
                    "Status": "ไม่พบในผลสแกน",
                    "Suggestion": "เพิ่มจำนวนสแกน หรือเช็ค ticker ต้องลงท้าย .BK"
                })
                continue

            cur = float(r["Close"])
            pnl = (cur - cost) / cost

            zone = str(r.get("TrendZone", "TREND_WEAK"))
            rr = rr_by_trend(3) if zone == "TREND_GOOD" else rr_by_trend(2) if zone == "TREND_MID" else rr_by_trend(0)

            tp = cost * (1 + rr["tp"])
            sl = cost * (1 - rr["sl"])
            peak = float(r.get("Peak60", cur))
            trail = trailing_from_peak(cost, peak, rr)

            action = str(r.get("Action", "WAIT"))

            if cur <= sl:
                status = "HIT SL"
                sug = "ถึงจุดตัดขาดทุน → CUT"
            elif cur >= tp:
                status = "HIT TP"
                sell_qty = qty * 0.5 if qty > 0 else None
                sug = "ถึง TP → ขาย 50% แล้วใช้ Trailing กับที่เหลือ"
                if sell_qty is not None:
                    sug += f" (ขาย ~{sell_qty:.0f} หุ้น)"
                if trail is not None:
                    sug += f" | Trailing≈{trail:.2f}"
            else:
                if pnl < 0 and action == "SELL":
                    status = "LOSS + SELL"
                    sug = "ติดลบ + SELL → ลดความเสี่ยง/รอเด้งออก (ถ้ายังไม่ถึง SL)"
                elif pnl < 0:
                    status = "LOSS"
                    sug = "ยังไม่ถึง SL → ถือรอตามระบบ (ห้ามเฉลี่ยลงจนกว่าจะกลับเป็น BUY ชัดเจน)"
                elif pnl >= 0 and action == "SELL":
                    status = "PROFIT + SELL"
                    sug = "กำไรอยู่แต่ SELL → ล็อกกำไร/ทยอยปิด (อย่างน้อย 50%)"
                else:
                    status = "PROFIT"
                    sug = "กำไรอยู่ → ถือและเลื่อน stop ตาม / รอถึง TP หรือ trailing"

            rows.append({
                "Ticker": t,
                "Cost": round(cost, 4),
                "Qty": qty,
                "Current": round(cur, 4),
                "PnL%": round(pnl * 100, 2),
                "TrendZone": zone,
                "SignalNow": action,
                "TP_Price": round(tp, 4),
                "SL_Price": round(sl, 4),
                "Trailing(if active)": None if trail is None else round(trail, 4),
                "Status": status,
                "Suggestion": sug,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.markdown(
    """
### วิธี Deploy บน Streamlit Cloud
1) เข้า https://share.streamlit.io หรือ Streamlit Community Cloud  
2) Login ด้วย GitHub  
3) กด **New app**  
4) เลือก repo ของคุณ → เลือก branch → เลือกไฟล์ `app.py`  
5) กด Deploy  

แก้ไขโค้ด: แก้ที่ GitHub → commit → Streamlit จะ redeploy อัตโนมัติ (หรือกด rerun)
"""
)
