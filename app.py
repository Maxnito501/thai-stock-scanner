import math
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ==========================================================
# Thai Short-Term Stock Scanner (single file)
# - Uses Yahoo Finance via yfinance (free)
# - Symbols source (recommended): upload CSV/XLSX OR paste symbols
# - Optional: try fetch from web (may be blocked sometimes)
# ==========================================================

st.set_page_config(page_title="Thai Short-Term Stock Scanner", layout="wide")
st.title("Thai Short-Term Stock Scanner (Yahoo Finance - Free)")

st.caption(
    "เล่นสั้น: BUY/SELL/WAIT + TP/SL/Trailing ตามเทรนด์ | "
    "เทรนด์ดี TP10% SL5% (เริ่ม trailing +5%) | "
    "เทรนด์กลาง TP7% SL4% (เริ่ม trailing +4%) | "
    "เทรนด์อ่อน TP5% SL3% (เริ่ม trailing +3%) | "
    "ถึง TP → แนะนำขาย 50% แล้ว Trailing ที่เหลือ"
)

with st.expander("วิธีใช้ (สรุป)", expanded=False):
    st.markdown(
        """
1) **แนะนำที่สุด:** อัปโหลดไฟล์ `symbols.csv` หรือ `symbols.xlsx` ที่มีคอลัมน์ `Symbol` (เช่น PTT, AOT, SCC)
2) หรือ **วางรายชื่อหุ้น** ในกล่องด้านซ้าย (1 บรรทัดต่อ 1 ตัว)
3) กด **Run Scan**
4) ถ้ามีหุ้นในพอร์ต: ใส่พอร์ตในกล่อง `Ticker,Cost,Qty` เช่น `PTT.BK,34.50,1000`

> หมายเหตุ: ถ้าดึงเว็บรายชื่อหุ้นแล้วโดน 403 ให้ใช้อัปโหลด/วางรายชื่อแทน จะไม่โดนบล็อกแน่นอน
"""
    )


# =========================
# Indicators
# =========================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# =========================
# Trend + Risk/Reward rules (ตามที่ตกลง)
# =========================
def trend_score(last: pd.Series) -> int:
    score = 0
    score += 1 if last["EMA20"] > last["EMA50"] else 0
    score += 1 if last["Close"] > last["EMA20"] else 0
    score += 1 if last["RSI14"] >= 50 else 0
    score += 1 if last["MACD"] > last["MACDsig"] else 0
    return int(score)


def rr_by_trend(score: int) -> dict:
    if score >= 3:   # เทรนด์ดี
        return {"zone": "TREND_GOOD", "tp": 0.10, "sl": 0.05, "trail_start": 0.05}
    if score == 2:   # เทรนด์กลาง
        return {"zone": "TREND_MID", "tp": 0.07, "sl": 0.04, "trail_start": 0.04}
    # เทรนด์อ่อน/ไม่ชัด
    return {"zone": "TREND_WEAK", "tp": 0.05, "sl": 0.03, "trail_start": 0.03}


def trailing_from_peak(entry: float, peak: float, rr: dict):
    start = entry * (1 + rr["trail_start"])
    if peak < start:
        return None
    return peak * (1 - rr["sl"])


# =========================
# Symbols loaders
# =========================
def load_symbols_from_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)  # openpyxl used automatically on cloud
    else:
        raise RuntimeError("รองรับเฉพาะ .csv หรือ .xlsx")

    sym_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["symbol", "ticker", "security symbol"]:
            sym_col = c
            break
    if sym_col is None:
        sym_col = next((c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
    if sym_col is None:
        raise RuntimeError("ไฟล์ไม่มีคอลัมน์ Symbol/Ticker")

    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.strip()})
    out = out[out["Symbol"].str.match(r"^[A-Z0-9\.\-]+$")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def load_symbols_from_paste(text: str) -> pd.DataFrame:
    lines = []
    for line in (text or "").splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        # allow "PTT", or "PTT.BK" (we'll normalize later)
        s = s.replace(" ", "")
        lines.append(s)
    if not lines:
        return pd.DataFrame(columns=["Symbol"])
    df = pd.DataFrame({"Symbol": lines})
    df["Symbol"] = df["Symbol"].str.replace(".BK", "", regex=False)
    df = df[df["Symbol"].str.match(r"^[A-Z0-9\.\-]+$")]
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_data(ttl=24 * 60 * 60)
def load_symbols_from_web_stockanalysis() -> pd.DataFrame:
    """
    Optional web fetch (may be blocked by 403 on some networks).
    Uses requests+UA then pd.read_html(StringIO(html)) — NOT pd.read_html(url).
    """
    url = "https://stockanalysis.com/list/stock-exchange-of-thailand/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "symbol" in cols:
            df = t.copy()
            break
    if df is None:
        raise RuntimeError("ไม่พบตาราง Symbol จากเว็บ")
    sym_col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), df.columns[0])
    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.strip()})
    out = out[out["Symbol"].str.match(r"^[A-Z0-9\.\-]+$")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


# =========================
# Yahoo validation + scanning
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def validate_yahoo_symbols(symbols: list[str], max_check: int = 250) -> list[str]:
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


def compute_signal_short(df: pd.DataFrame) -> dict:
    df = df.dropna().copy()
    if len(df) < 80:
        raise ValueError("Not enough data")

    close = df["Close"]
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI14"] = rsi(close, 14)
    df["MACD"], df["MACDsig"] = macd(close)

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
        "MACD": float(last["MACD"]) if not math.isnan(last["MACD"]) else np.nan,
        "MACDsig": float(last["MACDsig"]) if not math.isnan(last["MACDsig"]) else np.nan,
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
        t = parts[0].strip().upper()
        if not t.endswith(".BK"):
            t = t + ".BK"
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
# Sidebar UI
# =========================
with st.sidebar:
    st.header("1) รายชื่อหุ้น (Symbols)")
    st.caption("แนะนำ: อัปโหลดไฟล์ CSV/XLSX (มีคอลัมน์ Symbol) เพื่อเลี่ยง 403")
    uploaded = st.file_uploader("อัปโหลด symbols.csv / symbols.xlsx", type=["csv", "xlsx"])

    st.caption("หรือวางสัญลักษณ์ 1 บรรทัดต่อ 1 ตัว (เช่น PTT, AOT, SCC)")
    paste_symbols = st.text_area("Paste symbols", height=140, placeholder="PTT\nAOT\nSCC\nCPALL\nBDMS")

    use_web = st.checkbox("ลองดึงรายชื่อหุ้นจากเว็บ (อาจโดน 403 ได้)", value=False)

    st.divider()
    st.header("2) ตั้งค่าสแกน")
    validate_n = st.slider("Validate Yahoo (.BK)", 20, 800, 200, 20)
    scan_n = st.slider("จำนวนหุ้นที่จะสแกนจริง", 10, 500, 100, 10)
    period = st.selectbox("ช่วงข้อมูลที่ใช้คำนวณ", ["6mo", "1y", "2y"], index=1)

    st.divider()
    st.header("3) My Portfolio (optional)")
    st.caption("บรรทัดละตัว: Ticker,Cost,Qty  (Ticker แบบ PTT.BK)")
    portfolio_text = st.text_area("Portfolio", height=120, placeholder="PTT.BK,34.50,1000\nAOT.BK,62.00,200")

    st.divider()
    run = st.button("Run Scan", type="primary")


# =========================
# Main
# =========================
if run:
    # Load symbols with priority: upload -> paste -> web (optional)
    with st.spinner("กำลังเตรียมรายชื่อหุ้น..."):
        sym_df = pd.DataFrame(columns=["Symbol"])
        source = ""

        if uploaded is not None:
            sym_df = load_symbols_from_upload(uploaded)
            source = f"ไฟล์อัปโหลด ({len(sym_df):,} symbols)"
        elif paste_symbols.strip():
            sym_df = load_symbols_from_paste(paste_symbols)
            source = f"ข้อความที่วาง ({len(sym_df):,} symbols)"
        elif use_web:
            try:
                sym_df = load_symbols_from_web_stockanalysis()
                source = f"เว็บ StockAnalysis ({len(sym_df):,} symbols)"
            except Exception as e:
                st.error("ดึงรายชื่อจากเว็บไม่สำเร็จ (มักเป็น 403) → แนะนำให้อัปโหลด/วางรายชื่อแทน")
                st.stop()
        else:
            st.error("ยังไม่มีรายชื่อหุ้น: กรุณาอัปโหลดไฟล์ หรือวางรายชื่อหุ้นในช่อง Paste symbols")
            st.stop()

    st.success(f"แหล่งรายชื่อหุ้น: {source}")

    base_symbols = sym_df["Symbol"].astype(str).str.upper().str.replace(".BK", "", regex=False).tolist()

    with st.spinner("กำลัง validate ว่าหุ้นมีข้อมูลใน Yahoo (.BK) ..."):
        ok_tickers = validate_yahoo_symbols(base_symbols, max_check=validate_n)

    st.write(f"ผ่าน validate: **{len(ok_tickers):,}** ตัว | จะสแกน: **{min(scan_n, len(ok_tickers)):,}** ตัว")

    if len(ok_tickers) == 0:
        st.error("ไม่ผ่าน validate เลย → ลด validate_n / ลองใหม่ / เปลี่ยนเน็ต หรือใช้รายชื่อหุ้นที่แน่ใจว่ามีใน Yahoo")
        st.stop()

    tickers = ok_tickers[:scan_n]

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

                sig = compute_signal_short(df)
                results.append({"Ticker": t, **sig})
            except Exception:
                continue

        done += 1
        prog.progress(min(1.0, done / total_chunks))
        time.sleep(0.05)

    prog.empty()

    out = pd.DataFrame(results)
    if len(out) == 0:
        st.error("สแกนไม่สำเร็จ (อาจโดน rate limit) → ลด scan_n / ลด validate_n / ใช้ period สั้นลง")
        st.stop()

    # Ranking
    order_map = {"BUY": 0, "WAIT": 1, "SELL": 2}
    out["Order"] = out["Action"].map(order_map).fillna(9).astype(int)
    out = out.sort_values(["Order", "Score"], ascending=[True, False]).drop(columns=["Order"])

    # No position advice
    st.subheader("คำแนะนำกรณี 'ไม่มีของ' (No Position)")
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY", int((out["Action"] == "BUY").sum()))
    c2.metric("WAIT", int((out["Action"] == "WAIT").sum()))
    c3.metric("SELL", int((out["Action"] == "SELL").sum()))
    st.write(
        "- ถ้า **BUY น้อย/ไม่มี**: ไม่ต้องฝืนเทรด → เก็บ Watchlist จาก WAIT ที่คะแนนสูง ๆ แล้วรอ\n"
        "- ถ้า **BUY มี**: เข้าได้ตามระบบ พร้อม TP/SL ตาม TrendZone\n"
        "- ถ้า **SELL**: ไม่เข้า รอจนกลับเป็น BUY"
    )

    # Tables
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
        st.subheader("คำแนะนำสำหรับ 'ของที่ถืออยู่' (เล่นสั้น)")
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
                    "Suggestion": "เพิ่ม scan_n/validate_n หรือเช็คว่า ticker ถูกต้อง (.BK)"
                })
                continue

            cur = float(r["Close"])
            pnl = (cur - cost) / cost

            zone = str(r.get("TrendZone", "TREND_WEAK"))
            rr = rr_by_trend(3) if zone == "TREND_GOOD" else rr_by_trend(2) if zone == "TREND_MID" else rr_by_trend(0)

            tp_price = cost * (1 + rr["tp"])
            sl_price = cost * (1 - rr["sl"])

            peak = float(r.get("Peak60", cur))
            trail_price = trailing_from_peak(cost, peak, rr)
            sig_now = str(r.get("Action", "WAIT"))

            if cur <= sl_price:
                status = "HIT SL"
                sug = "ถึงจุดตัดขาดทุน → แนะนำ CUT"
            elif cur >= tp_price:
                status = "HIT TP"
                sell_qty = qty * 0.5 if qty > 0 else None
                sug = "ถึงเป้ากำไร → แนะนำขาย 50% แล้วปล่อยที่เหลือด้วย Trailing"
                if sell_qty is not None:
                    sug += f" (ขาย ~{sell_qty:.0f} หุ้น)"
                if trail_price is not None:
                    sug += f" | Trailing≈{trail_price:.2f} (หลุดแล้วปิดที่เหลือ)"
            else:
                if pnl < 0 and sig_now == "SELL":
                    status = "LOSS + SELL"
                    sug = "ติดลบ + สัญญาณ SELL → ลดความเสี่ยง/รอเด้งออก (ถ้ายังไม่ถึง SL)"
                elif pnl < 0:
                    status = "LOSS"
                    sug = "ยังไม่ถึง SL → ถือรอตามระบบ (ห้ามเฉลี่ยลงจนกว่าจะกลับเป็น BUY ชัดเจน)"
                elif pnl >= 0 and sig_now == "SELL":
                    status = "PROFIT + SELL"
                    sug = "กำไรอยู่แต่ SELL → แนะนำล็อกกำไร/ทยอยปิด (อย่างน้อย 50%)"
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
                "SignalNow": sig_now,
                "TP_Price": round(tp_price, 4),
                "SL_Price": round(sl_price, 4),
                "Trailing(if active)": None if trail_price is None else round(trail_price, 4),
                "Status": status,
                "Suggestion": sug,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.markdown(
        """
### Deploy (GitHub → Streamlit Cloud)
- ใส่ไฟล์ `app.py` นี้ + `requirements.txt` (ดูด้านล่าง)
- Streamlit Cloud → New app → เลือก repo → เลือก `app.py` → Deploy

### requirements.txt
```
streamlit
yfinance
pandas
numpy
requests
lxml
openpyxl
```
"""
    )
