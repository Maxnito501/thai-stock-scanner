import math
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Thai Short-Term Stock Scanner", layout="wide")
st.title("Thai Short-Term Stock Scanner (Yahoo Finance - Free)")
st.caption(
    "สแกนหุ้นไทยโหมดเล่นสั้น: BUY/SELL/WAIT + แผน TP/SL/Trailing ตามเทรนด์ "
    "(เทรนด์ดี TP10 SL5 | เทรนด์กลาง TP7 SL4 | เทรนด์อ่อน TP5 SL3) | ถึง TP → ขาย 50% แล้ว Trailing ที่เหลือ"
)

with st.expander("ข้อจำกัดและวิธีแก้", expanded=False):
    st.write(
        "- Yahoo ผ่าน yfinance เป็นข้อมูลฟรี อาจช้า/โดนจำกัดการเรียกเป็นช่วง ๆ\n"
        "- ถ้าเว็บรายชื่อหุ้นโดนบล็อก ให้ใช้อัปโหลดไฟล์รายชื่อหุ้น (CSV/XLSX) จะเสถียรที่สุด\n"
        "- ถ้าสแกนแล้วช้า: ลดจำนวนสแกน/ลด validate/เลือก period สั้นลง\n"
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
    # EMA20/EMA50 + Close>EMA20 + RSI>=50 + MACD>Signal
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
    # เริ่ม trailing เมื่อ peak >= entry*(1+trail_start)
    start = entry * (1 + rr["trail_start"])
    if peak < start:
        return None
    # เลื่อน stop ตาม peak ด้วยระยะ sl%
    return peak * (1 - rr["sl"])


# =========================
# Symbols sources
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def load_symbols_from_stockanalysis() -> pd.DataFrame:
    """
    ดึงรายชื่อหุ้นไทยจาก StockAnalysis แบบกัน 403:
    ใช้ requests + User-Agent แล้วค่อย feed HTML ให้ pd.read_html
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
        raise RuntimeError("ไม่พบตาราง Symbol จาก StockAnalysis")

    sym_col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), df.columns[0])
    out = pd.DataFrame({"Symbol": df[sym_col].astype(str).str.strip()})
    out = out[out["Symbol"].str.match(r"^[A-Z0-9\.\-]+$")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def load_symbols_from_upload(uploaded_file) -> pd.DataFrame:
    """
    รองรับ CSV/XLSX ที่ผู้ใช้อัปโหลดเอง (เสถียรสุด)
    ต้องมีคอลัมน์ Symbol หรือ Ticker
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
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


# =========================
# Validate Yahoo (.BK) + scanning
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def validate_yahoo_symbols(symbols: list[str], max_check: int = 250) -> list[str]:
    """
    ตรวจว่าดึงราคาได้จริงใน Yahoo: symbol -> symbol.BK
    ทำเบา ๆ เพื่อลดโอกาสโดนจำกัด
    """
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
    """
    คำนวณสัญญาณ + โซนเทรนด์ + TP/SL/TrailingStart
    ใช้ OHLCV (yfinance) ใน df columns: Open High Low Close Volume
    """
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

    # --- Signal scoring for short ---
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

    # --- Trend zone for TP/SL ---
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
    """
    บรรทัดละตัว: Ticker,Cost,Qty  (Ticker แบบ PTT.BK)
    """
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
# Sidebar UI
# =========================
with st.sidebar:
    st.header("1) เลือกแหล่งรายชื่อหุ้น")
    st.caption("แนะนำ: อัปโหลด CSV/XLSX (เสถียรกว่าเว็บ) | ต้องมีคอลัมน์ Symbol/Ticker")
    uploaded = st.file_uploader("อัปโหลดรายชื่อหุ้น (CSV/XLSX)", type=["csv", "xlsx"])

    st.divider()
    st.header("2) ตั้งค่าการสแกน")
    validate_n = st.slider("จำนวนที่จะ validate ว่าดึงจาก Yahoo ได้ (.BK)", 50, 800, 250, 50)
    scan_n = st.slider("จำนวนหุ้นที่จะสแกนจริง", 20, 500, 120, 10)
    period = st.selectbox("ช่วงข้อมูลที่ใช้คำนวณ", ["6mo", "1y", "2y"], index=1)

    st.divider()
    st.header("3) My Portfolio (optional)")
    st.caption("บรรทัดละตัว: Ticker,Cost,Qty  (Ticker แบบ PTT.BK)")
    portfolio_text = st.text_area("ตัวอย่าง:\nPTT.BK,34.50,1000\nAOT.BK,62.00,200", height=120)

    st.divider()
    run = st.button("Run Scan", type="primary")


# =========================
# Run
# =========================
if run:
    # 1) Load symbols
    with st.spinner("กำลังโหลดรายชื่อหุ้น..."):
        if uploaded is not None:
            sym_df = load_symbols_from_upload(uploaded)
            source_msg = f"ใช้รายชื่อจากไฟล์อัปโหลด ({len(sym_df):,} symbols)"
        else:
            try:
                sym_df = load_symbols_from_stockanalysis()
                source_msg = f"ใช้รายชื่อจากเว็บ StockAnalysis ({len(sym_df):,} symbols)"
            except Exception:
                st.error("ดึงรายชื่อหุ้นจากเว็บไม่สำเร็จ → แนะนำให้อัปโหลดไฟล์ CSV/XLSX รายชื่อหุ้นแทน")
                st.stop()

    st.success(source_msg)

    base_symbols = sym_df["Symbol"].tolist()

    # 2) Validate yfinance availability
    with st.spinner("กำลัง validate ว่าหุ้นมีข้อมูลใน Yahoo (.BK) ..."):
        ok_tickers = validate_yahoo_symbols(base_symbols, max_check=validate_n)

    st.write(f"ผ่าน validate: **{len(ok_tickers):,}** ตัว | จะสแกน: **{min(scan_n, len(ok_tickers)):,}** ตัว")

    if len(ok_tickers) == 0:
        st.error("ไม่ผ่าน validate เลย (อาจโดนจำกัดจาก Yahoo) → ลด validate_n หรือเปลี่ยนเน็ต/ลองใหม่")
        st.stop()

    tickers = ok_tickers[:scan_n]

    # 3) Download & compute
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

    # Ranking: BUY first then score high
    order_map = {"BUY": 0, "WAIT": 1, "SELL": 2}
    out["Order"] = out["Action"].map(order_map).fillna(9).astype(int)
    out = out.sort_values(["Order", "Score"], ascending=[True, False]).drop(columns=["Order"])

    # =========================
    # No Position advice
    # =========================
    st.subheader("คำแนะนำกรณี 'ไม่มีของ' (No Position)")
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY", int((out["Action"] == "BUY").sum()))
    c2.metric("WAIT", int((out["Action"] == "WAIT").sum()))
    c3.metric("SELL", int((out["Action"] == "SELL").sum()))
    st.write(
        "- ถ้า **BUY น้อย/ไม่มี**: ไม่ต้องฝืนเทรด → เก็บ Watchlist จาก WAIT ที่คะแนนสูง ๆ แล้วรอให้เป็น BUY\n"
        "- ถ้า **BUY มี**: เข้าได้ตามระบบ พร้อม TP/SL ตาม TrendZone\n"
        "- ถ้า **SELL**: ไม่เข้า รอจนกลับเป็น BUY"
    )

    # =========================
    # Tables
    # =========================
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top BUY")
        st.dataframe(out[out["Action"] == "BUY"].head(30), use_container_width=True)
    with colB:
        st.subheader("Top SELL / ระวัง")
        st.dataframe(out[out["Action"] == "SELL"].head(30), use_container_width=True)

    st.subheader("ทั้งหมด (ค้นหา/กรองได้)")
    st.dataframe(out, use_container_width=True)

    # =========================
    # Portfolio advice
    # =========================
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
                    "Suggestion": "เพิ่ม scan_n หรือเช็ค ticker ต้องลงท้าย .BK",
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

            # decision engine ตามที่คุย
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
                    sug = "กำไรอยู่แต่สัญญาณ SELL → แนะนำล็อกกำไร/ทยอยปิด (อย่างน้อย 50%)"
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
### Deploy บน Streamlit Cloud (สั้น ๆ)
1) Push `app.py` + `requirements.txt` ขึ้น GitHub  
2) ไปที่ Streamlit Community Cloud → New app → เลือก repo → เลือก `app.py` → Deploy  
3) แก้โค้ดใน GitHub แล้ว commit → ระบบ redeploy อัตโนมัติ
"""
    )
