#!/usr/bin/env python3
"""Signal bot: sends BUY/SELL signals (with SL / BE / TP) to ntfy. It never trades.

  python bot.py            normal run (what the GitHub workflow does)
  python bot.py --status   print the rule checks per coin, send nothing
"""
import json, os, sys, time
from pathlib import Path

import pandas as pd
import requests

PAIRS = ["BNB", "XRP", "DOGE", "NEAR", "DOT", "SOL"]  # coins we signal on
FILTER = "BTC"                                         # bias filter only
QUOTE = "USDT"
BASES = ["https://data-api.binance.vision", "https://api.binance.com"]

TOPIC = os.getenv("NTFY_TOPIC", "")
NTFY = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
DRY = os.getenv("DRY_RUN", "0") == "1"
NEWS_REQUIRED = os.getenv("NEWS_REQUIRED", "0") == "1"  # 1 = calendar unavailable -> no signal
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"   # Forex Factory calendar feed
NEWS_CURRENCIES = {c.strip().upper() for c in os.getenv("NEWS_CURRENCIES", "USD").split(",") if c.strip()}
NEWS_BEFORE_MIN = int(os.getenv("NEWS_BEFORE_MIN", "30"))   # block this long before red news
NEWS_AFTER_MIN = int(os.getenv("NEWS_AFTER_MIN", "30"))     # ...and this long after
NEWS_REFRESH_H = 3        # re-download the calendar at most this often (feed is rate limited)
STATE = Path("state.json")

# ---- tunables ----
SETUP_LOOKBACK = 4        # 15m candles a CRT/CSD stays valid
SWING_K = 2               # candles each side of an M1 swing point
BREAK_MAX_AGE_MIN = 20    # the M1 break must be this fresh
MOMENTUM_BARS = 3         # BTC "higher closes" lookback (15m)
ATR_N = 14
SL_BUFFER_ATR = 0.25      # SL sits this far beyond the sweep
MIN_RISK_ATR, MAX_RISK_ATR = 0.3, 3.0   # skip too-tight / too-wide stops
BE_R, TP1_R = 1.0, 1.5
TP2_R = {"A+": 3.0, "BUY": 2.0}         # every signal is at least 1:2
CHASE_R = 0.5             # "don't enter beyond" distance
MAX_OPEN_HOURS = 48


# ---------------- data ----------------
def klines(sym, interval, limit):
    err = None
    for base in BASES:
        try:
            r = requests.get(f"{base}/api/v3/klines", timeout=15,
                             params={"symbol": sym + QUOTE, "interval": interval, "limit": limit})
            r.raise_for_status()
            d = pd.DataFrame(r.json()).iloc[:, :6].astype(float)
            d.columns = ["t", "o", "h", "l", "c", "v"]
            d["t"] = d["t"].astype("int64")
            return d.iloc[:-1].reset_index(drop=True)   # drop the unfinished candle
        except Exception as e:
            err = e
    raise RuntimeError(f"{sym} {interval}: {err}")


def mirror(d):
    """Flip prices so the bullish logic can detect bearish setups (sells)."""
    m = d.copy()
    m["o"], m["c"], m["h"], m["l"] = -d["o"], -d["c"], -d["l"], -d["h"]
    return m


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(d, n=ATR_N):
    pc = d["c"].shift()
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(), (d["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------------- rules ----------------
def find_setup(d):
    """Latest bullish CRT / CSD in the last few closed 15m candles.
    CRT: candle sweeps the previous low and closes back above it.
    CSD: after 2+ down candles, a candle closes above the open of the last down candle."""
    o, h, l, c, t = (d[k].values for k in ("o", "h", "l", "c", "t"))
    n = len(d)
    for i in range(n - 1, n - 1 - SETUP_LOOKBACK, -1):
        kinds, low = [], float("inf")
        if l[i] < l[i - 1] and c[i] > l[i - 1]:
            kinds.append("CRT")
            low = min(low, l[i])
        if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2] and c[i] > o[i - 1]:
            kinds.append("CSD")
            low = min(low, l[i], l[i - 1], l[i - 2])
        if kinds:
            return "+".join(kinds), int(t[i]), float(low)
    return None


def m1_break(m, since_ms):
    """M1 close above the latest confirmed swing high (BOS/CHoCH), during/after the setup candle."""
    h, c, t = m["h"].values, m["c"].values, m["t"].values
    for i in range(len(m) - 1, 2 * SWING_K, -1):
        if t[i] < since_ms:
            break
        for j in range(i - SWING_K - 1, SWING_K - 1, -1):
            if h[j] == h[j - SWING_K: j + SWING_K + 1].max():
                if c[i - 1] <= h[j] < c[i]:
                    return int(t[i]), float(h[j])
                break
    return None


def load_news(st):
    """Red-folder (High impact) Forex Factory events for the chosen currencies.
    Cached in state.json so we stay under the feed's rate limit. None = no data."""
    n = st.get("news") or {}
    if n and time.time() - n.get("fetched", 0) < NEWS_REFRESH_H * 3600:
        return n["events"]
    try:
        r = requests.get(NEWS_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ev = [{"ts": pd.Timestamp(e["date"]).timestamp(), "cur": e["country"], "title": e["title"]}
              for e in r.json() if e.get("impact") == "High" and e.get("country") in NEWS_CURRENCIES]
        st["news"] = {"fetched": time.time(), "events": ev}
        return ev
    except Exception as e:
        print("Forex Factory feed failed:", e)
        if n:   # keep the stale copy and retry in ~10 minutes
            n["fetched"] = time.time() - NEWS_REFRESH_H * 3600 + 600
            st["news"] = n
        return n.get("events")


def news_flag(events):
    """True = inside a red-news window, False = clear, None = no calendar data."""
    if events is None:
        return None
    now = time.time()
    return any(e["ts"] - NEWS_BEFORE_MIN * 60 <= now <= e["ts"] + NEWS_AFTER_MIN * 60 for e in events)


def check(sym, side, d15, d1, b15, events, with_news=True):
    sg = 1 if side == "BUY" else -1
    if sg == -1:
        d15, d1, b15 = mirror(d15), mirror(d1), mirror(b15)
    price = d1["c"].iloc[-1]
    e50, e200 = ema(d15["c"], 50).iloc[-1], ema(d15["c"], 200).iloc[-1]
    setup = find_setup(d15)
    brk = m1_break(d1, setup[1]) if setup else None
    fresh = bool(brk) and d1["t"].iloc[-1] - brk[0] <= BREAK_MAX_AGE_MIN * 60000
    bc = b15["c"]
    btc_ok = bc.iloc[-1] > ema(bc, 50).iloc[-1] and bc.iloc[-1] > bc.iloc[-1 - MOMENTUM_BARS]

    chk = {"crt_csd": bool(setup), "above_ema50": bool(price > e50), "m1_break": fresh,
           "btc_filter": bool(btc_ok), "risk_ok": False, "ema50>200 (bonus)": bool(e50 > e200)}
    sig = None
    if setup and fresh and chk["above_ema50"] and btc_ok:
        a = atr(d15).iloc[-1]
        sl = setup[2] - SL_BUFFER_ATR * a
        risk = price - sl
        chk["risk_ok"] = bool(MIN_RISK_ATR * a <= risk <= MAX_RISK_ATR * a)
        if chk["risk_ok"]:
            news = news_flag(events) if with_news else None
            blocked = news is True or (news is None and with_news and NEWS_REQUIRED)
            if not blocked:
                grade = "A+" if chk["ema50>200 (bonus)"] else "BUY"
                lv = lambda r: sg * (price + r * risk)
                sig = dict(sym=sym, side=side, grade=grade, setup=setup[0], setup_t=setup[1],
                           entry=sg * price, sl=sg * sl, be_at=lv(BE_R), tp1_px=lv(TP1_R),
                           tp2_px=lv(TP2_R[grade]), chase=lv(CHASE_R), rr=TP2_R[grade],
                           news_checked=news is not None)
    return chk, sig


# ---------------- alerts ----------------
def fmt(x):
    return f"{x:.6g}"


def notify(title, body, prio="default", tags=""):
    print(f"[{prio}] {title}\n{body}\n")
    if DRY:
        return
    if not TOPIC:
        print("NTFY_TOPIC not set - nothing sent")
        return
    try:
        requests.post(f"{NTFY}/{TOPIC}", data=body.encode("utf-8"), timeout=15,
                      headers={"Title": title, "Priority": prio, "Tags": tags})
    except Exception as e:
        print("ntfy failed:", e)


def signal_text(s):
    buy = s["side"] == "BUY"
    lines = [
        f"Entry ~{fmt(s['entry'])} (skip if price already {'above' if buy else 'below'} {fmt(s['chase'])})",
        f"SL {fmt(s['sl'])}",
        f"Move SL to entry once price hits {fmt(s['be_at'])} (1R)",
        f"TP1 {fmt(s['tp1_px'])} (take about half)",
        f"TP2 {fmt(s['tp2_px'])}  |  R:R 1:{s['rr']:g}",
        f"Setup: {s['setup']} + M1 structure break | BTC {'bullish' if buy else 'bearish'}",
    ]
    if not s["news_checked"]:
        lines.append("Note: news NOT checked")
    return "\n".join(lines)


# ---------------- open-signal tracking ----------------
def manage(st, d1):
    keep = []
    for s in st["open"]:
        sg = 1 if s["side"] == "BUY" else -1
        name = f"{s['sym']}/USDT {s['side']}"
        done = False
        for t, h, l in zip(d1[s["sym"]]["t"], d1[s["sym"]]["h"], d1[s["sym"]]["l"]):
            if t <= s["checked"]:
                continue
            s["checked"] = int(t)
            adverse, fav = (l, h) if sg == 1 else (h, l)
            stop = s["entry"] if s["be"] else s["sl"]
            if sg * adverse <= sg * stop:
                if s["be"]:
                    notify(f"{name}: stopped at BE", "Closed at entry" +
                           (" (TP1 already banked)." if s["tp1"] else "."), tags="white_check_mark")
                else:
                    notify(f"{name}: SL hit", f"Stopped out at {fmt(s['sl'])} (-1R).", "high", "x")
                done = True
                break
            if sg * fav >= sg * s["tp2_px"]:
                notify(f"{name}: TP2 hit", f"Target 2 reached at {fmt(s['tp2_px'])}. Close the rest.",
                       "high", "tada")
                done = True
                break
            if not s["tp1"] and sg * fav >= sg * s["tp1_px"]:
                s["tp1"] = s["be"] = True
                notify(f"{name}: TP1 hit", f"Take ~half at {fmt(s['tp1_px'])} and move SL to entry "
                       f"{fmt(s['entry'])}.", tags="white_check_mark")
            elif not s["be"] and sg * fav >= sg * s["be_at"]:
                s["be"] = True
                notify(f"{name}: move SL to BE", f"Price hit {fmt(s['be_at'])}. Move SL to entry "
                       f"{fmt(s['entry'])}.", tags="warning")
        if not done and time.time() - s["opened"] > MAX_OPEN_HOURS * 3600:
            notify(f"{name}: expired", f"Open {MAX_OPEN_HOURS}h with no result. Stop tracking.")
            done = True
        if not done:
            keep.append(s)
    st["open"] = keep


# ---------------- main ----------------
def main():
    status = "--status" in sys.argv
    st = json.loads(STATE.read_text()) if STATE.exists() else {"open": [], "seen": {}}
    try:
        b15 = klines(FILTER, "15m", 500)
    except Exception as e:
        print("BTC data failed:", e)
        return
    d15, d1 = {}, {}
    for s in PAIRS:
        try:
            d15[s], d1[s] = klines(s, "15m", 500), klines(s, "1m", 300)
        except Exception as e:
            print(e)
    events = load_news(st)
    if status:
        print("Red news window now:", {None: "unknown (no calendar data)", True: "YES", False: "no"}[news_flag(events)])
    else:
        manage(st, d1)
    open_syms = {o["sym"] for o in st["open"]}
    now = time.time()
    st["seen"] = {k: v for k, v in st["seen"].items() if now - v < 72 * 3600}

    for s in d15:
        for side in ("BUY", "SELL"):
            chk, sig = check(s, side, d15[s], d1[s], b15, events)
            if status:
                print(f"{side:4} {s:5}", "  ".join(f"{k}={'Y' if v else '-'}" for k, v in chk.items()))
                continue
            if not sig or s in open_syms:
                continue
            key = f"{s}|{side}|{sig['setup_t']}"
            if key in st["seen"]:
                continue
            st["seen"][key] = now
            title = f"{side} {s}/USDT" + (" A+" if sig["grade"] == "A+" else "")
            notify(title, signal_text(sig), "high" if sig["grade"] == "A+" else "default",
                   "chart_with_upwards_trend" if side == "BUY" else "chart_with_downwards_trend")
            sig.update(id=key, opened=now, checked=int(d1[s]["t"].iloc[-1]), be=False, tp1=False)
            st["open"].append(sig)
            open_syms.add(s)

    if not status and not DRY:
        STATE.write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
