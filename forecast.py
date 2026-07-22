"""
Ping Forecast (live, cloud) - inference-only daily job.

Self-contained: uses pre-trained XGBoost models (models/*.json) and the
set_1b column list (assets/feature_cols_1b.json). No thesis data, no
training. Runs on a GitHub Actions runner:

  1. Fetch 14 days of daily discharge for P.1/P.67/P.20/P.75 (RID Region 1).
  2. Fetch the 7-day NWP rain forecast at P.1 (Open-Meteo).
  3. Build set_1b features for origin = last complete day; forecast leads
     1,2,3,4,7 with persistence, xgb_1b, and xgb_1b_nwp.
  4. Append to forecast_log.csv (state, committed back by the workflow).
  5. Verify matured rows against observations; roll up MAE/bias.
  6. Render docs/index.html from template.html.

Models were trained in the thesis pipeline (identical params). Endpoint
values are the PROVISIONAL RID revision (~2-3 m3/s off the thesis dataset
at P.1) - live demo / prospective verification only, never for retraining.
"""
import os, sys, json, re, random, urllib.request
from datetime import datetime, timedelta, date

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

import numpy as np
import pandas as pd
import xgboost as xgb
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
LOG_CSV = os.path.join(ROOT, "forecast_log.csv")
DOCS = os.path.join(ROOT, "docs")

FLOOD_THRESHOLD = 416
STATIONS = {"P.1": "p1", "P.67": "p67", "P.20": "p20", "P.75": "p75"}
LEADS = [1, 2, 3, 4, 7]
LAT, LON = 18.79, 98.98

WATER_URL = ("https://www.hydro-1.net/Data/HD-04/water_today/"
             "check_water_json.php?callback=x&date={d}&level=3&search=p")
RAIN_FC_URL = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={LAT}&longitude={LON}"
               "&daily=precipitation_sum&forecast_days=8&timezone=Asia/Bangkok")

with open(os.path.join(ROOT, "assets", "feature_cols_1b.json")) as f:
    COLS_1B = json.load(f)


def fetch_jsonp(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    txt = raw.decode("cp874", "replace").strip().lstrip("﻿")
    m = re.search(r"^[^(\[{]*\((.*)\)\s*;?\s*$", txt, re.S)
    return json.loads(m.group(1) if m else txt)


def to_float(v):
    try:
        f = float(str(v).replace(",", ""))
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_water_window(end_day):
    frames, p1_level = {}, {}
    for req_day in (end_day, end_day - timedelta(days=7)):
        data = fetch_jsonp(WATER_URL.format(d=req_day.isoformat()))
        items = data if isinstance(data, list) else data.get("data", [])
        for it in items:
            sid = str(it.get("station_id", "")).strip()
            if sid not in STATIONS:
                continue
            for i in range(1, 8):
                d = req_day - timedelta(days=i - 1)
                frames.setdefault(sid, {})[d] = to_float(it.get(f"dischg{i}"))
                if sid == "P.1":
                    p1_level[d] = to_float(it.get(f"level{i}"))
            m = re.search(r"\d+", str(it.get("day3", "")))
            if m and int(m.group()) != (req_day - timedelta(days=2)).day:
                raise RuntimeError(
                    f"endpoint day ordering changed (day3={it.get('day3')!r}) "
                    "- refusing to build lags")
    q_df = pd.DataFrame(frames).sort_index()
    q_df.index = pd.to_datetime(q_df.index)
    return q_df, p1_level


def fetch_rain_forecast():
    data = json.loads(urllib.request.urlopen(
        urllib.request.Request(RAIN_FC_URL, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=60).read().decode())
    return {pd.Timestamp(d): (v if v is not None else np.nan)
            for d, v in zip(data["daily"]["time"],
                            data["daily"]["precipitation_sum"])}


def load_model(kind, lead):
    path = os.path.join(MODEL_DIR, f"{kind}_lead{lead}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing model {path} - commit pre-trained models")
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m


def main():
    now = datetime.now()
    origin = date.today() - timedelta(days=1)
    print(f"[{now:%Y-%m-%d %H:%M}] live forecast - origin {origin}")

    q_df, p1_level = fetch_water_window(origin)
    rain_fc = fetch_rain_forecast()

    o = pd.Timestamp(origin)
    need = [o - pd.Timedelta(days=k) for k in range(0, 8)]
    missing = [d.date() for d in need if d not in q_df.index or q_df.loc[d].isna().any()]
    if missing:
        raise SystemExit(f"missing/NaN discharge for {missing} - skip run")

    feat = {}
    for sid, suff in STATIONS.items():
        for k in range(1, 8):
            feat[f"Discharge_{suff}_lag{k}"] = q_df.loc[o - pd.Timedelta(days=k), sid]
    x_row = pd.DataFrame([feat])[COLS_1B]

    rows = []
    q_now = q_df.loc[o, "P.1"]
    for lead in LEADS:
        target = o + pd.Timedelta(days=lead)
        rain_days = [rain_fc.get(o + pd.Timedelta(days=k), np.nan) for k in range(1, lead + 1)]
        rain_accum = float(np.nansum(rain_days)) if rain_days else 0.0

        preds = {"persistence": float(q_now)}
        preds["xgb_1b"] = float(load_model("xgb_1b", lead).predict(x_row)[0])
        x2 = x_row.copy()
        x2[f"rain_future_{lead}d"] = rain_accum
        preds["xgb_1b_nwp"] = float(
            load_model("xgb_1b_nwp", lead).predict(x2[COLS_1B + [f"rain_future_{lead}d"]])[0])

        for model_name, pred in preds.items():
            rows.append({
                "issued_at": now.isoformat(timespec="minutes"),
                "origin_date": origin.isoformat(), "model": model_name, "lead": lead,
                "target_date": target.date().isoformat(), "predicted": round(pred, 1),
                "nwp_rain_accum_mm": round(rain_accum, 1), "actual": "", "verified_at": "",
            })

    cols = list(rows[0].keys())
    log = (pd.read_csv(LOG_CSV, dtype=str) if os.path.exists(LOG_CSV)
           else pd.DataFrame(columns=cols))
    new = pd.DataFrame(rows).astype(str)
    key = ["origin_date", "model", "lead"]
    log = pd.concat([log[~log.set_index(key).index.isin(new.set_index(key).index)],
                     new], ignore_index=True)

    obs = {d.date().isoformat(): float(q_df.loc[d, "P.1"])
           for d in q_df.index if not pd.isna(q_df.loc[d, "P.1"])}
    unv = (log["actual"] == "") & log["target_date"].isin(obs)
    log.loc[unv, "actual"] = log.loc[unv, "target_date"].map(lambda d: f"{obs[d]:.1f}")
    log.loc[unv, "verified_at"] = now.isoformat(timespec="minutes")
    log.sort_values(["origin_date", "lead", "model"], inplace=True)
    log.to_csv(LOG_CSV, index=False)
    n_matured = int((log["actual"] != "").sum())
    print(f"  log: {len(log)} rows, {n_matured} verified")

    lv = log[log["actual"] != ""].copy()
    acc = []
    if len(lv):
        lv["err"] = lv["predicted"].astype(float) - lv["actual"].astype(float)
        for (mname, lead), g in lv.groupby(["model", lv["lead"].astype(int)]):
            acc.append({"model": mname, "lead": int(lead), "n": len(g),
                        "mae": round(g["err"].abs().mean(), 1),
                        "bias": round(g["err"].mean(), 1)})

    hist = [{"date": d.date().isoformat(), "q": round(float(q_df.loc[d, "P.1"]), 1)}
            for d in q_df.index if not pd.isna(q_df.loc[d, "P.1"])]
    fc_out = [{"model": r["model"], "lead": int(r["lead"]),
               "target_date": r["target_date"], "predicted": float(r["predicted"])}
              for r in rows]
    recent = lv.sort_values("target_date").tail(12)
    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"), "origin_date": origin.isoformat(),
        "flood_threshold": FLOOD_THRESHOLD,
        "p1_now": {"q": float(q_now), "level": p1_level.get(o, None), "level_limit": 3.70},
        "p67_now": {"q": float(q_df.loc[o, "P.67"]),
                    "q_prev": float(q_df.loc[o - pd.Timedelta(days=1), "P.67"])},
        "history": hist, "forecasts": fc_out,
        "rain_forecast": [{"date": d.date().isoformat(), "mm": round(float(v), 1)}
                          for d, v in sorted(rain_fc.items()) if not pd.isna(v)][:8],
        "accuracy": acc, "n_log_rows": len(log), "n_verified": n_matured,
        "recent_verified": [
            {"target_date": r["target_date"], "model": r["model"], "lead": int(r["lead"]),
             "predicted": float(r["predicted"]), "actual": float(r["actual"])}
            for _, r in recent.iterrows()],
    }

    with open(os.path.join(ROOT, "template.html"), encoding="utf-8") as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  dashboard -> {os.path.join(DOCS, 'index.html')}")


if __name__ == "__main__":
    main()
