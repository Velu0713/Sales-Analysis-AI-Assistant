# analysis_engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# forecasting fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor

def fmt_currency(x):
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return str(x)

def try_parse_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = c.lower()
        if "order date" in lc or "orderdate" in lc or lc == "date" or ("date" in lc and "order" in lc):
            try:
                pd.to_datetime(df[c], errors="coerce")
                return c
            except Exception:
                continue
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None

def analyze_query(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[plt.Figure]]:
    q = (query or "").lower().strip()
    if q == "":
        return None, None

    # Ensure numeric columns
    for col in ["Sales", "Profit", "Discount", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    date_col = try_parse_date_column(df)

    # 1) Totals
    if ("total sales" in q or "sum sales" in q or "sales total" in q) and "Sales" in df.columns:
        total = df["Sales"].sum()
        return f"Total Sales (all rows): {fmt_currency(total)}", None

    if ("total profit" in q or "sum profit" in q or "profit total" in q) and "Profit" in df.columns:
        total = df["Profit"].sum()
        return f"Total Profit (all rows): {fmt_currency(total)}", None

    # 2) Top N
    import re
    m = re.search(r"top\s+(\d+)", q)
    n = int(m.group(1)) if m else 5
    if ("top" in q or "highest" in q or "most" in q) and ("sales" in q or "profit" in q):
        if "state" in q and "State" in df.columns and "Sales" in df.columns:
            agg = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(n)
            text = "Top {} states by Sales:\n".format(n) + "\n".join([f"{i+1}. {s} — {fmt_currency(v)}" for i,(s,v) in enumerate(agg.items())])
            fig = plt.figure(figsize=(6,3)); agg.sort_values().plot(kind="barh"); plt.xlabel("Sales"); plt.tight_layout()
            return text, fig
        if "category" in q and "Category" in df.columns and "Sales" in df.columns:
            agg = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).head(n)
            text = "Top {} categories by Sales:\n".format(n) + "\n".join([f"{i+1}. {s} — {fmt_currency(v)}" for i,(s,v) in enumerate(agg.items())])
            fig = plt.figure(figsize=(6,3)); agg.sort_values().plot(kind="barh"); plt.xlabel("Sales"); plt.tight_layout()
            return text, fig
        if "product" in q and "Product Name" in df.columns and "Sales" in df.columns:
            agg = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(n)
            text = "Top {} products by Sales:\n".format(n) + "\n".join([f"{i+1}. {s} — {fmt_currency(v)}" for i,(s,v) in enumerate(agg.items())])
            fig = plt.figure(figsize=(6,3)); agg.plot(kind="bar"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            return text, fig

    # 3) Average discount
    if ("average discount" in q or "avg discount" in q or "mean discount" in q) and "Discount" in df.columns:
        avg = df["Discount"].mean()
        return f"Average discount is {avg*100:.2f}% (if Discount is fraction).", None

    # 4) Monthly trend
    if date_col and ("trend" in q or "monthly" in q or "month-wise" in q or "sales trend" in q):
        tmp = df.copy(); tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        monthly = tmp.set_index(date_col).resample("M")["Sales"].sum().dropna()
        if len(monthly) >= 3:
            text = f"Monthly sales trend from {monthly.index.min().strftime('%Y-%m')} to {monthly.index.max().strftime('%Y-%m')}."
            fig = plt.figure(figsize=(8,3)); monthly.plot(marker="o"); plt.ylabel("Sales"); plt.xlabel("Month"); plt.tight_layout()
            return text, fig

    # 5) Filtered totals by year
    yr = re.search(r"\b(20\d{2})\b", q)
    if yr and date_col:
        year = int(yr.group(1))
        tmp = df.copy(); tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp[tmp[date_col].dt.year == year]
        if "Sales" in tmp.columns:
            total = tmp["Sales"].sum()
            return f"Total sales in {year}: {fmt_currency(total)}", None

    # 6) Row count
    if ("number of orders" in q or "count orders" in q or "orders count" in q or "how many orders" in q):
        return f"Total rows/orders in dataset: {len(df)}", None

    # 7) Forecasting (Prophet or RandomForest fallback)
    if ("forecast" in q or "predict" in q) and ("month" in q or "next" in q):
        if not date_col or "Sales" not in df.columns:
            return "Not enough time-series data to forecast (missing date or Sales column).", None
        tmp = df.copy(); tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        monthly = tmp.set_index(date_col).resample("M")["Sales"].sum().dropna().reset_index().rename(columns={date_col:"ds","Sales":"y"})
        if len(monthly) < 2:
            return "Insufficient history for forecasting.", None
        if PROPHET_AVAILABLE and len(monthly) >= 12:
            m = Prophet(); m.fit(monthly)
            future = m.make_future_dataframe(periods=6, freq="M")
            forecast = m.predict(future)
            fig = m.plot(forecast)
            return "Sales forecast (Prophet) for next 6 months.", fig
        else:
            monthly["month_num"] = range(len(monthly))
            X = monthly[["month_num"]]; y = monthly["y"]
            rf = RandomForestRegressor(n_estimators=200, random_state=0); rf.fit(X, y)
            future_n = 6; future_X = pd.DataFrame({"month_num": range(len(monthly), len(monthly)+future_n)})
            preds = rf.predict(future_X)
            fig = plt.figure(figsize=(8,3))
            all_y = pd.concat([monthly["y"], pd.Series(preds)], ignore_index=True)
            idx = list(monthly["ds"].dt.to_period("M").astype(str)) + [f"future_{i+1}" for i in range(len(preds))]
            plt.plot(range(len(all_y)), all_y, marker="o"); plt.xticks(range(len(all_y)), idx, rotation=45); plt.title("Sales (history + predicted)"); plt.tight_layout()
            return f"Forecast (RandomForest) for next {future_n} months.", fig

    return None, None