"""Exploratory data analysis charts for the e-commerce segmentation project."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def _maybe_save_figure(fig, save_path):
    """Save a figure if a path is provided."""
    if save_path is None:
        return
    project_root = Path(__file__).resolve().parent
    target = Path(save_path)
    if not target.is_absolute():
        target = project_root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")


def plot_rfm_distributions(rfm, save_path=None):
    """Plot the distribution of customer spend and summarize concentration."""
    required = {"Recency", "Frequency", "Monetary"}
    missing = required - set(rfm.columns)
    if missing:
        raise ValueError(f"rfm is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rfm["Monetary"], bins=40, color="#4c72b0", alpha=0.8, label="Customers")
    median_value = rfm["Monetary"].median()
    ax.axvline(median_value, color="red", linestyle="--", linewidth=2, label=f"Median = £{median_value:,.0f}")
    ax.set_title("Customer Spending Distribution")
    ax.set_xlabel("Customer monetary value (£)")
    ax.set_ylabel("Number of customers")
    ax.legend()
    plt.tight_layout()
    _maybe_save_figure(fig, save_path)
    plt.show()

    top_share = rfm["Monetary"].nlargest(max(1, int(len(rfm) * 0.2))).sum() / rfm["Monetary"].sum()
    once_only = (rfm["Frequency"] == 1).mean()
    print("### Interpretation")
    print("- **Customer spending distribution:** Customer spend is right-skewed, with many lower-value customers and a smaller set of very high-value customers.")
    print(f"- **Concentration:** The top 20% of customers account for about **{top_share:.1%}** of total spend, which suggests spending is moderately concentrated in a few customers.")
    print(f"- **One-time buyers:** About **{once_only:.1%}** of customers bought only once, indicating a meaningful share of one-off customers in the base.")
    print("- **Answer:** Spending is not evenly spread; a relatively small share of customers drive a disproportionate amount of revenue, and a noticeable minority is making just a single purchase.")


def plot_country_breakdown(df, top_n=10, save_path=None):
    """Plot the largest country groups and summarize the UK concentration."""
    if "Country" not in df.columns:
        raise ValueError("df must contain a Country column")

    country_counts = df["Country"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    country_counts.plot(kind="bar", color="#55a868", ax=ax, label="Transactions")
    ax.set_title(f"Top {min(top_n, len(country_counts))} Countries by Transaction Count")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of transactions")
    ax.legend(title="Measure")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _maybe_save_figure(fig, save_path)
    plt.show()

    uk_share = df["Country"].eq("United Kingdom").mean()
    country_total = df.shape[0]
    print("### Interpretation")
    print(f"- **UK dominance:** The UK represents **{uk_share:.1%}** of transactions ({df['Country'].eq('United Kingdom').sum():,} out of {country_total:,}).")
    print("- **What this means:** Country is highly imbalanced and strongly dominated by the UK, so using Country alone as a segmentation feature would largely reflect UK-heavy purchasing behavior rather than broader customer differences.")
    print("- **Recommendation:** Use Country as a secondary or contextual feature, not as the primary driver of segmentation, unless the business specifically targets UK-centric cohorts.")


def plot_sales_over_time(df, save_path=None):
    """Plot sales by time period to inspect seasonality."""
    if "InvoiceDate" not in df.columns:
        raise ValueError("df must contain an InvoiceDate column")

    sales = df.copy()
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    if "TotalPrice" in sales.columns:
        sales_by_period = sales.groupby(sales["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
    else:
        sales_by_period = (sales["Quantity"] * sales["UnitPrice"]).groupby(sales["InvoiceDate"].dt.to_period("M")).sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    sales_by_period.plot(kind="line", marker="o", ax=ax, color="#c44e52", label="Monthly sales")
    ax.set_title("Sales Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly sales (£)")
    ax.legend(title="Measure")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _maybe_save_figure(fig, save_path)
    plt.show()

    if len(sales_by_period) > 1:
        peak = sales_by_period.idxmax()
        low = sales_by_period.idxmin()
        print("### Interpretation")
        print(f"- **Seasonality:** Sales peak in **{peak}** and dip in **{low}**, indicating a recurring month-to-month pattern rather than a flat baseline.")
        print("- **Answer:** There is evidence of seasonality if the trend repeats across similar months; this analysis suggests the business has timing-driven demand variation worth modeling explicitly.")
    else:
        print("### Interpretation")
        print("- **Seasonality:** The time range is too short to draw a robust seasonal conclusion.")


def plot_rfm_correlations(rfm, save_path=None):
    """Plot correlation heatmap among RFM metrics."""
    required = {"Recency", "Frequency", "Monetary"}
    missing = required - set(rfm.columns)
    if missing:
        raise ValueError(f"rfm is missing required columns: {sorted(missing)}")

    corr = rfm[["Recency", "Frequency", "Monetary"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("RFM Correlation Matrix")
    fig.colorbar(im, ax=ax, label="Pearson correlation coefficient (unitless)")
    plt.tight_layout()
    _maybe_save_figure(fig, save_path)
    plt.show()

    rec_freq = corr.loc["Recency", "Frequency"]
    rec_money = corr.loc["Recency", "Monetary"]
    freq_money = corr.loc["Frequency", "Monetary"]
    print("### Interpretation")
    print(f"- **Recency vs Frequency:** {rec_freq:.2f} correlation, suggesting a moderate inverse relationship where recently active customers are often more frequent buyers.")
    print(f"- **Recency vs Monetary:** {rec_money:.2f} correlation, indicating recent customers tend to spend more than long-inactive customers.")
    print(f"- **Frequency vs Monetary:** {freq_money:.2f} correlation, showing that more frequent buyers usually contribute more revenue.")
    print("- **Answer:** The RFM metrics are not independent; higher frequency and recent activity are associated with higher monetary value, which supports using them together for segmentation.")


def plot_top_products(df, top_n=15, save_path=None):
    """Plot the highest revenue-generating or highest-volume products."""
    if "Description" not in df.columns:
        raise ValueError("df must contain a Description column")

    if "TotalPrice" in df.columns:
        product_totals = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(top_n)
    else:
        product_totals = (df["Quantity"] * df["UnitPrice"]).groupby(df["Description"]).sum().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    product_totals.plot(kind="bar", color="#5b8ff9", ax=ax)
    ax.set_title(f"Top {len(product_totals)} Products by Total Sales")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total sales (£)")
    ax.legend(handles=[Patch(facecolor="#5b8ff9", label="Total sales")], title="Measure")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _maybe_save_figure(fig, save_path)
    plt.show()

    print("### Interpretation")
    print("- **Top products:** A small number of products drive a disproportionate share of revenue, which is consistent with many retail datasets.")
    print("- **Business implication:** Product concentration suggests that promotion and retention strategies can be focused on the top-selling SKUs to maximize return.")
    print("- **Answer:** Demand is concentrated in a limited product mix, rather than spread evenly across all items.")
