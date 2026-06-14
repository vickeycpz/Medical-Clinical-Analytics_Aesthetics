"""
Medical Aesthetics — Kaggle — Power BI Data Prep
======================================
Reads the three raw tables produced by enrich_kaggle.py and outputs
five clean, analysis-ready CSVs optimised for Power BI import.

Input files (place in --data dir)
───────────────────────────────────
  appointments_enriched.csv   main enriched Kaggle table
  clinics.csv                 clinic reference table
  treatments.csv              treatment reference table

Output files (written to --out dir)
───────────────────────────────────
  fact_appointments.csv       cleaned event-level fact table
  dim_patients.csv            one row per patient, RFM + LTV tier
  dim_clinics.csv             enriched clinic dimension
  agg_monthly_funnel.csv      month × clinic × channel funnel KPIs
  agg_cohort_retention.csv    cohort month × period retention matrix

Usage
──────
  pip install pandas numpy
  python prep_for_pbi.py
  python prep_for_pbi.py --data ./my_raw_data --out ./pbi_data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Prep MedSpa data for Power BI")
    p.add_argument("--data", default="./data/processed",
                   help="Directory containing the three input CSVs")
    p.add_argument("--out",  default="./pbi_data",
                   help="Directory where output CSVs are written")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["appointments_enriched.csv", "clinics.csv", "treatments.csv"]
    missing  = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"[error] Missing input files in {data_dir}: {missing}")
        sys.exit(1)

    raw   = pd.read_csv(data_dir / "appointments_enriched.csv")
    clin  = pd.read_csv(data_dir / "clinics.csv")
    treat = pd.read_csv(data_dir / "treatments.csv")

    print(f"  appointments_enriched : {len(raw):>7,} rows × {raw.shape[1]} cols")
    print(f"  clinics               : {len(clin):>7,} rows")
    print(f"  treatments            : {len(treat):>7,} rows")
    return raw, clin, treat


def _safe_qcut(series: pd.Series, q: int, labels: list) -> pd.Series:
    """qcut with rank-based tie-breaking — never fails on small/skewed data."""
    try:
        return pd.qcut(series.rank(method="first"), q=q, labels=labels)
    except Exception:
        return pd.Series(labels[len(labels) // 2], index=series.index)


# ══════════════════════════════════════════════════════════════
#  OUTPUT 1 — fact_appointments
#  Power BI page: all four pages use this as the core fact table
# ══════════════════════════════════════════════════════════════

def make_fact_appointments(
        raw: pd.DataFrame,
        clin: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean event-level fact table.

    Key transformations:
    - Parse dates, derive 7 date-part columns for PBI time intelligence
    - Convert no_show to 0/1 integer (enables SUM directly in DAX)
    - Add booking lead-time bucket for no-show risk analysis
    - Add age_band for demographic slicing
    - Merge monthly_rent_usd from clinics (needed for revenue-to-rent DAX)
    - Drop columns that live in dim tables to avoid redundancy
    """
    df = raw.copy()

    # ── parse dates ─────────────────────────────────────────
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["ScheduledDay"]   = pd.to_datetime(df["ScheduledDay"])

    # ── type fixes ──────────────────────────────────────────
    df["no_show"]            = (df["No-show"] == "Yes").astype(int)
    df["is_first_visit"]     = df["is_first_visit"].astype(int)
    df["revenue_usd"]        = pd.to_numeric(df["revenue_usd"],  errors="coerce").fillna(0)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["booking_lead_days"]  = pd.to_numeric(df["booking_lead_days"], errors="coerce").fillna(0).astype(int)
    df["Age"]                = df["Age"].clip(lower=0)

    # ── date parts (PBI time intelligence requires these) ───
    df["appt_date"]       = df["AppointmentDay"].dt.date          # plain date for axis
    df["appt_year"]       = df["AppointmentDay"].dt.year
    df["appt_month_num"]  = df["AppointmentDay"].dt.month
    df["appt_month_name"] = df["AppointmentDay"].dt.strftime("%b")
    df["appt_quarter"]    = "Q" + df["AppointmentDay"].dt.quarter.astype(str)
    df["appt_yearmonth"]  = df["AppointmentDay"].dt.strftime("%Y-%m")
    df["appt_week"]       = df["AppointmentDay"].dt.isocalendar().week.astype(int)
    df["appt_dayofweek"]  = df["AppointmentDay"].dt.day_name()

    # ── booking lead bucket ──────────────────────────────────
    df["lead_time_bucket"] = pd.cut(
        df["booking_lead_days"],
        bins=[-1, 0, 7, 30, 9_999],
        labels=["Same day", "1–7 days", "8–30 days", "30+ days"],
    ).astype(str)

    # ── age band ────────────────────────────────────────────
    df["age_band"] = pd.cut(
        df["Age"],
        bins=[17, 29, 44, 59, 120],
        labels=["18–29", "30–44", "45–59", "60+"],
    ).astype(str)

    # ── merge rent from clinics (for revenue-to-rent DAX) ───
    # monthly_rent_usd is optional — only merge if the column exists
    if "monthly_rent_usd" in clin.columns:
        df = df.merge(
            clin[["clinic_id", "monthly_rent_usd"]],
            on="clinic_id", how="left",
            suffixes=("", "_clinic"),
        )
    else:
        df["monthly_rent_usd"] = pd.NA
        print("  [warn] clinics.csv has no 'monthly_rent_usd' — revenue-to-rent ratio will be NA")

    # ── drop columns that belong to dim tables ──────────────
    drop = ["No-show", "clinic_name", "first_name", "last_name",
            "email", "Neighbourhood"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    return df


# ══════════════════════════════════════════════════════════════
#  OUTPUT 2 — dim_patients
#  Power BI page: Patient behaviour (cohort, LTV, RFM segment)
# ══════════════════════════════════════════════════════════════

def make_dim_patients(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per PatientId with:
    - Basic demographics (from first visit row)
    - Behavioural aggregates: attended visits, total spend, tenure
    - RFM scores (Recency / Frequency / Monetary, each 1–4)
    - LTV tier label: Champion / Loyalist / At-Risk / Lost
    """
    df = raw.copy()
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["no_show"]        = (df["No-show"] == "Yes").astype(int)
    df["revenue_usd"]    = pd.to_numeric(df["revenue_usd"], errors="coerce").fillna(0)

    ref_date = df["AppointmentDay"].max()   # "today" for recency calculation

    # ── aggregate to patient level ───────────────────────────
    agg = df.groupby("PatientId").agg(
        first_name          = ("first_name",          "first"),
        last_name           = ("last_name",           "first"),
        email               = ("email",               "first"),
        gender              = ("Gender",              "first"),
        age                 = ("Age",                 "first"),
        clinic_id           = ("clinic_id",           "first"),
        acquisition_channel = ("acquisition_channel", "first"),
        first_visit_date    = ("AppointmentDay",      "min"),
        last_visit_date     = ("AppointmentDay",      "max"),
        total_appointments  = ("AppointmentID",       "count"),
        attended_visits     = ("no_show",             lambda x: (x == 0).sum()),
        lifetime_revenue    = ("revenue_usd",         "sum"),
        avg_spend_per_visit = ("revenue_usd",         lambda x: x[x > 0].mean()),
        avg_satisfaction    = ("satisfaction_score",  "mean"),
        treatment_diversity = ("treatment_category",  "nunique"),
    ).reset_index()

    # ── derived fields ───────────────────────────────────────
    agg["recency_days"] = (ref_date - agg["last_visit_date"]).dt.days
    agg["frequency"]    = agg["attended_visits"]
    agg["monetary"]     = agg["lifetime_revenue"].round(2)
    agg["tenure_days"]  = (agg["last_visit_date"] - agg["first_visit_date"]).dt.days

    # ── RFM scoring (1 = worst, 4 = best) ───────────────────
    #   Recency: lower days = higher score (invert labels)
    #   Frequency & Monetary: higher = higher score
    agg["r_score"] = _safe_qcut(agg["recency_days"], 4, labels=[4, 3, 2, 1])
    agg["f_score"] = _safe_qcut(agg["frequency"],    4, labels=[1, 2, 3, 4])
    agg["m_score"] = _safe_qcut(agg["monetary"],     4, labels=[1, 2, 3, 4])
    agg["rfm_score"] = (
            agg["r_score"].astype(int) +
            agg["f_score"].astype(int) +
            agg["m_score"].astype(int)
    )

    # ── LTV tier ─────────────────────────────────────────────
    def _ltv_tier(score: int) -> str:
        if score >= 10: return "Champion"    # high R + F + M
        if score >= 7:  return "Loyalist"
        if score >= 5:  return "At-Risk"
        return "Lost"

    agg["ltv_tier"] = agg["rfm_score"].apply(_ltv_tier)

    # ── tidy types ───────────────────────────────────────────
    agg["avg_spend_per_visit"] = agg["avg_spend_per_visit"].round(2)
    agg["avg_satisfaction"]    = agg["avg_satisfaction"].round(2)
    agg["lifetime_revenue"]    = agg["lifetime_revenue"].round(2)
    agg["first_visit_date"]    = agg["first_visit_date"].dt.date
    agg["last_visit_date"]     = agg["last_visit_date"].dt.date

    return agg


# ══════════════════════════════════════════════════════════════
#  OUTPUT 3 — dim_clinics  (+ treatments merged as separate CSV)
#  Power BI page: Clinic benchmarks, all pages (slicer)
# ══════════════════════════════════════════════════════════════

def make_dim_clinics(clin: pd.DataFrame) -> pd.DataFrame:
    """Reference dimension — enriched with derived fields.
    Tolerates clinics.csv files that omit optional columns such as
    opened_date, capacity_chairs, and monthly_rent_usd."""
    df = clin.copy()

    # opened_date + years_open — only if column exists
    if "opened_date" in df.columns:
        df["opened_date"] = pd.to_datetime(df["opened_date"], errors="coerce")
        df["years_open"]  = (
                (pd.Timestamp.today() - df["opened_date"]).dt.days / 365
        ).round(1)
    else:
        df["opened_date"] = pd.NaT
        df["years_open"]  = pd.NA
        print("  [warn] clinics.csv has no 'opened_date' column — years_open set to NA")

    # optional columns: fill with NA if absent so downstream DAX still works
    for col in ("capacity_chairs", "monthly_rent_usd"):
        if col not in df.columns:
            df[col] = pd.NA
            print(f"  [warn] clinics.csv has no '{col}' column — filled with NA")

    df["tier_label"] = df["tier"].str.capitalize()

    # sort premium first for PBI slicer default ordering
    df["tier_sort"] = df["tier"].map({"premium": 0, "standard": 1}).fillna(9)
    df.sort_values(["tier_sort", "clinic_id"], inplace=True)
    df.drop(columns=["tier_sort"], inplace=True)

    return df


# ══════════════════════════════════════════════════════════════
#  OUTPUT 4 — agg_monthly_funnel
#  Power BI page: Marketing funnel (page 2)
# ══════════════════════════════════════════════════════════════

def make_agg_monthly_funnel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Month × clinic × channel aggregation.

    Metrics:
    - total_leads         all appointment rows (proxy for leads)
    - new_patients        is_first_visit == 1
    - attended            no_show == 0
    - no_shows            no_show == 1
    - total_revenue       sum of revenue_usd
    - avg_revenue         mean revenue among attended visits
    - show_up_rate_pct    attended / total_leads
    - new_patient_rate_pct new_patients / total_leads
    """
    df = raw.copy()
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["no_show"]        = (df["No-show"] == "Yes").astype(int)
    df["revenue_usd"]    = pd.to_numeric(df["revenue_usd"], errors="coerce").fillna(0)
    df["is_first_visit"] = df["is_first_visit"].astype(int)

    # date parts for PBI axis
    df["yearmonth"]   = df["AppointmentDay"].dt.strftime("%Y-%m")
    df["year"]        = df["AppointmentDay"].dt.year
    df["month_num"]   = df["AppointmentDay"].dt.month
    df["month_name"]  = df["AppointmentDay"].dt.strftime("%b")
    df["quarter"]     = "Q" + df["AppointmentDay"].dt.quarter.astype(str)

    grp = df.groupby(
        ["yearmonth", "year", "month_num", "month_name", "quarter",
         "clinic_id", "clinic_tier", "acquisition_channel"],
        observed=True,
    )

    agg = grp.agg(
        total_leads   = ("AppointmentID",   "count"),
        new_patients  = ("is_first_visit",  "sum"),
        attended      = ("no_show",         lambda x: (x == 0).sum()),
        no_shows      = ("no_show",         "sum"),
        total_revenue = ("revenue_usd",     "sum"),
        avg_revenue   = ("revenue_usd",     lambda x: x[x > 0].mean()),
    ).reset_index()

    agg["show_up_rate_pct"]      = (agg["attended"]     / agg["total_leads"] * 100).round(1)
    agg["new_patient_rate_pct"]  = (agg["new_patients"] / agg["total_leads"] * 100).round(1)
    agg["total_revenue"]         = agg["total_revenue"].round(2)
    agg["avg_revenue"]           = agg["avg_revenue"].round(2)

    agg.sort_values(["yearmonth", "clinic_id", "acquisition_channel"], inplace=True)
    return agg


# ══════════════════════════════════════════════════════════════
#  OUTPUT 5 — agg_cohort_retention
#  Power BI page: Patient behaviour — cohort heatmap
# ══════════════════════════════════════════════════════════════

def make_agg_cohort_retention(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Classic cohort retention matrix: one row per (cohort_month, activity_month).

    Columns:
    - cohort_month_str    first-visit month of the cohort  e.g. "2016-04"
    - activity_month_str  month of activity                e.g. "2016-06"
    - period_number       months since cohort acquisition  (0 = acquisition month)
    - cohort_size         number of patients acquired that month
    - active_patients     patients from that cohort active in activity_month
    - retention_rate      active_patients / cohort_size × 100

    PBI usage: import as a standalone table, use cohort_month_str on Y-axis,
    period_number on X-axis, retention_rate as value → Matrix visual
    with conditional formatting = retention heatmap.
    """
    df = raw.copy()
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["no_show"]        = (df["No-show"] == "Yes").astype(int)
    df["is_first_visit"] = df["is_first_visit"].astype(int)

    # suppress Period timezone warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["period"] = df["AppointmentDay"].dt.to_period("M")

    # cohort = acquisition month (first visit per patient)
    cohort_map = (
        df[df["is_first_visit"] == 1]
            .groupby("PatientId")["period"]
            .min()
            .rename("cohort_period")
    )
    df = df.join(cohort_map, on="PatientId")

    # only attended visits count
    active = df[df["no_show"] == 0].copy()

    agg = (
        active
            .groupby(["cohort_period", "period"])["PatientId"]
            .nunique()
            .reset_index(name="active_patients")
    )

    # cohort size = patients in their period-0 month
    cohort_size = (
        agg[agg["cohort_period"] == agg["period"]]
        [["cohort_period", "active_patients"]]
            .rename(columns={"active_patients": "cohort_size"})
    )
    agg = agg.merge(cohort_size, on="cohort_period", how="left")

    agg["period_number"]      = (agg["period"] - agg["cohort_period"]).apply(lambda x: x.n)
    agg["retention_rate"]     = (agg["active_patients"] / agg["cohort_size"] * 100).round(1)
    agg["cohort_month_str"]   = agg["cohort_period"].astype(str)
    agg["activity_month_str"] = agg["period"].astype(str)

    result = (
        agg[["cohort_month_str", "activity_month_str",
             "period_number", "cohort_size",
             "active_patients", "retention_rate"]]
            .sort_values(["cohort_month_str", "period_number"])
            .reset_index(drop=True)
    )
    return result


# ══════════════════════════════════════════════════════════════
#  SUMMARY PRINT
# ══════════════════════════════════════════════════════════════

def _print_summary(outputs: dict[str, pd.DataFrame]) -> None:
    print("\n── Output summary ───────────────────────────────────────")
    widths = [28, 8, 6]
    print(f"  {'File':<{widths[0]}} {'Rows':>{widths[1]}} {'Cols':>{widths[2]}}")
    print("  " + "-" * (sum(widths) + 4))
    for name, df in outputs.items():
        print(f"  {name+'.csv':<{widths[0]}} {len(df):>{widths[1]},} {df.shape[1]:>{widths[2]}}")

    fact = outputs["fact_appointments"]
    dim_p = outputs["dim_patients"]
    funnel = outputs["agg_monthly_funnel"]

    print("\n── Business metric spot-checks ──────────────────────────")
    attended = fact[fact["no_show"] == 0]
    print(f"  Overall show-up rate     : {(fact['no_show']==0).mean():.1%}")
    print(f"  Total revenue (attended) : ${attended['revenue_usd'].sum():>12,.0f}")
    print(f"  Avg revenue / visit      : ${attended['revenue_usd'].mean():>8,.0f}")
    print(f"  New patient mix          : {fact['is_first_visit'].mean():.1%}")
    print(f"  Champion patients        : {(dim_p['ltv_tier']=='Champion').sum():,}  "
          f"({(dim_p['ltv_tier']=='Champion').mean():.1%})")
    print(f"  Funnel date range        : {funnel['yearmonth'].min()} → {funnel['yearmonth'].max()}")

    print("\n── LTV tier distribution ────────────────────────────────")
    for tier, cnt in dim_p["ltv_tier"].value_counts().items():
        pct = cnt / len(dim_p) * 100
        print(f"  {tier:<12} {cnt:>5,}  ({pct:.1f}%)")

    print("\n── Revenue by clinic ────────────────────────────────────")
    for _, row in (
            fact.groupby("clinic_id")["revenue_usd"]
                    .sum().sort_values(ascending=False).reset_index().iterrows()
    ):
        print(f"  {row['clinic_id']}  ${row['revenue_usd']:>10,.0f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data)
    out_dir  = Path(args.out)

    if not data_dir.exists():
        print(f"[error] Input directory not found: {data_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {data_dir}")
    print(f"Output : {out_dir}\n")
    print("Reading input files...")
    raw, clin, treat = _load_inputs(data_dir)

    print("\nBuilding outputs...")

    outputs: dict[str, pd.DataFrame] = {}

    print("  [1/5] fact_appointments ...")
    outputs["fact_appointments"]   = make_fact_appointments(raw, clin)

    print("  [2/5] dim_patients ...")
    outputs["dim_patients"]        = make_dim_patients(raw)

    print("  [3/5] dim_clinics + treatments ...")
    outputs["dim_clinics"]         = make_dim_clinics(clin)
    outputs["dim_treatments"]      = treat.copy()   # pass-through, already clean

    print("  [4/5] agg_monthly_funnel ...")
    outputs["agg_monthly_funnel"]  = make_agg_monthly_funnel(raw)

    print("  [5/5] agg_cohort_retention ...")
    outputs["agg_cohort_retention"]= make_agg_cohort_retention(raw)

    print("\nWriting CSVs...")
    for name, df in outputs.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name}.csv  →  {path}")

    _print_summary(outputs)
    print("\nDone. Import the files in ./pbi_data into Power BI Desktop.")
    print("Recommended import order: dim_clinics → dim_treatments →")
    print("  dim_patients → fact_appointments → agg tables")


if __name__ == "__main__":
    main()