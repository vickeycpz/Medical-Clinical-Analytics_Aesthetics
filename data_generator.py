"""
MedSpa Partners — Portfolio Data Generator
===========================================
Generates five relational CSV tables that simulate a network of 6 medical
aesthetics clinics from Jan 2022 to Dec 2024.

Output tables
─────────────
  clinics.csv           6 rows   — clinic master data
  treatments.csv       10 rows   — treatment catalogue with base prices
  patients.csv       3,000 rows  — patient demographics + acquisition channel
  visits.csv         ~9,000 rows — individual treatment visits with revenue
  marketing_funnel.csv 1,296 rows — monthly leads / consultations / conversions
                                    per clinic per channel

Usage
─────
  python data_generator.py                  # writes to ./medspa_data/
  python data_generator.py --out /my/path   # custom output directory
  python data_generator.py --seed 99        # different random seed

Requirements: pandas, numpy  (pip install pandas numpy)
"""

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate MedSpa demo datasets")
    p.add_argument("--out",        default="./medspa_data", help="Output directory")
    p.add_argument("--patients",   type=int, default=3000,  help="Number of patients")
    p.add_argument("--seed",       type=int, default=42,    help="Random seed")
    return p.parse_args()


# ── REFERENCE DATA ────────────────────────────────────────────────────────────

CLINIC_DATA = [
    {"clinic_id": "C01", "clinic_name": "Glow & Co – Midtown",    "city": "New York",    "state": "NY", "tier": "premium",  "opened_date": "2019-03-15", "capacity_chairs": 6, "monthly_rent_usd": 22000},
    {"clinic_id": "C02", "clinic_name": "Pure Skin Westside",      "city": "Los Angeles", "state": "CA", "tier": "premium",  "opened_date": "2018-07-01", "capacity_chairs": 8, "monthly_rent_usd": 28000},
    {"clinic_id": "C03", "clinic_name": "Radiance Medical Spa",    "city": "Chicago",     "state": "IL", "tier": "standard", "opened_date": "2020-01-10", "capacity_chairs": 5, "monthly_rent_usd": 14000},
    {"clinic_id": "C04", "clinic_name": "Luminary Aesthetics",     "city": "Houston",     "state": "TX", "tier": "standard", "opened_date": "2019-11-20", "capacity_chairs": 5, "monthly_rent_usd": 12000},
    {"clinic_id": "C05", "clinic_name": "Revive MedSpa Austin",    "city": "Austin",      "state": "TX", "tier": "standard", "opened_date": "2021-05-05", "capacity_chairs": 4, "monthly_rent_usd":  9500},
    {"clinic_id": "C06", "clinic_name": "Clarity Skin & Wellness", "city": "Miami",       "state": "FL", "tier": "premium",  "opened_date": "2018-09-30", "capacity_chairs": 7, "monthly_rent_usd": 20000},
]

TREATMENT_DATA = [
    {"treatment_id": "T01", "treatment_name": "Botox",               "category": "injectables",    "base_price": 450, "duration_min": 30},
    {"treatment_id": "T02", "treatment_name": "Dermal Fillers",      "category": "injectables",    "base_price": 650, "duration_min": 45},
    {"treatment_id": "T03", "treatment_name": "Laser Hair Removal",  "category": "laser",          "base_price": 300, "duration_min": 60},
    {"treatment_id": "T04", "treatment_name": "IPL Photofacial",     "category": "laser",          "base_price": 350, "duration_min": 60},
    {"treatment_id": "T05", "treatment_name": "Chemical Peel",       "category": "skin_care",      "base_price": 180, "duration_min": 45},
    {"treatment_id": "T06", "treatment_name": "Microneedling",       "category": "skin_care",      "base_price": 280, "duration_min": 60},
    {"treatment_id": "T07", "treatment_name": "HydraFacial",         "category": "skin_care",      "base_price": 200, "duration_min": 75},
    {"treatment_id": "T08", "treatment_name": "CoolSculpting",       "category": "body_contouring","base_price": 800, "duration_min": 90},
    {"treatment_id": "T09", "treatment_name": "Kybella",             "category": "injectables",    "base_price": 550, "duration_min": 30},
    {"treatment_id": "T10", "treatment_name": "Morpheus8",           "category": "laser",          "base_price": 900, "duration_min": 90},
]

CHANNELS        = ["Google Ads", "Instagram", "Facebook", "Referral", "Walk-in", "Email Campaign"]
CHANNEL_WEIGHTS = [0.28, 0.22, 0.12, 0.20, 0.10, 0.08]

# Conversion rates: lead → consultation booked
CHANNEL_CONSULT_RATE = {
    "Google Ads":     0.22,
    "Instagram":      0.18,
    "Facebook":       0.15,
    "Referral":       0.55,
    "Walk-in":        0.70,
    "Email Campaign": 0.25,
}
# Conversion rates: consultation → new patient
CHANNEL_CLOSE_RATE = {
    "Google Ads":     0.40,
    "Instagram":      0.38,
    "Facebook":       0.35,
    "Referral":       0.68,
    "Walk-in":        0.75,
    "Email Campaign": 0.45,
}
# Average cost per lead (USD); Walk-in = organic, no spend
CHANNEL_CPL = {
    "Google Ads":     45,
    "Instagram":      35,
    "Facebook":       28,
    "Referral":        8,
    "Walk-in":         0,
    "Email Campaign": 12,
}

# Realistic US first/last names (avoids Faker dependency)
FIRST_NAMES = [
    "Emma","Olivia","Sophia","Ava","Isabella","Mia","Charlotte","Amelia",
    "Harper","Evelyn","Chloe","Grace","Lily","Zoe","Natalie","Hannah",
    "Aria","Scarlett","Victoria","Luna","James","Liam","Noah","Oliver",
    "William","Elijah","Benjamin","Lucas","Mason","Logan","Ethan","Daniel",
    "Henry","Sebastian","Michael","Alexander","David","Joseph","Samuel","Ryan",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Hall","Allen","Young","King","Wright","Scott","Torres","Nguyen","Hill",
]

START_DATE = date(2022, 1, 1)
END_DATE   = date(2024, 12, 31)

# Monthly seasonality index (aesthetics peaks pre-summer and pre-holiday)
SEASONAL_INDEX = {
    1: 0.75, 2: 0.80, 3: 0.95, 4: 1.10, 5: 1.15, 6: 1.05,
    7: 0.90, 8: 0.88, 9: 1.00, 10: 1.10, 11: 1.20, 12: 1.25,
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def rand_dates(start: date, end: date, n: int) -> list[date]:
    """Return n random dates uniformly distributed between start and end."""
    delta = (end - start).days
    offsets = np.random.randint(0, delta, n)
    return [start + timedelta(days=int(d)) for d in offsets]


def make_emails(first_names, last_names) -> list[str]:
    domains  = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "hotmail.com"]
    dom_pick = np.random.choice(domains, len(first_names))
    suffixes = np.random.randint(10, 999, len(first_names))
    return [
        f"{fn.lower()}.{ln.lower()}{sx}@{dm}"
        for fn, ln, sx, dm in zip(first_names, last_names, suffixes, dom_pick)
    ]


def next_month(d: date) -> date:
    """Return the first day of the following month."""
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


# ── TABLE BUILDERS ────────────────────────────────────────────────────────────

def build_clinics() -> pd.DataFrame:
    return pd.DataFrame(CLINIC_DATA)


def build_treatments() -> pd.DataFrame:
    return pd.DataFrame(TREATMENT_DATA)


def build_patients(n: int) -> pd.DataFrame:
    first_names = np.random.choice(FIRST_NAMES, n)
    last_names  = np.random.choice(LAST_NAMES,  n)

    # Premium clinics attract proportionally more patients
    clinic_ids = [c["clinic_id"] for c in CLINIC_DATA]
    clinic_w   = np.array([1.6, 1.8, 1.0, 1.0, 0.9, 1.7])
    clinic_w  /= clinic_w.sum()
    assigned_clinics = np.random.choice(clinic_ids, n, p=clinic_w)

    # Sort by first-visit date so patient IDs are roughly chronological
    first_visits = rand_dates(START_DATE, END_DATE, n)
    first_visits.sort()

    clinic_lookup = {c["clinic_id"]: c for c in CLINIC_DATA}

    return pd.DataFrame({
        "patient_id":          [f"P{i+1:05d}" for i in range(n)],
        "first_name":          first_names,
        "last_name":           last_names,
        "email":               make_emails(first_names, last_names),
        "age":                 np.random.normal(38, 10, n).clip(21, 70).astype(int),
        "gender":              np.random.choice(["F", "M", "Non-binary"], n, p=[0.74, 0.23, 0.03]),
        "clinic_id":           assigned_clinics,
        "city":                [clinic_lookup[c]["city"]  for c in assigned_clinics],
        "state":               [clinic_lookup[c]["state"] for c in assigned_clinics],
        "acquisition_channel": np.random.choice(CHANNELS, n, p=CHANNEL_WEIGHTS),
        "first_visit_date":    [str(d) for d in first_visits],
    })


def build_visits(df_patients: pd.DataFrame) -> pd.DataFrame:
    """
    Generate visit history for every patient.

    Behavioural rules:
    - Patients have 1–8 visits; premium-clinic patients skew higher.
    - First treatment is random; subsequent visits have 70% chance of
      staying within the same treatment category (treatment loyalty).
    - Visit gaps follow an exponential distribution (~60-day mean).
    - Revenue = base price × clinic-tier multiplier × small noise.
    """
    treatment_ids     = [t["treatment_id"] for t in TREATMENT_DATA]
    treatment_prices  = {t["treatment_id"]: t["base_price"]  for t in TREATMENT_DATA}
    treatment_cats    = {t["treatment_id"]: t["category"]    for t in TREATMENT_DATA}
    premium_clinic_ids = {c["clinic_id"] for c in CLINIC_DATA if c["tier"] == "premium"}

    category_treatments: dict[str, list[str]] = {}
    for t in TREATMENT_DATA:
        category_treatments.setdefault(t["category"], []).append(t["treatment_id"])

    rows: list[dict] = []

    for _, pat in df_patients.iterrows():
        is_premium = pat["clinic_id"] in premium_clinic_ids
        n_visits   = np.random.choice(range(1, 9), p=[0.25, 0.22, 0.18, 0.13, 0.09, 0.07, 0.04, 0.02])
        if is_premium:
            n_visits = min(8, n_visits + np.random.randint(0, 2))

        first_tx      = np.random.choice(treatment_ids)
        preferred_cat = treatment_cats[first_tx]
        visit_date    = date.fromisoformat(pat["first_visit_date"])

        for v in range(n_visits):
            if v == 0:
                tx_id = first_tx
            elif np.random.random() < 0.70:
                tx_id = np.random.choice(category_treatments[preferred_cat])
            else:
                tx_id = np.random.choice(treatment_ids)

            price_mult = np.random.normal(1.0, 0.08)
            if is_premium:
                price_mult *= 1.15
            revenue = round(treatment_prices[tx_id] * max(0.70, price_mult), 2)

            rows.append({
                "visit_id":       f"V{len(rows)+1:06d}",
                "patient_id":     pat["patient_id"],
                "clinic_id":      pat["clinic_id"],
                "treatment_id":   tx_id,
                "visit_date":     str(visit_date),
                "revenue_usd":    revenue,
                "satisfaction":   np.random.choice([3, 4, 5], p=[0.10, 0.45, 0.45]),
                "is_first_visit": v == 0,
            })

            gap = int(np.random.exponential(scale=60)) + 14
            visit_date = visit_date + timedelta(days=gap)
            if visit_date > END_DATE:
                break

    return pd.DataFrame(rows)


def build_marketing_funnel() -> pd.DataFrame:
    """
    Monthly funnel metrics (leads → consultations → new patients) per clinic
    per channel, with realistic seasonality, YoY growth, and ad spend.
    """
    rows: list[dict] = []
    current = START_DATE.replace(day=1)
    end_mo  = END_DATE.replace(day=1)

    while current <= end_mo:
        month_str  = current.strftime("%Y-%m")
        seasonal   = SEASONAL_INDEX[current.month]
        yoy_growth = 1.0 + 0.18 * ((current.year - 2022) + (current.month - 1) / 12)

        for clinic in CLINIC_DATA:
            base_leads = np.random.randint(60, 120) if clinic["tier"] == "premium" else np.random.randint(30, 70)

            for ch in CHANNELS:
                ch_weight     = CHANNEL_WEIGHTS[CHANNELS.index(ch)]
                leads         = int(base_leads * ch_weight * seasonal * yoy_growth * np.random.normal(1, 0.12))
                leads         = max(0, leads)

                consult_rate  = CHANNEL_CONSULT_RATE[ch] * np.random.normal(1, 0.10)
                close_rate    = CHANNEL_CLOSE_RATE[ch]   * np.random.normal(1, 0.10)
                consultations = max(0, int(leads * consult_rate))
                new_patients  = max(0, int(consultations * close_rate))
                ad_spend      = max(0.0, round(leads * CHANNEL_CPL[ch] * np.random.normal(1, 0.08), 2))

                rows.append({
                    "month":         month_str,
                    "clinic_id":     clinic["clinic_id"],
                    "channel":       ch,
                    "leads":         leads,
                    "consultations": consultations,
                    "new_patients":  new_patients,
                    "ad_spend_usd":  ad_spend,
                })

        current = next_month(current)

    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("Generating MedSpa demo data...")
    print(f"  Seed       : {args.seed}")
    print(f"  Patients   : {args.patients:,}")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Output dir : {args.out}\n")

    tables = {
        "clinics":           build_clinics(),
        "treatments":        build_treatments(),
        "patients":          build_patients(args.patients),
    }
    # visits depends on patients
    tables["visits"]           = build_visits(tables["patients"])
    tables["marketing_funnel"] = build_marketing_funnel()

    for name, df in tables.items():
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name}.csv".ljust(30) + f"{len(df):>6,} rows  →  {path}")

    # ── sanity checks ────────────────────────────────────────────────────────
    visits  = tables["visits"]
    funnel  = tables["marketing_funnel"]
    patients = tables["patients"]

    print("\n── Sanity checks ────────────────────────────────────")
    print(f"  Avg visits / patient : {len(visits) / args.patients:.2f}")
    print(f"  Total revenue        : ${visits['revenue_usd'].sum():>12,.0f}")
    print(f"  Funnel months        : {funnel['month'].nunique()}")
    print(f"  Avg lead→consult     : {(funnel['consultations'].sum() / funnel['leads'].sum()):.1%}")
    print(f"  Avg consult→patient  : {(funnel['new_patients'].sum() / funnel['consultations'].sum()):.1%}")
    print()
    print("  Visits per clinic:")
    for row in visits.groupby("clinic_id").size().items():
        print(f"    {row[0]}  {row[1]:,}")
    print()
    print("  Patients per acquisition channel:")
    for row in patients["acquisition_channel"].value_counts().items():
        print(f"    {row[0]:<20}  {row[1]:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()