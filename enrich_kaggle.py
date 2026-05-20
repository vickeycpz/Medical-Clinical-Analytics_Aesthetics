"""
Medical Aesthetics — Kaggle 数据增强脚本
======================================
将 Kaggle「Medical Appointment No Shows」数据集的 14 个原始字段，
扩充为 25 个包含美容医疗业务场景的字段，作为后续分析和 Power BI
仪表板的底层数据。

新增字段（11 个）
─────────────────────────────────────────────────────────
  clinic_id            诊所编号（哈希稳定，同一患者跨行一致）
  clinic_name          诊所名称
  clinic_tier          premium / standard
  treatment_id         治疗项目编号
  treatment_name       治疗项目名称
  treatment_category   injectables / laser / skin_care / body_contouring
  revenue_usd          本次实收金额（爽约=0，premium 诊所溢价 15%）
  acquisition_channel  获客渠道（受 Gender + Age + Scholarship 影响）
  is_first_visit       是否首次就诊（按 PatientId 去重判断）
  satisfaction_score   满意度 3-5（爽约记为 NaN）
  booking_lead_days    预约提前天数（由原始日期字段直接计算）

使用方法
─────────────────────────────────────────────────────────
  # 依赖：pip install pandas numpy faker
  # Faker 可选；脚本会自动降级到内置 random，效果相同

  python enrich_kaggle.py                              # 默认读写当前目录
  python enrich_kaggle.py --input KaggleV2Medium.csv  # 指定输入文件
  python enrich_kaggle.py --out   ./medspa_data       # 指定输出目录
  python enrich_kaggle.py --seed  99                  # 固定随机种子

输出
─────────────────────────────────────────────────────────
  appointments_enriched.csv   完整增强数据（原始 + 新增字段）
  clinics.csv                 诊所参考表
  treatments.csv              治疗项目参考表
"""

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Tuple,List

import numpy as np
import pandas as pd

# ── Faker 可选降级 ─────────────────────────────────────────────────────────────
try:
    from faker import Faker
    _fake = Faker("en_US")
    Faker.seed(42)
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    print("[info] Faker 未安装，使用内置 random 生成姓名/邮箱，效果相同。")
    print("       可选安装：pip install faker\n")


# ══════════════════════════════════════════════════════════════════════════════
#  参考数据
# ══════════════════════════════════════════════════════════════════════════════

CLINIC_DATA = [
    {"clinic_id": "C01", "clinic_name": "Glow & Co – Midtown",    "city": "New York",    "state": "NY", "tier": "premium",  "monthly_rent_usd": 22000, "capacity_chairs": 6},
    {"clinic_id": "C02", "clinic_name": "Pure Skin Westside",      "city": "Los Angeles", "state": "CA", "tier": "premium",  "monthly_rent_usd": 28000, "capacity_chairs": 8},
    {"clinic_id": "C03", "clinic_name": "Radiance Medical Spa",    "city": "Chicago",     "state": "IL", "tier": "standard", "monthly_rent_usd": 14000, "capacity_chairs": 5},
    {"clinic_id": "C04", "clinic_name": "Luminary Aesthetics",     "city": "Houston",     "state": "TX", "tier": "standard", "monthly_rent_usd": 12000, "capacity_chairs": 5},
    {"clinic_id": "C05", "clinic_name": "Revive MedSpa Austin",    "city": "Austin",      "state": "TX", "tier": "standard", "monthly_rent_usd":  9500, "capacity_chairs": 4},
    {"clinic_id": "C06", "clinic_name": "Clarity Skin & Wellness", "city": "Miami",       "state": "FL", "tier": "premium",  "monthly_rent_usd": 20000, "capacity_chairs": 7},
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

CHANNELS = ["Google Ads", "Instagram", "Facebook", "Referral", "Walk-in", "Email Campaign"]

# 年龄段 → 治疗项目偏好权重（10 项治疗对应 10 个权重，顺序与 TREATMENT_DATA 一致）
# 设计依据：
#   18-29  偏向 Laser Hair Removal、Chemical Peel、HydraFacial（基础护肤）
#   30-44  Botox 开始出现，Fillers 增加（抗衰需求启动）
#   45-59  Botox + Fillers + Morpheus8 主导（深度抗衰）
#   60+    高价项目占比上升（Morpheus8、CoolSculpting）
AGE_TREATMENT_WEIGHTS = {
    "18-29": [0.10, 0.05, 0.22, 0.12, 0.20, 0.14, 0.12, 0.02, 0.02, 0.01],
    "30-44": [0.25, 0.18, 0.12, 0.10, 0.10, 0.10, 0.08, 0.04, 0.02, 0.01],
    "45-59": [0.22, 0.20, 0.06, 0.10, 0.08, 0.08, 0.05, 0.08, 0.06, 0.07],
    "60+":   [0.18, 0.16, 0.03, 0.08, 0.08, 0.06, 0.05, 0.10, 0.08, 0.18],
}

# 获客渠道权重（默认 / 年轻女性 / 低收入）
CHANNEL_WEIGHTS_DEFAULT   = [0.28, 0.22, 0.12, 0.20, 0.10, 0.08]
CHANNEL_WEIGHTS_YOUNG_F   = [0.18, 0.38, 0.12, 0.16, 0.08, 0.08]  # Instagram 主导
CHANNEL_WEIGHTS_LOW_INCOME= [0.15, 0.12, 0.10, 0.30, 0.25, 0.08]  # 口碑 + Walk-in

# 备用姓名池（无 Faker 时使用）
_FIRST_NAMES = [
    "Emma","Olivia","Sophia","Ava","Isabella","Mia","Charlotte","Amelia",
    "Harper","Evelyn","James","Liam","Noah","Oliver","William","Elijah",
    "Chloe","Grace","Lily","Zoe","Natalie","Hannah","Aria","Scarlett",
    "Victoria","Luna","Ethan","Daniel","Henry","Michael","David","Ryan",
]
_LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
]
_DOMAINS = ["gmail.com","yahoo.com","outlook.com","icloud.com","hotmail.com"]


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _age_bucket(age: int) -> str:
    """将年龄映射到治疗偏好分层。"""
    if age < 30:  return "18-29"
    if age < 45:  return "30-44"
    if age < 60:  return "45-59"
    return "60+"


def _stable_hash_choice(key: str, items: list, weights: list):
    """
    用 PatientId 的 MD5 哈希做稳定随机选择。
    同一个 PatientId 无论出现多少行，分配的诊所永远一致。
    """
    h   = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    tot = sum(weights)
    r   = (h % 100_000) / 100_000 * tot
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if r <= cum:
            return item
    return items[-1]


def _fake_name_email(gender: str) -> Tuple[str, str, str]:
    """生成姓名和邮箱；有 Faker 用 Faker，否则用内置池。"""
    if HAS_FAKER:
        if gender == "F":
            first = _fake.first_name_female()
        else:
            first = _fake.first_name_male()
        last  = _fake.last_name()
        email = _fake.email()
    else:
        first = random.choice(_FIRST_NAMES)
        last  = random.choice(_LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{random.randint(10,999)}@{random.choice(_DOMAINS)}"
    return first, last, email


def _channel_weights(row: pd.Series) -> List[float]:
    """根据人口特征返回对应的获客渠道权重。"""
    if row["Gender"] == "F" and row["Age"] < 40:
        return CHANNEL_WEIGHTS_YOUNG_F
    if row["Scholarship"] == 1:
        return CHANNEL_WEIGHTS_LOW_INCOME
    return CHANNEL_WEIGHTS_DEFAULT


# ══════════════════════════════════════════════════════════════════════════════
#  核心增强函数
# ══════════════════════════════════════════════════════════════════════════════

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    接收原始 Kaggle DataFrame，返回增强后的 DataFrame。
    所有新字段都有业务逻辑依据，不是纯随机。
    """
    n = len(df)
    rng = np.random.default_rng(seed=42)

    # ── 准备查找表 ────────────────────────────────────────────────────────────
    clinic_ids    = [c["clinic_id"]   for c in CLINIC_DATA]
    clinic_w      = [1.6, 1.8, 1.0, 1.0, 0.9, 1.7]   # premium 诊所引流更多
    clinic_lookup = {c["clinic_id"]: c for c in CLINIC_DATA}

    tx_ids    = [t["treatment_id"]   for t in TREATMENT_DATA]
    tx_lookup = {t["treatment_id"]: t for t in TREATMENT_DATA}

    # ── 1. 诊所分配（PatientId 哈希 → 稳定）────────────────────────────────────
    #    同一患者若在 Kaggle 数据里出现多行（多次预约），诊所保持一致
    df["clinic_id"]   = df["PatientId"].astype(str).apply(
        lambda pid: _stable_hash_choice(pid, clinic_ids, clinic_w)
    )
    df["clinic_name"] = df["clinic_id"].map({c["clinic_id"]: c["clinic_name"] for c in CLINIC_DATA})
    df["clinic_tier"] = df["clinic_id"].map({c["clinic_id"]: c["tier"]        for c in CLINIC_DATA})

    # ── 2. 治疗类型（年龄段偏好权重随机）────────────────────────────────────────
    def _pick_tx(age: int) -> dict:
        w = AGE_TREATMENT_WEIGHTS[_age_bucket(int(age))]
        return random.choices(TREATMENT_DATA, weights=w, k=1)[0]

    tx_series              = df["Age"].apply(_pick_tx)
    df["treatment_id"]     = tx_series.apply(lambda t: t["treatment_id"])
    df["treatment_name"]   = tx_series.apply(lambda t: t["treatment_name"])
    df["treatment_category"] = tx_series.apply(lambda t: t["category"])

    # ── 3. 客单价 ─────────────────────────────────────────────────────────────
    #    爽约(No-show=Yes) → 取消费 0
    #    premium 诊所溢价 15%，价格有 ±8% 自然波动
    base_prices = tx_series.apply(lambda t: t["base_price"]).values
    tier_mult   = np.where(df["clinic_tier"] == "premium", 1.15, 1.0)
    noise       = rng.normal(1.0, 0.08, n).clip(0.75, 1.30)
    raw_revenue = np.round(base_prices * tier_mult * noise, 2)
    df["revenue_usd"] = np.where(df["No-show"] == "Yes", 0.0, raw_revenue)

    # ── 4. 获客渠道（人口特征驱动权重）─────────────────────────────────────────
    df["acquisition_channel"] = df.apply(
        lambda row: random.choices(CHANNELS, weights=_channel_weights(row), k=1)[0],
        axis=1
    )

    # ── 5. 是否首次就诊（PatientId 去重，第一次出现 = 首诊）─────────────────────
    df["is_first_visit"] = ~df.duplicated(subset="PatientId", keep="first")

    # ── 6. 满意度（仅到诊患者；3-5 分，爽约记 NaN）───────────────────────────
    sat_arr = rng.choice([3, 4, 5], size=n, p=[0.10, 0.45, 0.45]).astype(float)
    df["satisfaction_score"] = np.where(df["No-show"] == "Yes", np.nan, sat_arr)

    # ── 7. 预约提前天数（0 成本，直接计算）──────────────────────────────────────
    appt   = pd.to_datetime(df["AppointmentDay"], errors="coerce")
    sched  = pd.to_datetime(df["ScheduledDay"],   errors="coerce")
    df["booking_lead_days"] = (appt - sched).dt.days.abs().fillna(0).astype(int)

    # ── 8. 姓名 & 邮箱（Faker 优先，否则内置池）─────────────────────────────────
    names  = df["Gender"].apply(lambda g: _fake_name_email(g))
    df["first_name"] = names.apply(lambda x: x[0])
    df["last_name"]  = names.apply(lambda x: x[1])
    df["email"]      = names.apply(lambda x: x[2])

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLI & 主流程
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Enrich Kaggle No-Show data with MedSpa fields")
    p.add_argument("--input", default="KaggleV2Medium.csv",
                   help="Kaggle CSV 文件路径（默认：KaggleV2Medium.csv）")
    p.add_argument("--out",   default="./medspa_data",
                   help="输出目录（默认：./medspa_data）")
    p.add_argument("--seed",  type=int, default=42, help="随机种子")
    return p.parse_args()


def _print_summary(df: pd.DataFrame):
    """打印业务指标速览，方便验证数据质量。"""
    showed = df[df["No-show"] == "No"]
    print("\n── 数据质量速览 ────────────────────────────────────────")
    print(f"  总预约数       : {len(df):,}")
    print(f"  到诊率         : {(df['No-show']=='No').mean():.1%}")
    print(f"  首诊占比       : {df['is_first_visit'].mean():.1%}")
    print(f"  平均客单价     : ${showed['revenue_usd'].mean():.0f}")
    print(f"  总营收(到诊)   : ${showed['revenue_usd'].sum():,.0f}")
    print(f"  平均提前天数   : {df['booking_lead_days'].mean():.1f} 天")
    print()
    print("  诊所营收分布：")
    for cid, rev in showed.groupby("clinic_id")["revenue_usd"].sum().items():
        tier = df[df["clinic_id"]==cid]["clinic_tier"].iloc[0]
        print(f"    {cid}  ({tier:8s})  ${rev:>10,.0f}")
    print()
    print("  治疗类别分布：")
    for cat, cnt in df["treatment_category"].value_counts().items():
        print(f"    {cat:<20}  {cnt:,}")
    print()
    print("  获客渠道分布：")
    for ch, cnt in df["acquisition_channel"].value_counts().items():
        print(f"    {ch:<20}  {cnt:,}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 读取 Kaggle CSV ───────────────────────────────────────────────────────
    if not Path(args.input).exists():
        print(f"[错误] 找不到输入文件：{args.input}")
        print("  请先下载 Kaggle 数据集并将 CSV 放到同目录，或用 --input 指定路径。")
        print("  下载地址：https://www.kaggle.com/datasets/joniarroba/noshowappointments")
        sys.exit(1)

    print(f"读取 Kaggle 数据：{args.input}")
    df_raw = pd.read_csv(args.input)
    print(f"原始数据：{df_raw.shape[0]:,} 行 × {df_raw.shape[1]} 列")

    # 列名标准化（原始文件有时大小写不一致）
    df_raw.columns = df_raw.columns.str.strip()

    # ── 增强 ──────────────────────────────────────────────────────────────────
    print("增强中，请稍候...")
    df_enriched = enrich(df_raw.copy())
    print(f"增强后：{df_enriched.shape[0]:,} 行 × {df_enriched.shape[1]} 列")

    # ── 输出 ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    # 主表
    out_main = os.path.join(args.out, "appointments_enriched.csv")
    df_enriched.to_csv(out_main, index=False)
    print(f"\n✓ appointments_enriched.csv  → {out_main}")

    # 参考表（方便后续 SQL JOIN 和 Power BI 关系建模）
    out_clinics = os.path.join(args.out, "clinics.csv")
    pd.DataFrame(CLINIC_DATA).to_csv(out_clinics, index=False)
    print(f"✓ clinics.csv                → {out_clinics}")

    out_tx = os.path.join(args.out, "treatments.csv")
    pd.DataFrame(TREATMENT_DATA).to_csv(out_tx, index=False)
    print(f"✓ treatments.csv             → {out_tx}")

    _print_summary(df_enriched)
    print("\n完成。")


if __name__ == "__main__":
    main()