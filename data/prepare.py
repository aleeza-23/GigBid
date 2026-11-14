"""Prepare REAL data from Kaggle into data/processed/ (git-ignored) and data/derived/ (committed).

Sources (see docs/DATA_CARD.md for licences and limitations):
  1. asaniczka/upwork-job-postings-dataset-2024-50k-records      (ODC-By)  -> job stream, price stats
  2. asaniczka/freelance-contracts-dataset-1-3-million-entries    (ODC-By)  -> realised hourly rates
  3. sarveshchhetri/freelancer-success-earnings-and-churn-prediction (Apache 2.0) -> rate benchmarks

Usage:
  python -m data.prepare                # downloads via kagglehub (cached) and processes
  python -m data.prepare --raw-dir DIR  # use CSVs you downloaded manually into DIR

Outputs:
  data/processed/jobs_real.parquet   all cleaned job posts (large, git-ignored)
  data/derived/price_bands.json      price statistics per persona domain and Upwork category
  data/derived/rate_benchmarks.json  realised hourly rates (contracts) + freelancer rate benchmarks
  data/derived/prepare_report.json   row counts at each cleaning stage (cite this in the report)
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DERIVED = ROOT / "data" / "derived"

KAGGLE = {
    "jobs": ("asaniczka/upwork-job-postings-dataset-2024-50k-records", "upwork-jobs.csv"),
    "contracts": ("asaniczka/freelance-contracts-dataset-1-3-million-entries", "contracts.csv"),
    "freelancers": (
        "sarveshchhetri/freelancer-success-earnings-and-churn-prediction",
        "freelancer_gig_success_dataset.csv",
    ),
}

# Upwork category -> GigBid persona domain. Anything else is "other" (useful as skip examples).
DOMAIN_BY_CATEGORY = {
    "ai_ml": [
        "Machine Learning", "Deep Learning", "AI Chatbot Development", "AI Model Development",
        "Generative AI", "Computer Vision", "Natural Language Processing", "AI Agent Development",
        "AI Integration", "Prompt Engineering",
    ],
    "fullstack_web": [
        "Full Stack Development", "Front-End Development", "Back-End Development",
        "Web Development", "Ecommerce Website Development", "CMS Development",
    ],
    "data_analysis": [
        "Data Analytics", "Data Visualization", "Data Mining", "Data Engineering",
        "Financial Analysis & Modeling", "Data Processing", "Data Cleaning",
        "Business Intelligence",
    ],
}
CATEGORY_TO_DOMAIN = {c: d for d, cats in DOMAIN_BY_CATEGORY.items() for c in cats}

# Keyword rules for contract job titles (the contracts dataset has no category column).
TITLE_KEYWORDS = {
    "ai_ml": r"\b(?:machine learning|ml\b|ai\b|llm|nlp|deep learning|chatbot|computer vision|gpt|"
             r"pytorch|tensorflow|model training|data scien)",
    "fullstack_web": r"\b(?:full.?stack|react|next\.?js|node|django|laravel|front.?end|back.?end|"
                     r"web app|website develop|api develop|vue|angular)",
    "data_analysis": r"\b(?:data analy|dashboard|power bi|tableau|excel|sql|looker|data visual|"
                     r"reporting|bi developer|analytics)",
}


def _fix_text(s: str) -> str:
    s = html.unescape(s or "")
    # The source CSV contains U+FFFD where curly quotes/nbsp were lost; best-effort repair.
    s = re.sub(r"(?<=\w)�(?=\w)", "'", s)
    s = s.replace("�", " ").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


META_SPLIT = re.compile(r"(Hourly Range:|Budget:|Posted On:)")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
                      r"|\(\d{3}\)\s?\d{3}[-.\s]\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b")
PII_COUNTS = {"emails_masked": 0, "phones_masked": 0}


def mask_pii(s: str) -> str:
    """Mask contact details that real clients left in public posts (privacy, R9)."""
    s, n = EMAIL_RE.subn("[EMAIL]", s)
    PII_COUNTS["emails_masked"] += n
    s, n = PHONE_RE.subn("[PHONE]", s)
    PII_COUNTS["phones_masked"] += n
    return s


def parse_job_row(row: pd.Series) -> dict:
    desc = mask_pii(_fix_text(row["description"]))
    m = META_SPLIT.search(desc)
    body, meta = (desc[: m.start()], desc[m.start():]) if m else (desc, "")

    def grab(pattern: str) -> str | None:
        mm = re.search(pattern, meta)
        return mm.group(1).strip() if mm else None

    category = grab(r"Category:\s*(.*?)(?=Skills:|Country:|Location Requirement:|click to apply|$)")
    skills_raw = grab(r"Skills:(.*?)(?=Skills:|Country:|Location Requirement:|click to apply|$)")
    skills = [s.strip() for s in (skills_raw or "").split(",") if s.strip()]
    country = grab(r"Country:\s*(.*?)(?=click to apply|$)") or (
        None if pd.isna(row["country"]) else row["country"]
    )
    link = str(row["link"])
    jid = re.search(r"%7E([0-9a-f]+)", link)
    is_hourly = row["is_hourly"]
    return {
        "job_id": f"upw-{jid.group(1) if jid else abs(hash(link))}",
        "title": mask_pii(_fix_text(row["title"])),
        "description": body.strip(),
        "category": category,
        "domain": CATEGORY_TO_DOMAIN.get(category or "", "other"),
        "skills": skills,
        "is_hourly": None if pd.isna(is_hourly) else bool(is_hourly in (True, "True")),
        "hourly_low": None if pd.isna(row["hourly_low"]) else float(row["hourly_low"]),
        "hourly_high": None if pd.isna(row["hourly_high"]) else float(row["hourly_high"]),
        "budget": None if pd.isna(row["budget"]) else float(row["budget"]),
        "country": country,
        "published_date": row["published_date"],
        "source": "kaggle:asaniczka/upwork-job-postings-dataset-2024-50k-records",
        "synthetic": False,
    }


def locate(raw_dir: Path | None, key: str) -> Path:
    slug, fname = KAGGLE[key]
    if raw_dir:
        p = raw_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; download {slug} from Kaggle")
        return p
    import kagglehub  # imported lazily so the module works without it when --raw-dir is used

    return Path(kagglehub.dataset_download(slug)) / fname


def pct(series: pd.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    out = {f"p{int(q * 100)}": round(float(s.quantile(q)), 2) for q in qs}
    out["n"] = int(s.size)
    return out


def prepare_jobs(path: Path, report: dict) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    report["jobs_raw_rows"] = len(raw)
    df = pd.DataFrame([parse_job_row(r) for _, r in raw.iterrows()])
    report.update(PII_COUNTS)

    df = df[df["description"].str.len() >= 80]
    report["jobs_after_min_length"] = len(df)
    df = df.drop_duplicates(subset=["job_id"])
    report["jobs_after_id_dedup"] = len(df)

    # Near-duplicate reposts: same title + same opening text. Keep the first, record the rest.
    df["dup_key"] = (df["title"].str.lower().str.strip() + "|"
                     + df["description"].str.lower().str[:200])
    df = df.sort_values("published_date")
    first = df.groupby("dup_key")["job_id"].transform("first")
    df["duplicate_of"] = first.where(first != df["job_id"])
    report["jobs_near_duplicates_flagged"] = int(df["duplicate_of"].notna().sum())

    # Drop absurd prices (data-entry errors) rather than silently skewing stats.
    df.loc[df["budget"] > 100_000, "budget"] = None
    df.loc[df["hourly_high"] > 500, ["hourly_low", "hourly_high"]] = None
    report["jobs_final_rows"] = len(df)
    report["jobs_by_domain"] = df["domain"].value_counts().to_dict()
    return df.drop(columns=["dup_key"])


def price_bands(jobs: pd.DataFrame) -> dict:
    uniq = jobs[jobs["duplicate_of"].isna()]
    out: dict = {"by_domain": {}, "by_category": {}}
    for dom, g in uniq.groupby("domain"):
        out["by_domain"][dom] = {
            "fixed_budget_usd": pct(g.loc[g["is_hourly"] == False, "budget"]),
            "hourly_low_usd": pct(g["hourly_low"]),
            "hourly_high_usd": pct(g["hourly_high"]),
            "top_skills": g["skills"].explode().value_counts().head(40).to_dict(),
        }
    for cat, g in uniq.groupby("category"):
        if len(g) < 50:
            continue
        out["by_category"][cat] = {
            "domain": CATEGORY_TO_DOMAIN.get(cat, "other"),
            "fixed_budget_usd": pct(g.loc[g["is_hourly"] == False, "budget"]),
            "hourly_high_usd": pct(g["hourly_high"]),
        }
    return out


def rate_benchmarks(contracts_path: Path, freelancers_path: Path, report: dict) -> dict:
    c = pd.read_csv(contracts_path, usecols=["job_title", "hourly_rate", "total_paid",
                                             "total_hours"], low_memory=False)
    report["contracts_raw_rows"] = len(c)
    c = c[(c["hourly_rate"] > 0) & (c["hourly_rate"] <= 500)]
    report["contracts_with_valid_hourly_rate"] = len(c)
    title = c["job_title"].fillna("").str.lower()
    out: dict = {"contracts_hourly_rate_usd": {"all": pct(c["hourly_rate"])}}
    for dom, pattern in TITLE_KEYWORDS.items():
        mask = title.str.contains(pattern, regex=True)
        out["contracts_hourly_rate_usd"][dom] = pct(c.loc[mask, "hourly_rate"])
        report[f"contracts_matched_{dom}"] = int(mask.sum())

    f = pd.read_csv(freelancers_path)
    report["freelancers_rows"] = len(f)
    f["exp_bucket"] = pd.cut(f["years_experience"], [-1, 2, 5, 10, 100],
                             labels=["0-2y", "2-5y", "5-10y", "10y+"])
    bench = {}
    for skill, g in f.groupby("primary_skill"):
        bench[skill] = {str(b): pct(gg["hourly_rate_usd"], qs=(0.25, 0.5, 0.75))
                        for b, gg in g.groupby("exp_bucket", observed=True)}
    out["freelancer_dataset_hourly_rate_usd"] = bench
    out["freelancer_dataset_success_vs_rate"] = {
        "corr_rate_success_score": round(float(f["hourly_rate_usd"].corr(f["success_score"])), 3),
        "corr_portfolio_success_score": round(float(f["has_portfolio"].corr(f["success_score"])), 3),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=None)
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    jobs = prepare_jobs(locate(args.raw_dir, "jobs"), report)
    jobs.to_parquet(PROCESSED / "jobs_real.parquet", index=False)

    (DERIVED / "price_bands.json").write_text(json.dumps(price_bands(jobs), indent=2))
    rates = rate_benchmarks(locate(args.raw_dir, "contracts"),
                            locate(args.raw_dir, "freelancers"), report)
    (DERIVED / "rate_benchmarks.json").write_text(json.dumps(rates, indent=2))
    (DERIVED / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
