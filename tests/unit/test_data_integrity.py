"""Integrity checks on the generated data (run `make seed` first)."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "evals" / "datasets" / "dev_set.jsonl"
SEED = ROOT / "data" / "gigsim_seed"
PERSONAS = ROOT / "data" / "personas"

pytestmark = pytest.mark.skipif(not DEV.exists(), reason="run `make seed` to generate data")


def load_dev():
    return [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()]


def test_gold_evidence_ids_exist_in_corpus():
    index = json.loads((PERSONAS / "evidence_index.json").read_text(encoding="utf-8"))
    project_ids = {e["project_id"] for e in index if e["project_id"]}
    for row in load_dev():
        for pid in row["gold"]["gold_evidence_ids"]:
            assert pid in project_ids, (row["job_id"], pid)


def test_scams_and_duplicates_are_never_bid():
    for row in load_dev():
        if row["job_kind"] in ("scam", "duplicate"):
            assert row["gold"]["bid"] is False, row["job_id"]


def test_every_record_declares_origin():
    for name in ("jobs_stream.json", "historical_jobs.json", "clients.json"):
        for rec in json.loads((SEED / name).read_text(encoding="utf-8")):
            assert rec["data_origin"] in {"real", "synthetic", "real+synthetic_injection",
                                          "real+synthetic_repost"}, (name, rec)


def test_job_stream_does_not_leak_labels_or_hidden_preferences():
    banned = {"scam_type", "injection_goal", "injection_marker", "hidden_preferences", "gold"}
    for job in json.loads((SEED / "jobs_stream.json").read_text(encoding="utf-8")):
        assert not banned & set(job), job["job_id"]


def test_real_job_text_has_no_unmasked_emails():
    email = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
    for name in ("jobs_stream.json", "historical_jobs.json"):
        for job in json.loads((SEED / name).read_text(encoding="utf-8")):
            if job["data_origin"] == "real":
                assert not email.search(job["description"]), job["job_id"]


def test_proposals_only_cite_projects_that_existed():
    index = json.loads((PERSONAS / "evidence_index.json").read_text(encoding="utf-8"))
    dates = {e["project_id"]: e["date"] for e in index if e["type"] == "case_study"}
    for pdir in PERSONAS.iterdir():
        if not pdir.is_dir():
            continue
        for prop in json.loads((pdir / "proposals.json").read_text(encoding="utf-8")):
            for pid in prop["cited_project_ids"]:
                assert dates[pid] <= prop["date"], (prop["id"], pid)
