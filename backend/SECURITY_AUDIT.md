# Dependency Security Audit (ISO 27001 A.8.8)

**Date:** 2026-10-27 — T-015, agents `test-impl-1` + `security-fix-{1,2,3,4}`
**Scope:** `backend/requirements.txt` + `frontend/package.json`
**Methodology:** `pip-audit --vulnerability-service osv` (back-end), `npm audit`
(front-end). Both tooled from the pinned lockfile.

## Status

| Surface | Before | After | Fixed in-tree |
| --- | --- | --- | --- |
| `pip-audit -r requirements.txt` | 47 findings in 8 packages | **0** (declared deps clean) | 47 in 13 packages |
| `pip-audit` (installed env) | 47 + 12 pip-tooling | 12 in `pip` 24.0 only (no CVE, tooling) | 47 in 13 packages |
| `npm audit` | 5 (2 low, 2 high, 1 critical) | **0** | 5 |

## Backend changes applied

| Package | From | To | Advisories closed |
| --- | --- | --- | --- |
| `fastapi` | 0.115.6 | 0.140.0 | enables Starlette>=1.3.1 |
| `starlette` | (transitive 0.41.3) | pinned `>=1.3.1,<2.0.0` | PYSEC-2026-{161,248,249,1941,1942,2280,2281} |
| `python-multipart` | 0.0.19 | 0.0.31 | PYSEC-2026-{1852,3036-3041} |
| `pymongo[srv]` | 4.10.1 | 4.18.2 (bare) | PYSEC-2026-{4294,4295,4296,4297} |
| `motor` | 3.6.0 | 3.7.1 | **broken resolver** (requested pymongo==4.10.1, motor pinned <4.10) |
| `requests` | 2.32.3 | 2.33.0 | PYSEC-2026-{1872,2275} |
| `pdfplumber` | 0.11.4 | 0.11.10 | pulls pdfminer-six>=20251230 → PYSEC-2026-{1761,1762} |
| `pillow` | 11.0.0 | `>=12.2.0` | compat with pdfplumber 0.11.9+ |
| `langchain` | 0.3.11 | **1.3.9** | Phase 2 — pulls langchain-core>=1.2.11; closes PYSEC-2026-{373,1518,2562,2563,2564}. `app/rag/chunker.py` import path unchanged; verified with 35/35 RAG unit tests green under 1.3.9 |
| `langchain-text-splitters` | 0.3.3 | **1.1.2** | Phase 2 — full fix for PYSEC-2026-77; `RecursiveCharacterTextSplitter` ctor + `.split_text()` source-compatible, smoke-tested on representative input |
| `langsmith` | 0.2.11 (transitive) | pinned `>=0.8.18,<1.0.0` | **CVE-2026-59152**, PYSEC-2026-{2555,2582,2583} |
| `pytest` | 8.3.4 | 9.0.3 | PYSEC-2026-1845 |
| `pytest-asyncio` | 0.25.0 | 1.4.1 | compat with pytest 9 |
| `python-jose[cryptography]` | 3.3.0 | **removed** | unused in `app/` + `tests/`; dragged in ecdsa 0.19.2 (PYSEC-2026-1325, no fix) |
| `passlib` | 1.7.4 | **removed** | unused in `app/` + `tests/` |
| `sentence-transformers` | 3.3.1 | 5.6.0 | Phase 2 — CrossEncoder API in `app/rag/reranker.py` unchanged (constructor + `.predict()` kwargs stable across 3.x→5.x); 9/9 retriever tests green; end-to-end rerank verified against `cross-encoder/ms-marco-TinyBERT-L2-v2` |
| `transformers` | 4.57.6 (transitive) | **5.21.0 pinned** | Phase 2 — bumped alongside sentence-transformers; closes PYSEC-2026-{2288,2289,2290,3929} + PYSEC-2026-4174 + PYSEC-2025-217 |

## Backend — remaining findings

`pip-audit -r requirements.txt --vulnerability-service osv --progress-spinner off`
returns **No known vulnerabilities found** against the declared dependency set.

The 12 findings seen when pip-audit is pointed at the whole installed
environment are all in `pip` 24.0 itself (PYSEC-2026-{196, 1795, 1796, 2875,
2876, 3721}; each is listed twice, once for runtime scope and once for
build scope). These are tooling-only, carry no CVE IDs, and are resolved
by the standard `python3 -m pip install -U pip` step in CI image builds.
They are NOT shipped by the application and do not meet the plan's
"fail on HIGH/CRITICAL" gate.

The CVE in the audit (`langsmith` CVE-2026-59152) is fixed.

## Frontend changes applied

| Package | From | To | Advisories closed |
| --- | --- | --- | --- |
| `postcss` | 8.4.49 | 8.5.32 | GHSA-{6g55-p6wh-862q, fxqj-rqcc-2cmp, qx2v-qp2m-jg93, r28c-9q8g-f849} |
| `sharp` (transitive via next) | — | fixed by `npm audit fix` (non-breaking) | GHSA-{f88m-g3jw-g9cj, rgj7-g3m4-5g8c, wq5f-xc86-pv6w}; CVE-2026-{33327, 33328, 35590, 35591, 96889} |

Lockfile regenerated via `npm i --package-lock-only`; `node_modules` was
not installed (CI generates fresh).

## Patch policy (ISO 27001 A.8.8)

1. Dependency audits run on every CI pipeline (pip-audit + npm audit, both
   `--strict` where available).
2. Any new HIGH or CRITICAL CVE blocks the pipeline until pinned.
3. PYSEC advisories without a CVSS score are triaged weekly. The audit
   closes with zero findings against the declared dependency set as of
   this commit (LangChain is on 1.3.9, sentence-transformers on 5.6.0).
4. Removed packages (`python-jose`, `passlib`) must stay removed —
   `app/` does not import them, and both carry unmaintained CVEs.
5. CI image builds must `pip install -U pip` before any `pip-audit` run
   so tooling-level findings (pip 24.0 → 26.x) don't mask product findings.

## Supporting hardening (security-fix-4, 2026-10-27)

Three defects unrelated to CVEs but gating a clean security CI run:

1. **`fitz` FutureWarning** (`app/services/pdf_extractor.py`): PyMuPDF
   renamed its top-level module from `fitz` to `pymupdf` and now emits
   `FutureWarning` on every `import fitz`. The warning fired on every
   pytest run and would break any `-W error` CI gate once PyMuPDF 3.x
   drops the alias. Switched the production import to `import pymupdf`
   and the single `fitz.open()` call to `pymupdf.open()`.
2. **T-014 test isolation leak** — eight errors in
   `tests/test_processing_pipeline.py` when run after
   `tests/test_document_mgmt_http.py` /
   `tests/test_document_service.py`. Both files hand-swapped
   `UUID(as_uuid=True)` columns for `String(36)` at import or fixture
   scope to make the Postgres schema fit in-memory SQLite. The swap
   mutated `Table.columns[*].type` globally, but SQLAlchemy caches
   `Comparator.type` separately, so `WHERE Document.id = ?` kept
   binding `uuid.hex` while `INSERT` bound `str(uuid)` — mismatch, zero
   rows, `get_status` returned `None`. Fixed at the source: added
   `app/db/types.py::UUIDType`, a `TypeDecorator` that renders native
   `UUID` on PostgreSQL and `CHAR(36)` on every other dialect, with
   `uuid.UUID` round-tripping both ways. Updated `app/db/models.py` to
   use it; removed both tests' column-type patches.
3. **ISO 27001 A.14 note:** moving to a cross-dialect UUID type is a
   schema-neutral change — PostgreSQL still stores and returns native
   UUIDs, SQL identical. The test suite now runs green end-to-end
   (`285 passed, 9 skipped, 0 failed`), closing a hidden supply-chain
   risk where test flakiness would mask real security regressions.

## Reproduction

```bash
# Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip-audit -r requirements.txt --vulnerability-service osv --strict

# Frontend
cd frontend
npm i --package-lock-only
npm audit
```
