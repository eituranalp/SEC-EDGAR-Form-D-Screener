# SEC Form D: CA Residential LLCs (One Per Company)

**Extracts Form D filings → filters to CA LLC residential real estate → deduplicates to one record per company.**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Pull filings for a date range
python source/pull-lists.py --start 2025-01-01 --end 2025-01-31

# Filter to final companies
python source/filter-lists.py --run-dir runs/<timestamp>
```

## Outputs

Each run creates `runs/<timestamp>/` with:
- **`master_list.csv`** - All Form D LLC filings in date range
- **`final_companies.csv`** - CA residential LLCs, one per company, with related persons
- **`pull_summary.json`** - Funnel: total filings → D/D-A → LLC name → master list
- **`filter_summary.json`** - Funnel: XML reachable → CA → residential → final count + latency/429s/dedupe metrics

**One-pager:** [docs/EDGARFormDScreener-OnePager](https://eituranalp.github.io/SEC-EDGAR-Form-D-Screener/EDGARFormDScreener-OnePager.pdf?v=2025-09-08)

## Politeness & Rate Limits

**User-Agent:** `ET/edgar-search-v1.0 erentibo@gmail.com Personal project`  
**Throttling:** 80ms between requests + exponential backoff on 429/503  
**Retries:** 3 attempts per request with respect for Retry-After headers

## Script Details

### `pull-lists.py`
Streams SEC master.idx files → filters to Form D/D-A with "LLC" in company name → resolves filing URLs.

**Mode:**
- `--mode hybrid` (default): Fast XML URL guessing
- `--mode index`: Fetch `index.json` per filing (slower, more robust)

### `filter-lists.py`
Fetches XML → parses issuer/offering data → applies filters (CA + residential + LLC entity type) → deduplicates globally → keeps newest filing per CIK.

## Limitations

- **Industry labeling variability:** "Residential" detection relies on `industryGroupType` field consistency
- **506(b) detection:** Heuristic text matching; not enforced in current filters  
- **Related person dedupe:** Global across all companies; may drop legitimate duplicates
- **XML availability:** Some filings may only have HTML/TXT; hybrid mode assumes XML exists

## Full Command Reference

### Environment Setup
```bash
python -m venv .venv
source .venv/Scripts/activate  # or .venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
```

### pull-lists.py Flags
- `--year 2025 --quarter Q1` - Specific quarter
- `--start YYYY-MM-DD --end YYYY-MM-DD` - Custom date range  
- `--mode [hybrid|index]` - URL resolution strategy
- `--out PATH` - Custom output CSV (default: `runs/<timestamp>/master_list.csv`)
- `--run-dir PATH` - Custom run directory

### filter-lists.py Flags  
- `--run-dir PATH` - Use specific run (default: most recent)
- `--in PATH` - Custom input CSV
- `--out PATH` - Custom output CSV  
- `--log` - Write `log.txt` summary
