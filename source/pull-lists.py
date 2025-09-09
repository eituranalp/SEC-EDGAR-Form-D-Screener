"""Build a Form D LLC master list from EDGAR master.idx and resolve filing URLs.

Outputs per-run artifacts under runs/<timestamp>/:
- params.json
- master_list.csv
- pull_summary.json
"""
 
import csv, datetime as dt, os, re, requests, json
import argparse
from time import sleep
from urllib.parse import urljoin

BASE = "https://www.sec.gov"
# My User-Agent string for SEC
UA = "ET/edgar-search-v1.0 erentibo@gmail.com Personal project"

# Base directories
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

def quarters_covering(start, end):  
    """Return [(year, "QTRn")] that overlap the inclusive [start, end] dates."""
    out = []
    y = start.year
    while dt.date(y, 1, 1) <= end:
        for q, months in enumerate([(1,2,3),(4,5,6),(7,8,9),(10,11,12)], start=1):
            q_start = dt.date(y, months[0], 1)
            last_m = months[-1]
            q_end = dt.date(y, 12, 31) if last_m == 12 else dt.date(y, last_m + 1, 1) - dt.timedelta(days=1)
            if q_end < start or q_start > end:
                continue
            out.append((y, f"QTR{q}"))
        y += 1
    return out

def stream_master_idx(year, qtr, session):
    """Yield rows from master.idx for a given year and quarter."""
    url = f"{BASE}/Archives/edgar/full-index/{year}/{qtr}/master.idx"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            sleep(1 + attempt)
    lines = r.text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
            start = i
            break
    if start is None:
        print(f"Warning: master.idx header not found: {url}")
        return
    for ln in lines[start+1:]:
        parts = ln.split("|")
        if len(parts) == 5:
            yield {
                "cik": parts[0].lstrip("0"),
                "company": parts[1],
                "form": parts[2],
                "date": parts[3],
                "path": parts[4],
            }

def company_has_llc(name):
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    return "llc" in s

def build_primary_xml_url_from_idx_path(relative_path):
    """Best-effort build of primary_doc.xml URL from master.idx path; else fallback."""
    try:
        p = relative_path.lstrip("/")
        parts = p.split("/")
        if len(parts) < 4 or parts[0] != "edgar" or parts[1] != "data":
            return urljoin(f"{BASE}/Archives/", relative_path)
        cik = parts[2]
        last_segment = parts[-1]
        if last_segment.endswith(".txt") and "-" in last_segment:
            accession_with_hyphens = last_segment.rsplit(".", 1)[0]
            accession_no_dashes = accession_with_hyphens.replace("-", "")
        else:
            accession_no_dashes = parts[3].replace("-", "")
        return f"{BASE}/Archives/edgar/data/{cik}/{accession_no_dashes}/primary_doc.xml"
    except Exception:
        return urljoin(f"{BASE}/Archives/", relative_path)

def build_best_filing_url_from_index_json(relative_path, session, log_stats):
    """Resolve the filing URL via index.json, preferring primary XML, then any XML,
    then primary HTML, then any HTML, otherwise the original .txt. Tally outcomes
    in log_stats.
    """
    try:
        # Normalize and split master.idx path
        p = relative_path.lstrip("/")
        parts = p.split("/")
        if len(parts) < 4 or parts[0] != "edgar" or parts[1] != "data":
            raise ValueError("Unexpected master.idx path shape")

        cik = parts[2]
        last_segment = parts[-1]
        if last_segment.endswith(".txt") and "-" in last_segment:
            accession_with_hyphens = last_segment.rsplit(".", 1)[0]
            accession_no_dashes = accession_with_hyphens.replace("-", "")
        else:
            # Some rows are already like edgar/data/{cik}/{accNoNoDashes}/something
            accession_no_dashes = parts[3].replace("-", "")

        base_folder = f"/Archives/edgar/data/{cik}/{accession_no_dashes}/"
        index_json_url = urljoin(BASE, base_folder + "index.json")

        # Fetch index.json with retries
        data = None
        for attempt in range(3):
            try:
                r = session.get(index_json_url, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 2:
                    raise
                sleep(1 + attempt)
        if not data:
            raise ValueError("No data from index.json")
        items = (data.get("directory") or {}).get("item") or []
        names = [it.get("name") for it in items if isinstance(it, dict) and it.get("name")]

        def join_url(name):
            return urljoin(BASE, base_folder + name)

        # 1) Exact primary_doc.xml
        for n in names:
            if n.lower() == "primary_doc.xml":
                log_stats["xml_primary"] = log_stats.get("xml_primary", 0) + 1
                return join_url(n)

        # 2) Any xml with 'primary' in name
        for n in names:
            if n.lower().endswith(".xml") and "primary" in n.lower():
                log_stats["xml_any_primary"] = log_stats.get("xml_any_primary", 0) + 1
                return join_url(n)

        # 3) First xml
        for n in names:
            if n.lower().endswith(".xml"):
                log_stats["xml_any"] = log_stats.get("xml_any", 0) + 1
                return join_url(n)

        # 4) HTML preference: primary first, then any
        html_candidates = [n for n in names if n.lower().endswith((".htm", ".html"))]
        primaries = [n for n in html_candidates if "primary" in n.lower()]
        if primaries:
            log_stats["html_primary"] = log_stats.get("html_primary", 0) + 1
            return join_url(primaries[0])
        if html_candidates:
            log_stats["html_any"] = log_stats.get("html_any", 0) + 1
            return join_url(html_candidates[0])

        # 5) Fallback to original .txt
        log_stats["txt_fallback"] = log_stats.get("txt_fallback", 0) + 1
        return urljoin(f"{BASE}/Archives/", relative_path)
    except Exception:
        log_stats["json_error"] = log_stats.get("json_error", 0) + 1
        return urljoin(f"{BASE}/Archives/", relative_path)

def _configure_session():
    """Create a requests.Session with retries and keep-alive headers."""
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Connection": "keep-alive"})
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    except Exception:
        pass
    return s

def resolve_filing_url(relative_path, session, log_stats, mode="hybrid"):
    """Resolve filing URL using fast XML guess (hybrid) or index.json (index)."""
    if mode == "index":
        return build_best_filing_url_from_index_json(relative_path, session, log_stats)
    url_xml_guess = build_primary_xml_url_from_idx_path(relative_path)
    log_stats["xml_guess"] = log_stats.get("xml_guess", 0) + 1
    return url_xml_guess

def find_form_d_llc(start_date, end_date, out_csv=None, resolver_mode="hybrid", run_dir=None):
    """Scan master.idx in the date window and write a Form D LLC CSV under runs/."""
    # Prepare run directory and outputs
    if run_dir is None:
        now = dt.datetime.now()
        ts = now.strftime("%Y-%m-%d_%H-%M-%S") + f"-{now.microsecond:06d}"
        run_dir = os.path.join(RUNS_DIR, ts)
        # Ensure unique folder - if somehow it exists, add counter
        counter = 1
        original_run_dir = run_dir
        while os.path.exists(run_dir):
            run_dir = f"{original_run_dir}-{counter}"
            counter += 1
    os.makedirs(run_dir, exist_ok=True)
    if out_csv is None:
        out_csv = os.path.join(run_dir, "master_list.csv")

    session = _configure_session()
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(end_date,   "%Y-%m-%d").date()

    # Counters for funnel summary
    total_in_window = 0
    count_form_d_da = 0
    count_llc_name = 0

    # Track resolver usage stats
    if not hasattr(find_form_d_llc, "_log_stats"):
        find_form_d_llc._log_stats = {}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["cik","company","form","date","filing_index_url"])
        for year, qtr in quarters_covering(start, end):
            try:
                for row in stream_master_idx(year, qtr, session):
                    rdate = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
                    if not (start <= rdate <= end):
                        continue
                    total_in_window += 1
                    form_type = (row.get("form") or "").strip().upper()
                    if form_type not in {"D", "D/A"}:
                        continue
                    count_form_d_da += 1
                    if not company_has_llc(row["company"]):
                        continue
                    count_llc_name += 1
                    # Resolve with selected mode and log usage
                    try:
                        filing_index_url = resolve_filing_url(row["path"], session, find_form_d_llc._log_stats, resolver_mode)
                        wr.writerow([row["cik"], row["company"], row["form"], row["date"], filing_index_url])
                        # light progress every 500 records
                        find_form_d_llc._written = getattr(find_form_d_llc, "_written", 0) + 1
                        if (find_form_d_llc._written % 500) == 0:
                            print(f"Wrote {find_form_d_llc._written} rows…")
                    except Exception as e:
                        # Do not crash the whole run due to one filing
                        print(f"Error resolving filing URL for CIK {row.get('cik')} on {row.get('date')}: {e}")
                        continue
            except Exception as e:
                # If a whole quarter fails (e.g., throttled), log and continue
                print(f"Warning: failed {year} {qtr} master.idx: {e}")
            # Small pause between quarters to reduce throttling
            sleep(0.2)

    # Print a one-line summary of resolver outcomes if available
    stats = getattr(find_form_d_llc, "_log_stats", None)
    if stats:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        print(f"Resolver summary: {summary}")

    # Write params.json
    params = {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "form_filter": ["D", "D/A"],
        "include_amendments": True,
        "resolver_mode": resolver_mode,
        "user_agent": UA,
    }
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as pf:
        json.dump(params, pf, indent=2)

    # Write pull_summary.json
    pull_summary = {
        "total_in_window": total_in_window,
        "form_d_or_da": count_form_d_da,
        "llc_name": count_llc_name,
        "resolver_usage": dict(sorted((stats or {}).items())),
        "master_list_csv": os.path.abspath(out_csv),
    }
    with open(os.path.join(run_dir, "pull_summary.json"), "w", encoding="utf-8") as sf:
        json.dump(pull_summary, sf, indent=2)

if __name__ == "__main__":
    def quarter_date_range(year: int, quarter: str):
        q = quarter.strip().upper().replace("QTR", "Q").replace("QUARTER", "Q").lstrip("Q")
        if q not in {"1", "2", "3", "4"}:
            raise ValueError("quarter must be 1..4 or Q1..Q4")
        q = int(q)
        q_to_months = {
            1: (1, 3),
            2: (4, 6),
            3: (7, 9),
            4: (10, 12),
        }
        start_m, end_m = q_to_months[q]
        start_date = dt.date(year, start_m, 1)
        # end of quarter = first day of next month minus 1 day
        if end_m == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, end_m + 1, 1) - dt.timedelta(days=1)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Fetch Form D LLC filings list")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--year", type=int, help="Year of quarter to fetch (e.g., 2025)")
    parser.add_argument("--quarter", type=str, default="Q1", help="Quarter (Q1..Q4 or 1..4). Used with --year")
    group.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (required if --start is set)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (defaults to runs/<timestamp>/master_list.csv)")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "index"], help="Resolver mode")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to new runs/<timestamp>/)")
    args = parser.parse_args()

    if args.year is not None:
        start_s, end_s = quarter_date_range(args.year, args.quarter)
    elif args.start and args.end:
        start_s, end_s = args.start, args.end
    else:
        # Default: current quarter only (small, fast test run)
        today = dt.date.today()
        m = today.month
        q = 1 if m <= 3 else 2 if m <= 6 else 3 if m <= 9 else 4
        start_s, end_s = quarter_date_range(today.year, f"Q{q}")

    find_form_d_llc(start_date=start_s, end_date=end_s, out_csv=args.out, resolver_mode=args.mode, run_dir=args.run_dir)
