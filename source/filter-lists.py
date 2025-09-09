"""Filter and enrich Form D XML rows to a deduped, CA residential LLC subset.

Reads master_list.csv from a runs/<timestamp>/ folder and writes:
- final_companies.csv (one row per CIK)
- filter_summary.json (funnel + latency + rate-limit + dedupe + reduction)
Optionally logs basic progress to log.txt.
"""

import csv, json, statistics
import os
import re
import sys
import time
from collections import deque
import datetime as dt
from typing import Dict, Tuple, Optional, List
import unicodedata

import requests
from xml.etree import ElementTree as ET

BASE_UA = "ET/edgar-search-v1.0 erentibo@gmail.com Personal project"


def configure_session() -> requests.Session:
    """Create a requests.Session with retries and keep-alive headers."""
    s = requests.Session()
    s.headers.update({"User-Agent": BASE_UA, "Connection": "keep-alive"})
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    except Exception:
        pass
    return s


ACCESSION_HYPHEN_RE = re.compile(r"(\d{10})-(\d{2})-(\d{6})")
ACCESSION_18_RE = re.compile(r"/data/\d+/(\d{18})/")
DETECT_506B_RE = re.compile(r"(?i)\b(?:506\s*\(\s*b\s*\)|506\s*b|06b)\b")


def strip_ns(tree: ET.ElementTree) -> None:
    """Remove XML namespaces in-place for simpler tag access."""
    root = tree.getroot()
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def find_text_ci(elem: ET.Element, *tags: str) -> Optional[str]:
    """Case-insensitive search for first matching tag text in subtree."""
    tagset = {t.lower() for t in tags}
    for child in elem.iter():
        if child.tag.lower() in tagset and (child.text or "").strip():
            return child.text.strip()
    return None


def _find_child_ci(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    """Case-insensitive direct child lookup."""
    tl = tag.lower()
    for ch in list(parent):
        if (ch.tag or "").lower() == tl:
            return ch
    return None


def _find_path_text_ci(root: ET.Element, path: str) -> str:
    """Case-insensitive slash-path lookup returning stripped text or empty string."""
    cur: Optional[ET.Element] = root
    for seg in path.split("/"):
        if cur is None:
            return ""
        cur = _find_child_ci(cur, seg)
    if cur is None:
        return ""
    return (cur.text or "").strip()


def _find_all_children_ci(parent: ET.Element, tag: str) -> List[ET.Element]:
    """All direct children with tag name (case-insensitive)."""
    tl = tag.lower()
    return [ch for ch in list(parent) if (ch.tag or "").lower() == tl]


def detect_506b(root: ET.Element) -> bool:
    """Heuristically detect mentions of 506(b) in text or *_code tags."""
    text = ET.tostring(root, encoding="utf-8", method="text").decode("utf-8", errors="ignore")
    if DETECT_506B_RE.search(text):
        return True
    for el in root.iter():
        if el.tag.lower().endswith("code") and el.text:
            if DETECT_506B_RE.search(el.text):
                return True
    return False


def parse_accession_tuple(xml_url: str) -> Tuple[int, int, int]:
    """Extract (10,2,6) accession parts for sorting; fallback is (0,0,0)."""
    m = ACCESSION_HYPHEN_RE.search(xml_url)
    if m:
        return tuple(int(x) for x in m.groups())  # type: ignore[return-value]
    m2 = ACCESSION_18_RE.search(xml_url)
    if m2:
        eighteen = m2.group(1)
        # Reconstruct as (10,2,6) if possible; else treat as big int chunks
        try:
            a = int(eighteen[:10])
            b = int(eighteen[10:12])
            c = int(eighteen[12:])
            return (a, b, c)
        except Exception:
            pass
        # Fallback: fit into tuple
        big = int(eighteen)
        return (big // 10**8, (big // 10**6) % 100, big % 10**6)
    return (0, 0, 0)


def _normalize_name(first: str, last: str) -> str:
    """Lowercase, ASCII-fold and trim non-alnum to normalize person names."""
    base = f"{(first or '').strip()} {(last or '').strip()}".strip().lower()
    base = ''.join(ch for ch in unicodedata.normalize('NFKD', base) if not unicodedata.combining(ch))
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    return base


def extract_full_record(xml_content: bytes) -> Dict[str, object]:
    """Parse primary_doc.xml bytes to a flat dict of issuer/offering fields."""
    tree = ET.ElementTree(ET.fromstring(xml_content))
    strip_ns(tree)
    root = tree.getroot()

    submission_type = _find_path_text_ci(root, "submissionType")

    cik = _find_path_text_ci(root, "primaryIssuer/cik")
    entity_name = _find_path_text_ci(root, "primaryIssuer/entityName")
    street1 = _find_path_text_ci(root, "primaryIssuer/issuerAddress/street1")
    street2 = _find_path_text_ci(root, "primaryIssuer/issuerAddress/street2")
    city = _find_path_text_ci(root, "primaryIssuer/issuerAddress/city")
    state_desc = _find_path_text_ci(root, "primaryIssuer/issuerAddress/stateOrCountryDescription")
    zip_code = _find_path_text_ci(root, "primaryIssuer/issuerAddress/zipCode")
    state_code = _find_path_text_ci(root, "primaryIssuer/issuerAddress/stateOrCountry")  # for filtering
    issuer_phone = _find_path_text_ci(root, "primaryIssuer/issuerPhoneNumber")
    juris_inc = _find_path_text_ci(root, "primaryIssuer/jurisdictionOfInc")
    entity_type = _find_path_text_ci(root, "primaryIssuer/entityType")
    year_inc = _find_path_text_ci(root, "primaryIssuer/yearOfInc/value")

    related_persons: List[Dict[str, str]] = []
    rplist = _find_child_ci(root, "relatedPersonsList")
    if rplist is not None:
        for rp in _find_all_children_ci(rplist, "relatedPersonInfo"):
            name_el = _find_child_ci(rp, "relatedPersonName")
            if name_el is None:
                name_el = rp
            first = _find_path_text_ci(name_el, "firstName")
            last = _find_path_text_ci(name_el, "lastName")
            addr_el = _find_child_ci(rp, "relatedPersonAddress")
            if addr_el is None:
                addr_el = rp
            rp_street1 = _find_path_text_ci(addr_el, "street1")
            rp_street2 = _find_path_text_ci(addr_el, "street2")
            rp_city = _find_path_text_ci(addr_el, "city")
            rp_state_desc = _find_path_text_ci(addr_el, "stateOrCountryDescription")
            rp_zip = _find_path_text_ci(addr_el, "zipCode")
            rel_list_el = _find_child_ci(rp, "relatedPersonRelationshipList")
            relationships: List[str] = []
            if rel_list_el is not None:
                for rr in _find_all_children_ci(rel_list_el, "relationship"):
                    if rr.text and rr.text.strip():
                        relationships.append(rr.text.strip())
            related_persons.append({
                "firstName": first,
                "lastName": last,
                "street1": rp_street1,
                "street2": rp_street2,
                "city": rp_city,
                "stateOrCountryDescription": rp_state_desc,
                "zipCode": rp_zip,
                "relationship": "|".join(relationships) if relationships else "",
                "_norm": _normalize_name(first, last),
            })

    industry_group_type = _find_path_text_ci(root, "offeringData/industryGroup/industryGroupType")
    revenue_range = _find_path_text_ci(root, "offeringData/issuerSize/revenueRange")
    offering_el = _find_child_ci(root, "offeringData")
    fee_el = _find_child_ci(offering_el, "federalExemptionsExclusions") if offering_el is not None else None
    fed_items: List[str] = []
    if fee_el is not None:
        for it in _find_all_children_ci(fee_el, "item"):
            if it.text and it.text.strip():
                fed_items.append(it.text.strip())
    fed_ex_item = "|".join(fed_items)
    date_first_sale = _find_path_text_ci(root, "offeringData/typeOfFiling/dateOfFirstSale/value")
    more_than_one_year = _find_path_text_ci(root, "offeringData/durationOfOffering/moreThanOneYear")
    is_equity = _find_path_text_ci(root, "offeringData/typesOfSecuritiesOffered/isEquityType")
    is_biz_combo = _find_path_text_ci(root, "offeringData/businessCombinationTransaction/isBusinessCombinationTransaction")
    total_offering_amt = _find_path_text_ci(root, "offeringData/offeringSalesAmounts/totalOfferingAmount")
    total_amount_sold = _find_path_text_ci(root, "offeringData/offeringSalesAmounts/totalAmountSold")
    total_remaining = _find_path_text_ci(root, "offeringData/offeringSalesAmounts/totalRemaining")
    has_non_accred = _find_path_text_ci(root, "offeringData/investors/hasNonAccreditedInvestors")
    total_already_invested = _find_path_text_ci(root, "offeringData/investors/totalNumberAlreadyInvested")
    gross_proceeds_used = _find_path_text_ci(root, "offeringData/useOfProceeds/grossProceedsUsed/dollarAmount")

    sig_block = _find_child_ci(root, "signatureBlock")
    if sig_block is not None:
        sig_issuer_name = find_text_ci(sig_block, "issuerName") or ""
        sig_signature_name = find_text_ci(sig_block, "signatureName") or ""
        sig_name_of_signer = find_text_ci(sig_block, "nameOfSigner") or ""
        sig_title = find_text_ci(sig_block, "signatureTitle") or ""
        sig_date = find_text_ci(sig_block, "signatureDate") or ""
    else:
        sig_issuer_name = ""
        sig_signature_name = ""
        sig_name_of_signer = ""
        sig_title = ""
        sig_date = ""

    entity_for_filter = entity_type or find_text_ci(root, "entityType", "issuerType") or ""
    state_for_filter = state_code or find_text_ci(root, "stateOrCountry", "state") or ""
    industry_for_filter = industry_group_type or find_text_ci(root, "industryGroupType", "industryGroup") or ""
    has_506b = detect_506b(root)

    return {
        "submissionType": submission_type,
        "cik": cik,
        "entityName": entity_name,
        "street1": street1,
        "street2": street2,
        "city": city,
        "stateOrCountryDescription": state_desc,
        "zipCode": zip_code,
        "issuerPhoneNumber": issuer_phone,
        "jurisdictionOfInc": juris_inc,
        "entityType": entity_type,
        "yearOfInc": year_inc,
        "relatedPersons": related_persons,
        "industryGroupType": industry_group_type,
        "revenueRange": revenue_range,
        "federalExemptionsExclusions_item": fed_ex_item,
        "dateOfFirstSale_value": date_first_sale,
        "durationOfOffering_moreThanOneYear": more_than_one_year,
        "typesOfSecuritiesOffered_isEquityType": is_equity,
        "isBusinessCombinationTransaction": is_biz_combo,
        "totalOfferingAmount": total_offering_amt,
        "totalAmountSold": total_amount_sold,
        "totalRemaining": total_remaining,
        "hasNonAccreditedInvestors": has_non_accred,
        "totalNumberAlreadyInvested": total_already_invested,
        "grossProceedsUsed_dollarAmount": gross_proceeds_used,
        "issuerName": sig_issuer_name,
        "signatureName": sig_signature_name,
        "nameOfSigner": sig_name_of_signer,
        "signatureTitle": sig_title,
        "signatureDate": sig_date,
        # For filtering
        "_filter_entity": entity_for_filter,
        "_filter_state": state_for_filter,
        "_filter_industry": industry_for_filter,
        "_has_506b": has_506b,
    }


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

def main(in_csv: str = None, out_csv: str = None, run_dir: str = None, write_log: bool = False) -> int:
    """Read the master list from runs/, fetch/parse XML, filter, and write outputs."""
    # Resolve run directory and filenames
    if run_dir is None:
        # Use most recent run directory if not provided
        try:
            entries = [os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)]
            run_dir = max((p for p in entries if os.path.isdir(p)), key=os.path.getmtime)
        except Exception:
            raise SystemExit("No runs/ directory found. Run pull-lists.py first.")
    os.makedirs(run_dir, exist_ok=True)
    if in_csv is None:
        in_csv = os.path.join(run_dir, "master_list.csv")
    if out_csv is None:
        out_csv = os.path.join(run_dir, "final_companies.csv")
    log_path = os.path.join(run_dir, "log.txt") if write_log else None

    session = configure_session()
    throttle_s = 0.08

    by_cik: Dict[str, list] = {}
    total_rows = 0
    xml_reachable = 0
    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            total_rows += 1
            cik = (row.get("cik") or "").lstrip("0")
            company = (row.get("company") or "").strip()
            date_s = (row.get("date") or "").strip()
            xml_url = (row.get("filing_index_url") or "").strip()
            if not cik or not company or not date_s or not xml_url:
                continue
            if not xml_url.lower().endswith(".xml"):
                continue
            try:
                file_date = dt.datetime.strptime(date_s, "%Y-%m-%d").date()
            except Exception:
                continue
            acc_tuple = parse_accession_tuple(xml_url)
            by_cik.setdefault(cik, []).append((file_date, acc_tuple, company, xml_url))

    for cik, lst in by_cik.items():
        lst.sort(key=lambda t: (t[0], t[1]), reverse=True)

    unique_ciks = len(by_cik)
    print(f"Loaded {total_rows} rows; {unique_ciks} unique CIKs. Processing newest filings per CIK…", flush=True)

    recent_req_durations = deque(maxlen=200)
    total_requests = 0
    rate_limited = 0
    retried_requests = 0
    prev_total_requests = 0
    prev_rate_limited = 0
    prev_retried_requests = 0

    results: Dict[str, List[str]] = {}
    seen_person_names: set = set()
    processed = 0
    kept = 0

    for cik, filings in by_cik.items():
        processed += 1
        found = False
        for file_date, acc_tuple, company, xml_url in filings:
            backoff = 0.5
            xml_bytes = b""
            req_start = time.perf_counter()
            attempts_made = 0
            saw_rate_limit = False
            for attempt in range(3):
                attempts_made += 1
                try:
                    r = session.get(xml_url, timeout=60)
                    if r.status_code in (429, 503):
                        saw_rate_limit = True
                        raise requests.HTTPError(f"{r.status_code} rate limited")
                    r.raise_for_status()
                    xml_bytes = r.content
                    xml_reachable += 1
                    break
                except Exception:
                    if attempt == 2:
                        break
                    time.sleep(backoff)
                    backoff *= 2
            req_dur = time.perf_counter() - req_start
            recent_req_durations.append(req_dur)
            total_requests += 1
            if saw_rate_limit:
                rate_limited += 1
            if attempts_made > 1:
                retried_requests += 1
            time.sleep(throttle_s)
            if not xml_bytes:
                continue

            try:
                rec = extract_full_record(xml_bytes)
            except Exception:
                continue

            entity = (rec.get("_filter_entity") or "").lower()
            state = (rec.get("_filter_state") or "").upper()
            industry = (rec.get("_filter_industry") or "").lower()
            if "limited liability company" not in entity:
                continue
            if state != "CA":
                continue
            if "residential" not in industry:
                continue

            persons: List[Dict[str, str]] = []
            for p in (rec.get("relatedPersons") or []):  # type: ignore[operator]
                norm = (p.get("_norm") or "")  # type: ignore[union-attr]
                if not norm:
                    continue
                if norm in seen_person_names:
                    continue
                seen_person_names.add(norm)
                persons.append(p)  # type: ignore[arg-type]
            if not persons:
                continue
            persons = persons[:3]

            # Build accession string
            m = ACCESSION_HYPHEN_RE.search(xml_url)
            if m:
                accession_str = "-".join(m.groups())
            else:
                m2 = ACCESSION_18_RE.search(xml_url)
                accession_str = m2.group(1) if m2 else ""

            latest_date_str = file_date.strftime("%Y-%m-%d")
            base_cols = [
                str(rec.get("submissionType") or ""),
                str(rec.get("cik") or ""),
                str(rec.get("entityName") or ""),
                str(rec.get("street1") or ""),
                str(rec.get("street2") or ""),
                str(rec.get("city") or ""),
                str(rec.get("stateOrCountryDescription") or ""),
                str(rec.get("zipCode") or ""),
                str(rec.get("issuerPhoneNumber") or ""),
                str(rec.get("jurisdictionOfInc") or ""),
                str(rec.get("entityType") or ""),
                str(rec.get("yearOfInc") or ""),
                str(rec.get("industryGroupType") or ""),
                str(rec.get("revenueRange") or ""),
                str(rec.get("federalExemptionsExclusions_item") or ""),
                str(rec.get("dateOfFirstSale_value") or ""),
                str(rec.get("durationOfOffering_moreThanOneYear") or ""),
                str(rec.get("typesOfSecuritiesOffered_isEquityType") or ""),
                str(rec.get("isBusinessCombinationTransaction") or ""),
                str(rec.get("totalOfferingAmount") or ""),
                str(rec.get("totalAmountSold") or ""),
                str(rec.get("totalRemaining") or ""),
                str(rec.get("hasNonAccreditedInvestors") or ""),
                str(rec.get("totalNumberAlreadyInvested") or ""),
                str(rec.get("grossProceedsUsed_dollarAmount") or ""),
            ]
            related_cols: List[str] = []
            for idx in range(3):
                if idx < len(persons):
                    rp = persons[idx]
                    related_cols.extend([
                        str(rp.get("firstName") or ""),
                        str(rp.get("lastName") or ""),
                        str(rp.get("street1") or ""),
                        str(rp.get("street2") or ""),
                        str(rp.get("city") or ""),
                        str(rp.get("stateOrCountryDescription") or ""),
                        str(rp.get("zipCode") or ""),
                        str(rp.get("relationship") or ""),
                    ])
                else:
                    related_cols.extend(["", "", "", "", "", "", "", ""])  

            provenance = [cik, company, latest_date_str, accession_str, xml_url]

            results[cik] = base_cols + related_cols + provenance
            kept += 1
            found = True
            break

        if (processed % 25) == 0:
            window_reqs = total_requests - prev_total_requests
            window_rates = rate_limited - prev_rate_limited
            window_retries = retried_requests - prev_retried_requests
            avg_ms = (sum(recent_req_durations) / len(recent_req_durations) * 1000.0) if recent_req_durations else 0.0
            print(
                f"Processed {processed}/{unique_ciks} CIKs… kept {kept} | "
                f"reqs:{window_reqs}, 429/503:{window_rates}, retries:{window_retries}, "
                f"avg_req_ms(last{len(recent_req_durations)}):{avg_ms:.0f}",
                flush=True,
            )
            prev_total_requests = total_requests
            prev_rate_limited = rate_limited
            prev_retried_requests = retried_requests

    rows = list(results.values())
    rows.sort(key=lambda r: ((r[2] or "").lower(), (r[-3] or "")))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        header = [
            "submissionType",
            "cik",
            "entityName",
            "street1",
            "street2",
            "city",
            "stateOrCountryDescription",
            "zipCode",
            "issuerPhoneNumber",
            "jurisdictionOfInc",
            "entityType",
            "yearOfInc",
            "industryGroupType",
            "revenueRange",
            "federalExemptionsExclusions_item",
            "dateOfFirstSale_value",
            "durationOfOffering_moreThanOneYear",
            "typesOfSecuritiesOffered_isEquityType",
            "isBusinessCombinationTransaction",
            "totalOfferingAmount",
            "totalAmountSold",
            "totalRemaining",
            "hasNonAccreditedInvestors",
            "totalNumberAlreadyInvested",
            "grossProceedsUsed_dollarAmount",
        ]
        for idx in range(1, 4):
            header.extend([
                f"related{idx}_firstName",
                f"related{idx}_lastName",
                f"related{idx}_street1",
                f"related{idx}_street2",
                f"related{idx}_city",
                f"related{idx}_stateOrCountryDescription",
                f"related{idx}_zipCode",
                f"related{idx}_relationship",
            ])
        header.extend(["_cik", "_company", "_latest_date", "_accession", "_xml_url"])
        wr.writerow(header)
        for row in rows:
            wr.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_csv}")

    # Build summary metrics
    latencies_ms = [ms * 1000.0 for ms in list(recent_req_durations)]
    avg_latency_ms = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    median_latency_ms = (statistics.median(latencies_ms) if latencies_ms else 0.0)
    kept_ciks = len(results)
    deduped_people = len(seen_person_names)
    # Dedupe rate: proportion of related person names dropped vs seen
    # Estimation: names kept = up to 3 per kept CIK
    names_kept = min(deduped_people, kept_ciks * 3)
    dedupe_rate = 1.0 - (names_kept / deduped_people) if deduped_people else 0.0
    reduction_pct = (1.0 - (kept_ciks / unique_ciks)) * 100.0 if unique_ciks else 0.0

    summary = {
        "xml_reachable": xml_reachable,
        "ciks_in_master": unique_ciks,
        "passes_ca": None,  # implicit in filters
        "passes_residential": None,  # implicit
        "passes_506b": None,  # detection computed but not enforced
        "unique_ciks_kept": kept_ciks,
        "avg_request_latency_ms": round(avg_latency_ms, 2),
        "median_request_latency_ms": round(median_latency_ms, 2),
        "429_or_503_count": rate_limited,
        "retried_requests": retried_requests,
        "total_requests": total_requests,
        "retry_rate": round((retried_requests / total_requests) * 100.0, 2) if total_requests else 0.0,
        "dedupe_rate": round(dedupe_rate * 100.0, 2),
        "percent_reduction_vs_unique_ciks": round(reduction_pct, 2),
        "final_companies_csv": os.path.abspath(out_csv),
    }
    with open(os.path.join(run_dir, "filter_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    # Optional log
    if log_path:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"Processed {processed} CIKs, kept {kept_ciks}\n")
            lf.write(f"Avg latency: {avg_latency_ms:.2f} ms, median: {median_latency_ms:.2f} ms\n")
            lf.write(f"429/503: {rate_limited}, retries: {retried_requests}/{total_requests}\n")
            lf.write(f"Final CSV: {out_csv}\n")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter master_list.csv and emit summaries")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to most recent runs/<timestamp>/)")
    parser.add_argument("--in", dest="in_csv", type=str, default=None, help="Input master_list.csv path")
    parser.add_argument("--out", dest="out_csv", type=str, default=None, help="Output final_companies.csv path")
    parser.add_argument("--log", dest="write_log", action="store_true", help="Write log.txt")
    args = parser.parse_args()
    sys.exit(main(in_csv=args.in_csv, out_csv=args.out_csv, run_dir=args.run_dir, write_log=args.write_log))


