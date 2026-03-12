from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------------
# Date formats tried in order when parsing a cell value
# ---------------------------------------------------------------------------
_DATE_FORMATS = [
    "%d/%m/%Y",   # 15/03/2024  — BOM, Kotak, SBI (some)
    "%d-%m-%Y",   # 15-03-2024  — Axis, HDFC variant
    "%d/%m/%y",   # 15/03/24   — HDFC short-year
    "%d-%m-%y",   # 15-03-24
    "%d %b %Y",   # 15 Mar 2024 — SBI
    "%d-%b-%Y",   # 15-Mar-2024 — ICICI
    "%d %B %Y",   # 15 March 2024
    "%Y-%m-%d",   # 2024-03-15  — ISO
    "%d.%m.%Y",   # 15.03.2024
]

_DATE_DETECT_RE = re.compile(
    r"^\d{1,2}[\s/\-.]\w{2,9}[\s/\-.]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"
)

# ---------------------------------------------------------------------------
# Bank fingerprint detection
# ---------------------------------------------------------------------------
_BANK_SIGNATURES: dict[str, list[str]] = {
    "SBI":                 ["state bank of india", "sbi "],
    "HDFC":                ["hdfc bank"],
    "ICICI":               ["icici bank"],
    "Axis Bank":           ["axis bank"],
    "Kotak":               ["kotak mahindra bank", "kotak bank"],
    "Yes Bank":            ["yes bank"],
    "PNB":                 ["punjab national bank"],
    "Bank of Baroda":      ["bank of baroda"],
    "Canara Bank":         ["canara bank"],
    "Union Bank":          ["union bank of india"],
    "IndusInd":            ["indusind bank"],
    "IDFC First":          ["idfc first bank", "idfc bank"],
    "Federal Bank":        ["federal bank"],
    "Bank of Maharashtra": ["bank of maharashtra", "mahabank", "mahb0"],
}

# ---------------------------------------------------------------------------
# Column header aliases → canonical name
#
# NOTE: Short tokens like "dr" / "cr" are only used for EXACT cell match,
# never for partial/substring matching, to avoid false positives
# (e.g. "address" containing "dr").
# ---------------------------------------------------------------------------

# Aliases that are safe for exact match only
_HEADER_ALIASES_EXACT: dict[str, list[str]] = {
    "sr_no":        ["sr", "sr no", "sl", "sl no", "sno", "#", "s.no", "sr.no", "no", "sr. no", "s. no"],
    "date":         ["date", "txn date", "transaction date", "value date", "trans date", "posting date"],
    "particulars":  ["particulars", "description", "narration", "details", "remarks",
                     "transaction details", "transaction narration", "cheque details"],
    "reference_no": ["reference", "ref no", "chq no", "cheque no", "ref", "chq/ref",
                     "reference number", "ref number", "cheque/ref no",
                     "cheque /reference no", "cheque/reference no"],
    "debit":        ["debit", "withdrawal", "dr", "debit amount", "withdrawal (dr)",
                     "withdrawals", "debit (dr)"],
    "credit":       ["credit", "deposit", "cr", "credit amount", "deposit (cr)",
                     "deposits", "credit (cr)"],
    "balance":      ["balance", "closing balance", "available balance", "bal",
                     "running balance"],
    "channel":      ["mode", "channel", "type", "transaction type", "txn mode",
                     "instrument"],
}

# Aliases safe for substring/partial matching (longer tokens only)
_HEADER_ALIASES_PARTIAL: dict[str, list[str]] = {
    "date":         ["date"],
    "particulars":  ["particulars", "description", "narration"],
    "debit":        ["debit", "withdrawal"],
    "credit":       ["credit", "deposit"],
    "balance":      ["balance"],
    "reference_no": ["reference", "cheque"],
}

TXN_COLUMNS = [
    "sr_no", "date", "particulars", "reference_no",
    "debit", "credit", "balance", "channel",
]


@dataclass
class ParseResult:
    transactions: pd.DataFrame
    page_count: int
    bank_name: str = "Unknown"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def parse_amount(value: object) -> float:
    text = _text(value).replace(",", "").replace(" ", "")
    if text in {"", "-", "—", "nil", "n/a", "n.a."}:
        return 0.0
    text = re.sub(r"(?i)(dr|cr)$", "", text).strip()
    try:
        return abs(float(text))
    except ValueError:
        return 0.0


def _parse_date(text: str) -> Optional[pd.Timestamp]:
    text = text.strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    try:
        result = pd.to_datetime(text, dayfirst=True)
        if 2000 <= result.year <= 2100:
            return result
    except Exception:
        pass
    return None


def _is_date_cell(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if _DATE_DETECT_RE.match(text):
        return True
    return _parse_date(text) is not None


# ---------------------------------------------------------------------------
# Bank detection
# ---------------------------------------------------------------------------

def _detect_bank(raw_text: str) -> str:
    lower = raw_text.lower()
    for bank, signatures in _BANK_SIGNATURES.items():
        if any(sig in lower for sig in signatures):
            return bank
    return "Unknown"


# ---------------------------------------------------------------------------
# Column mapping detection
# ---------------------------------------------------------------------------

def _detect_col_map_from_header(header_row: list[str]) -> dict[str, int]:
    """Match header cells to canonical column names.

    Returns a mapping only if it looks like a genuine transaction-table header:
    must contain at least ``date`` AND ``debit`` AND ``credit``.
    """
    mapping: dict[str, int] = {}
    normalized = [_text(h).lower().strip() for h in header_row]

    for col, aliases in _HEADER_ALIASES_EXACT.items():
        for i, cell in enumerate(normalized):
            if cell in aliases:
                if col not in mapping:
                    mapping[col] = i
                break

    # Second pass: partial/substring matching for columns not yet found
    for col, aliases in _HEADER_ALIASES_PARTIAL.items():
        if col in mapping:
            continue
        for i, cell in enumerate(normalized):
            if i in mapping.values():
                continue  # column index already assigned
            if any(alias in cell for alias in aliases):
                mapping[col] = i
                break

    # Require the three critical columns to accept this as a real header
    if not ({"date", "debit", "credit"} <= mapping.keys()):
        return {}

    return mapping


def _detect_col_map_from_data(sample_rows: list[list[str]]) -> dict[str, int]:
    """Infer column positions from data patterns when no header row is found."""
    if not sample_rows:
        return {}

    n_cols = max((len(r) for r in sample_rows), default=0)
    if n_cols < 4:
        return {}

    date_scores = [0] * n_cols
    amount_scores = [0] * n_cols

    for row in sample_rows:
        for i, cell in enumerate(row):
            if _is_date_cell(cell):
                date_scores[i] += 1
            raw = cell.replace(",", "").strip()
            try:
                float(raw)
                if raw:
                    amount_scores[i] += 1
            except ValueError:
                pass

    mapping: dict[str, int] = {}

    if max(date_scores, default=0) > 0:
        date_col = date_scores.index(max(date_scores))
        mapping["date"] = date_col
        if date_col > 0:
            mapping["sr_no"] = 0

    # Last 3 numeric columns → debit, credit, balance
    if n_cols >= 4:
        mapping["debit"] = n_cols - 3
        mapping["credit"] = n_cols - 2
        mapping["balance"] = n_cols - 1

    return mapping


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def _map_row(row: list[str], col_map: dict[str, int]) -> Optional[dict]:
    """Convert a raw table row to a canonical transaction record using col_map."""
    date_idx = col_map.get("date")
    if date_idx is None or date_idx >= len(row):
        return None

    date_val = _parse_date(row[date_idx])
    if date_val is None:
        return None

    def get(key: str) -> str:
        idx = col_map.get(key)
        if idx is None or idx >= len(row):
            return ""
        return row[idx]

    sr_raw = get("sr_no")
    try:
        sr_no = int(sr_raw)
    except (ValueError, TypeError):
        sr_no = 0

    return {
        "sr_no": sr_no,
        "date": date_val,
        "particulars": get("particulars"),
        "reference_no": get("reference_no"),
        "debit": parse_amount(get("debit")),
        "credit": parse_amount(get("credit")),
        "balance": parse_amount(get("balance")),
        "channel": get("channel"),
    }


# ---------------------------------------------------------------------------
# Table-extraction strategy
# ---------------------------------------------------------------------------

def _extract_via_tables(pdf) -> list[dict]:
    """Primary strategy: use pdfplumber table extraction."""
    all_records: list[dict] = []
    # Persist col_map across pages (same bank = same layout)
    # but re-detect per-table if we haven't found a good one yet.
    global_col_map: dict[str, int] = {}

    for page in pdf.pages:
        tables = page.extract_tables() or []
        for table in tables:
            if not table:
                continue

            working_table = table
            table_col_map: dict[str, int] = {}

            # Always try header detection for this table
            for i, raw_row in enumerate(table[:5]):
                row = [_text(c) for c in raw_row]
                candidate = _detect_col_map_from_header(row)
                if candidate:
                    table_col_map = candidate
                    working_table = table[i + 1:]
                    break

            # Use this table's header if found, otherwise fall back to global
            if table_col_map:
                global_col_map = table_col_map
            elif not global_col_map:
                # Try inferring from data
                data_rows = [[_text(c) for c in r] for r in table if r]
                inferred = _detect_col_map_from_data(data_rows)
                if inferred:
                    global_col_map = inferred

            active_map = table_col_map or global_col_map
            if not active_map:
                continue

            for raw_row in working_table:
                if not raw_row:
                    continue
                row = [_text(c) for c in raw_row]
                record = _map_row(row, active_map)
                if record:
                    all_records.append(record)

    return all_records


# ---------------------------------------------------------------------------
# Text-line fallback strategy
# ---------------------------------------------------------------------------

_TEXT_LINE_RE = re.compile(
    r"(\d{1,2}[\s/\-.]\w{2,9}[\s/\-.]\d{2,4})"  # date
    r"\s+"
    r"(.+?)"                                        # description
    r"\s+([\d,]+\.\d{2})"                          # amount 1
    r"\s+([\d,]+\.\d{2}|-)"                        # amount 2
    r"\s+([\d,]+\.\d{2})"                          # balance
)


def _extract_via_text(pdf) -> list[dict]:
    """Fallback strategy: regex over raw extracted text lines."""
    records: list[dict] = []
    sr_counter = 1

    for page in pdf.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            m = _TEXT_LINE_RE.search(line)
            if not m:
                continue
            date_val = _parse_date(m.group(1))
            if date_val is None:
                continue

            a1 = parse_amount(m.group(3))
            a2 = parse_amount(m.group(4))
            bal = parse_amount(m.group(5))

            if a2 == 0:
                debit, credit = a1, 0.0
            elif a1 == 0:
                debit, credit = 0.0, a2
            else:
                continue

            records.append({
                "sr_no": sr_counter,
                "date": date_val,
                "particulars": m.group(2).strip(),
                "reference_no": "",
                "debit": debit,
                "credit": credit,
                "balance": bal,
                "channel": "",
            })
            sr_counter += 1

    return records


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _build_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        df = pd.DataFrame(columns=TXN_COLUMNS)
        df["date"] = pd.Series(dtype="datetime64[ns]")
        for col in ["debit", "credit", "balance"]:
            df[col] = pd.Series(dtype="float64")
        return df

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["date", "sr_no"], na_position="last").reset_index(drop=True)
    df["sr_no"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_statement_pdf(pdf_path: str) -> ParseResult:
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        raw_text = " ".join(
            (p.extract_text() or "") for p in pdf.pages[:3]
        )
        bank_name = _detect_bank(raw_text)

        records = _extract_via_tables(pdf)

        if len(records) < 2:
            records = _extract_via_text(pdf)

    df = _build_dataframe(records)
    return ParseResult(transactions=df, page_count=page_count, bank_name=bank_name)
