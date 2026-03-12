from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Direction helper
# ---------------------------------------------------------------------------

def _direction(row: pd.Series) -> str:
    if row["credit"] > 0:
        return "inflow"
    if row["debit"] > 0:
        return "outflow"
    return "neutral"


def _normalize(text: object) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


# ---------------------------------------------------------------------------
# Classification rules
#
# Each entry: (category, direction_filter, keywords)
# Rules are tried top-to-bottom; first match wins.
# Empty keyword list = catch-all.
#
# IMPORTANT: keywords are matched against PARTICULARS only (not channel)
# to avoid false positives like "atm" matching the "ATM/POS" channel for
# POS purchases. The channel is checked separately where needed.
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, Optional[str], list[str]]] = [

    # ---- INFLOW -----------------------------------------------------------
    ("salary",            "inflow",  [
        "salary", "payroll", "wages", "sal/", "/sal", "pay credit",
        "monthly pay", "basic pay",
    ]),
    ("interest",          "inflow",  [
        "interest cr", "int.cr", "fd interest", "od interest",
        "interest credit", "saving interest", "sb int",
    ]),
    ("incoming_refund",   "inflow",  [
        "refund", "reversal", "cashback", "chargeback", "return",
        "reimbursement", "reimb", "rebate",
    ]),
    ("incoming_cash",     "inflow",  [
        "cash deposit", "cdm deposit", "atm deposit", "cash cr",
    ]),
    ("incoming_upi",      "inflow",  [
        "upi/", "/upi", "upi-", "upi cr", "upi credit", "upi ",
        "@", "googlepay", "phonepe", "paytm", "bhim",
    ]),
    ("incoming_transfer", "inflow",  [
        "neft", "rtgs", "imps", "transfer from", "trf from",
        "fund transfer", "bank transfer",
    ]),
    ("incoming_other",    "inflow",  []),  # catch-all inflow

    # ---- OUTFLOW ----------------------------------------------------------
    # Charges MUST come before transfer/imps rules
    ("charges",           "outflow", [
        "imps chg", "imps chrg", "neft chg", "rtgs chg",
        "gst", "service charge", "sms charge", "sms alert",
        "maintenance charge", "folio charge", "min bal charge",
        "penalty", "cess",
    ]),
    ("emi",               "outflow", [
        "emi", "loan repay", "auto debit", "nach debit", "ecs debit",
        "ecs dr", "mandate", "nach dr", "loan emi",
    ]),
    ("bill_payment",      "outflow", [
        "electricity", "elec bill", "msedcl", "bescom", "tpddl", "tneb",
        "torrent power", "adani electricity",
        "bsnl", "airtel", "jio", "vodafone", "vi ", "idea", "broadband",
        "recharge",
        "insurance", "lic ", "premium", "policy",
        "netflix", "hotstar", "disney", "spotify", "amazon prime",
        "youtube premium", "zee5",
        "fastag", "toll", "irctc", "makemytrip", "goibibo", "ola", "uber",
        "water bill", "gas bill", "piped gas", "igl ", "mgl ", "adani gas",
    ]),
    ("fuel",              "outflow", [
        "petrol", "diesel", "hpcl", "bpcl", "indian oil", "hp petro",
        "essar", "shell", "fuel", "cng",
    ]),
    ("investment",        "outflow", [
        "mutual fund", "mf invest", "sip", "zerodha", "groww",
        "upstox", "angel", "nse", "bse", "demat", "ipo",
    ]),
    ("outgoing_upi",      "outflow", [
        "upi/", "/upi", "upi-", "upi dr", "upi debit", "upi ",
        "upil", "@", "googlepay", "phonepe", "paytm", "bhim",
        "sent using paytm", "razorpay",
    ]),
    ("outgoing_transfer", "outflow", [
        "neft", "rtgs", "imps", "transfer to", "trf to",
        "fund transfer", "bank transfer",
    ]),
    ("outgoing_other",    "outflow", []),  # catch-all outflow
]


# ---------------------------------------------------------------------------
# ATM detection — separate logic to avoid "atm" matching channel "ATM/POS"
# for POS purchases. Only matches on PARTICULARS content.
# ---------------------------------------------------------------------------

# Card-number pattern for ATM withdrawals: "508853XXXXXX3709 ..."
_ATM_CARD_RE = re.compile(r"\d{6}x{4,6}\d{4}", re.IGNORECASE)

# Specific keywords that indicate ATM cash withdrawal (in particulars only)
_ATM_KEYWORDS = [
    "cash withdrawal", "atm withdrawal", "atm cash", "cdm",
    "cash dr", "atm wd",
]


def _is_atm_withdrawal(particulars: str) -> bool:
    """Check if a transaction is an ATM cash withdrawal based on particulars only."""
    p = particulars.lower()
    if _ATM_CARD_RE.search(p):
        return True
    return any(kw in p for kw in _ATM_KEYWORDS)


# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------

def classify_transaction(row: pd.Series) -> str:
    direction = _direction(row)
    if direction == "neutral":
        return "neutral"

    particulars = _normalize(row.get("particulars", ""))
    channel = _normalize(row.get("channel", ""))

    # ATM detection uses particulars ONLY (not channel) to avoid
    # POS purchases on "ATM/POS" channel being misclassified
    if direction == "outflow" and _is_atm_withdrawal(particulars):
        return "atm"

    # Inflow: check if channel indicates transfer (mobile banking)
    if direction == "inflow" and channel == "mobile banking":
        if "transfer from" in particulars:
            return "incoming_transfer"

    # Outflow: check if channel indicates transfer (mobile banking)
    if direction == "outflow" and channel == "mobile banking":
        if "transfer to" in particulars:
            return "outgoing_transfer"

    # Recon/adjustment entries
    if "recon" in channel or "cradj" in particulars or "nfscradj" in particulars:
        if direction == "inflow":
            return "incoming_other"
        return "outgoing_other"

    # Rule-table matching (against PARTICULARS only)
    for category, dir_filter, keywords in _RULES:
        if dir_filter and dir_filter != direction:
            continue
        if not keywords:
            return category          # catch-all
        if any(kw in particulars for kw in keywords):
            return category

    return "incoming_other" if direction == "inflow" else "outgoing_other"


# ---------------------------------------------------------------------------
# Merchant / sender extraction
# ---------------------------------------------------------------------------

_UPI_PARTS_RE = re.compile(
    r"upi[/\-](?:cr|dr)?[/\-]?\d+[/\-](.+?)[/\-][A-Z]{4}\d",
    re.IGNORECASE,
)

_TRANSFER_PARTS_RE = re.compile(
    r"(?:neft|rtgs|imps|transfer)[/\-\s]+(.+?)(?:[/\-]|$)",
    re.IGNORECASE,
)


def extract_merchant(row: pd.Series) -> str:
    particulars = _normalize(row.get("particulars", ""))

    # Try UPI pattern first
    m = _UPI_PARTS_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:40]

    # Try NEFT/IMPS/transfer pattern
    m = _TRANSFER_PARTS_RE.search(particulars)
    if m:
        candidate = m.group(1).strip()
        if not re.fullmatch(r"[\d\s]+", candidate):
            return candidate.title()[:40]

    # Fallback
    clean = re.sub(
        r"^(upi|neft|rtgs|imps|ach|ecs|nach|atm|pos|dp|cdm|mb)[/\-\s]*",
        "",
        particulars,
        flags=re.IGNORECASE,
    ).strip()

    if ":" in clean:
        clean = clean.split(":", 1)[1].strip()
    clean = clean.split("/")[0].strip()
    clean = clean.split("  ")[0].strip()

    return clean.title()[:40] if clean else "—"


# ---------------------------------------------------------------------------
# Beneficiary extraction — specifically for transfers
# ---------------------------------------------------------------------------

# "TO 60563723476 TRANSFER TO 60563723476 TO Miss. Kamal Uttam Chavhan"
_TRANSFER_TO_RE = re.compile(
    r"transfer\s+to\s+\d+\s+to\s+(.+)",
    re.IGNORECASE,
)

# "FROM 60563723476 TRANSFER FROM 60563723476 FRM Miss. Kamal Uttam Chavhan"
_TRANSFER_FROM_RE = re.compile(
    r"transfer\s+from\s+\d+\s+frm?\s+(.+)",
    re.IGNORECASE,
)

# "IMPS/48/605815018939/**/ARMAN KUMAR KHAMIYA/Saving..."
_IMPS_NAME_RE = re.compile(
    r"imps/\d+/\d+/\*\*/(.+?)(?:/|$)",
    re.IGNORECASE,
)

# "IMPS/48/606414012525/**6306/Arman Kumar Khamiya/Fu..."
_IMPS_NAME2_RE = re.compile(
    r"imps/\d+/\d+/\*\*\d+/(.+?)(?:/|$)",
    re.IGNORECASE,
)

# NEFT/RTGS: "NEFT SBINN52026021918501591 Mr P UMESH SBIN0011718"
# or "RTGS IOBAR52026021800600914 K SHEKAR IOBA0001897"
_NEFT_RTGS_RE = re.compile(
    r"(?:neft|rtgs)\s+\w+\s+(.+?)\s+[A-Z]{4}\d{7}",
    re.IGNORECASE,
)


def extract_beneficiary(row: pd.Series) -> str:
    """Extract beneficiary/sender name specifically from transfer transactions."""
    particulars = row.get("particulars", "")
    category = row.get("category", "")

    if category not in (
        "outgoing_transfer", "incoming_transfer",
    ):
        return ""

    # Mobile Banking transfers
    m = _TRANSFER_TO_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:50]

    m = _TRANSFER_FROM_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:50]

    # IMPS transfers
    m = _IMPS_NAME_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:50]

    m = _IMPS_NAME2_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:50]

    # NEFT/RTGS
    m = _NEFT_RTGS_RE.search(particulars)
    if m:
        return m.group(1).strip().title()[:50]

    return extract_merchant(row)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        out = transactions.copy()
        for col in ("direction", "category", "merchant", "beneficiary"):
            out[col] = pd.Series(dtype="object")
        out["amount"] = pd.Series(dtype="float64")
        return out

    out = transactions.copy()
    out["direction"] = out.apply(_direction, axis=1)
    out["category"] = out.apply(classify_transaction, axis=1)
    out["merchant"] = out.apply(extract_merchant, axis=1)
    out["amount"] = out["credit"].where(out["credit"] > 0, out["debit"])
    out["beneficiary"] = out.apply(extract_beneficiary, axis=1)
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(enriched_df: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {
        "total_inflow":  float(enriched_df["credit"].sum()),
        "total_outflow": float(enriched_df["debit"].sum()),
        "net":           float(enriched_df["credit"].sum() - enriched_df["debit"].sum()),
    }
    for cat, group in enriched_df.groupby("category"):
        result[str(cat)] = float(group["amount"].sum())
    return result


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "inflow", "outflow", "net"])
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    agg = (
        tmp.groupby("month")
        .agg(inflow=("credit", "sum"), outflow=("debit", "sum"))
        .reset_index()
    )
    agg["net"] = agg["inflow"] - agg["outflow"]
    return agg.sort_values("month")


def top_merchants(df: pd.DataFrame, direction: str, n: int = 10) -> pd.DataFrame:
    sub = df[df["direction"] == direction].copy()
    if sub.empty:
        return pd.DataFrame(columns=["merchant", "total_amount", "transactions"])
    result = (
        sub.groupby("merchant")
        .agg(total_amount=("amount", "sum"), transactions=("sr_no", "count"))
        .sort_values("total_amount", ascending=False)
        .head(n)
        .reset_index()
    )
    result["total_amount"] = result["total_amount"].round(2)
    return result


def beneficiary_summary(df: pd.DataFrame, direction: str = "outflow") -> pd.DataFrame:
    """Breakdown of transfers by beneficiary name."""
    transfer_cats = {"outgoing_transfer"} if direction == "outflow" else {"incoming_transfer"}
    sub = df[df["category"].isin(transfer_cats) & (df["beneficiary"] != "")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["beneficiary", "total_amount", "transactions"])
    result = (
        sub.groupby("beneficiary")
        .agg(total_amount=("amount", "sum"), transactions=("sr_no", "count"))
        .sort_values("total_amount", ascending=False)
        .reset_index()
    )
    result["total_amount"] = result["total_amount"].round(2)
    return result
