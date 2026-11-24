#!/usr/bin/env python3
"""
Generate a CSV file `dates_dataset.csv` with 1 000 rows.

Each row contains
    • prompt : “Convert the following date to MM‑DD‑YYYY format. Answer with only the date. Date: <date>”
    • answer : the same date expressed as MM‑DD‑YYYY (computed directly from the datetime object)

Additional requirements satisfied
---------------------------------
1. No external date‑parsing library – the answer is produced with
   `dt.strftime("%m-%d-%Y")`.
2. At least 20 rows contain the leap‑day 29 Feb.
3. Some date strings are randomly converted to ALL UPPERCASE or all lowercase.
4. Date range spans **1900‑01‑01** through **2099‑12‑31**.
5. Unused imports and the auto‑install of `dateutil` have been removed.
"""

import csv
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------
# 1️⃣  Helper – ordinal suffixes (1st, 2nd, 3rd, …)
# ----------------------------------------------------------------------
_ORDINAL_RE = re.compile(r'(\d+)(?!\d)')

def _ordinal(n: int) -> str:
    """Return an ordinal string for an integer (1 → '1st')."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# ----------------------------------------------------------------------
# 2️⃣  Fixed random seed (reproducible)
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ----------------------------------------------------------------------
# 3️⃣  Date range
# ----------------------------------------------------------------------
START_DATE = datetime(1900, 1, 1)
END_DATE   = datetime(2099, 12, 31)

def random_date() -> datetime:
    """Pick a random day (including leap days) between START_DATE and END_DATE."""
    delta_days = (END_DATE - START_DATE).days
    return START_DATE + timedelta(days=random.randint(0, delta_days))

def random_leap_day() -> datetime:
    """Return a random Feb 29 between 1900 and 2099 (there are many)."""
    # Pick a random year that is a leap year
    while True:
        year = random.randint(1900, 2099)
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return datetime(year, 2, 29)

# ----------------------------------------------------------------------
# 4️⃣  Textual representations (10 different styles)
# ----------------------------------------------------------------------
def fmt_1(dt: datetime) -> str:
    # "March 5, 2024"
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"

def fmt_2(dt: datetime) -> str:
    # "5 Mar 2024"
    return f"{dt.day} {dt.strftime('%b')} {dt.year}"

def fmt_3(dt: datetime) -> str:
    # "03/05/2024"  (zero‑padded month/day)
    return dt.strftime("%m/%d/%Y")

def fmt_4(dt: datetime) -> str:
    # "Tuesday, March 5th, 2024"
    weekday = dt.strftime("%A")
    month   = dt.strftime("%B")
    return f"{weekday}, {month} {_ordinal(dt.day)}, {dt.year}"

def fmt_5(dt: datetime) -> str:
    # Date embedded inside a sentence (numeric style)
    templates = [
        "The deadline is on {}.",
        "We met on {} and discussed the plan.",
        "My birthday was {}!",
        "Remember to file the report by {}.",
        "The event happened {}."
    ]
    return random.choice(templates).format(fmt_3(dt))

def fmt_6(dt: datetime) -> str:
    # ISO‑like "2024-03-05"
    return dt.strftime("%Y-%m-%d")

def fmt_7(dt: datetime) -> str:
    # "5th of March, 2024"
    return f"{_ordinal(dt.day)} of {dt.strftime('%B')}, {dt.year}"

def fmt_8(dt: datetime) -> str:
    # "Mar 05 2024"
    return dt.strftime("%b %d %Y")

def fmt_9(dt: datetime) -> str:
    # "05-Mar-2024"
    return dt.strftime("%d-%b-%Y")

def fmt_10(dt: datetime) -> str:
    # "On the 5th of March 2024, we ..."
    return f"On the {_ordinal(dt.day)} of {dt.strftime('%B')} {dt.year}, we ..."

FORMATTERS = [fmt_1, fmt_2, fmt_3, fmt_4, fmt_5,
             fmt_6, fmt_7, fmt_8, fmt_9, fmt_10]

# ----------------------------------------------------------------------
# 5️⃣  Build the dataset
# ----------------------------------------------------------------------
NUM_ROWS      = 1000
LEAP_ROWS    = 20           # guaranteed leap‑day rows
OUTPUT_FILE = Path("dates_dataset.csv")

rows = []   # keep rows in memory for the final preview

with OUTPUT_FILE.open(mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["prompt", "answer"])
    writer.writeheader()

    # ---- 5.1  First, guarantee LEAP_ROWS leap‑day entries ----
    for _ in range(LEAP_ROWS):
        dt = random_leap_day()
        date_str = random.choice(FORMATTERS)(dt)

        # Randomly change case (upper / lower / original)
        case_choice = random.choice(["original", "upper", "lower"])
        if case_choice == "upper":
            date_str = date_str.upper()
        elif case_choice == "lower":
            date_str = date_str.lower()

        prompt = (
            "Convert the following date to MM-DD-YYYY format. "
            f"Answer with only the date. Date: {date_str}"
        )
        answer = dt.strftime("%m-%d-%Y")
        writer.writerow({"prompt": prompt, "answer": answer})
        rows.append({"prompt": prompt, "answer": answer})

    # ---- 5.2  Fill the remaining rows with any random date ----
    for _ in range(NUM_ROWS - LEAP_ROWS):
        dt = random_date()
        date_str = random.choice(FORMATTERS)(dt)

        # Randomly change case (upper / lower / original)
        case_choice = random.choice(["original", "upper", "lower"])
        if case_choice == "upper":
            date_str = date_str.upper()
        elif case_choice == "lower":
            date_str = date_str.lower()

        prompt = (
            "Convert the following date to MM-DD-YYYY format. "
            f"Answer with only the date. Date: {date_str}"
        )
        answer = dt.strftime("%m-%d-%Y")
        writer.writerow({"prompt": prompt, "answer": answer})
        rows.append({"prompt": prompt, "answer": answer})

# ----------------------------------------------------------------------
# 6️⃣  Print a quick sanity‑check (20 random rows)
# ----------------------------------------------------------------------
print("\n--- 20 random rows from the generated CSV ---\n")
sample = random.sample(rows, 20)
for i, r in enumerate(sample, 1):
    print(f"{i:2}. Prompt: {r['prompt']}")
    print(f"   Answer: {r['answer']}\n")