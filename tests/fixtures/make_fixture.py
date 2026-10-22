"""Generate tests/fixtures/mini_cfpb.parquet: synthetic complaints in the raw CFPB
format (with XXXX-style redactions), so tests and CI never need the real download.

Every narrative is generated from templates below. None is real complaint text.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    ("Debt collection", "Attempts to collect debt not owed"): [
        "A collector keeps calling me about a debt I do not owe. I sent a debt validation letter on {date} and they never responded.",
        "{name} reported a collection account of {amt} that is not mine. I disputed it and asked for validation but the calls continue.",
        "The collection agency says I owe {amt} on an old account. This debt was paid in full on {date} and I have the receipt.",
    ],
    ("Debt collection", "Communication tactics"): [
        "The debt collector called my workplace three times on {date} after I asked them to stop. They threatened garnishment of my wages.",
        "I receive calls every day from {name} about a debt. They were rude and said they would sue me if I did not pay {amt} today.",
    ],
    ("Credit reporting or other personal consumer reports", "Incorrect information on your report"): [
        "My credit report from Experian shows a late payment on {date} that is wrong. I filed a dispute and the reinvestigation found nothing.",
        "There is an account on my TransUnion report that I never opened. The balance shown is {amt}. Please remove it from my credit report.",
        "Equifax lists my account {name} as delinquent but I have autopay and never missed a payment. My credit score dropped.",
    ],
    ("Credit reporting or other personal consumer reports", "Problem with a company's investigation into an existing problem"): [
        "I disputed inaccurate information with Equifax on {date}. Under the FCRA they must investigate, but they only verified it with the furnisher.",
        "The reinvestigation results came back on {date} saying the information was verified. They did not review the documents I sent.",
    ],
    ("Mortgage", "Trouble during payment process"): [
        "My mortgage servicer {name} did not apply my payment of {amt} received on {date}. Now they charge a late fee and escrow is short.",
        "I asked for forbearance during my hardship. The servicer approved it and then reported me as delinquent. My escrow balance is wrong.",
    ],
    ("Credit card", "Problem with a purchase shown on your statement"): [
        "I disputed a charge of {amt} on my credit card from {date}. The merchant never delivered. The bank denied the chargeback.",
        "There is an unauthorized charge on my statement for {amt}. I reported it to {name} but they say I am responsible for the balance.",
    ],
    ("Checking or savings account", "Managing an account"): [
        "My checking account was frozen on {date} without notice. I could not access {amt} for two weeks and was charged overdraft fees.",
        "The bank closed my savings account and has not returned my balance of {amt}. I called {name} many times.",
    ],
    ("Vehicle loan or lease", "Struggling to pay your loan"): [
        "I fell behind on my auto loan after losing my job. The lender threatened repossession on {date} even though I offered {amt}.",
        "My car was repossessed after one late payment. The lender {name} says I still owe {amt} after the auction.",
    ],
}
NAMES = ["XXXX", "XXXX XXXX", "XXXX XXXX XXXX"]
DATES = ["XX/XX/XXXX", "XX/XX/2022", "XX/XX/XXXX"]
AMTS = ["{$1200.00}", "{$45.00}", "{$3,500.00}", "{$89.99}"]
COMPANIES = ["Alpha Bank", "Beta Credit Union", "Gamma Collections", "Delta Servicing"]


def make(n=360, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(LABELS)
    start, end = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-06-30")
    span = (end - start).days
    rows = []
    for i in range(n):
        product, issue = keys[i % len(keys)]
        tmpls = LABELS[(product, issue)]
        a, b = rng.choice(len(tmpls), 2)
        text = " ".join([
            tmpls[a].format(name=NAMES[rng.integers(3)], date=DATES[rng.integers(3)], amt=AMTS[rng.integers(4)]),
            tmpls[b].format(name=NAMES[rng.integers(3)], date=DATES[rng.integers(3)], amt=AMTS[rng.integers(4)]),
            f"Reference number XXXX{rng.integers(1000, 9999)}.",
        ])
        rows.append({
            "date_received": start + pd.Timedelta(days=int(rng.integers(span))),
            "product": product, "sub_product": None, "issue": issue, "sub_issue": None,
            "narrative": text, "company": COMPANIES[rng.integers(4)], "complaint_id": 1_000_000 + i,
        })
    df = pd.DataFrame(rows)
    # two exact duplicates and one near duplicate so dedup has work to do
    extra = df.iloc[:3].copy()
    extra["complaint_id"] = [2_000_000, 2_000_001, 2_000_002]
    extra.loc[extra.index[2], "narrative"] = extra.iloc[2]["narrative"] + " Thank you."
    return pd.concat([df, extra], ignore_index=True)


if __name__ == "__main__":
    out = Path(__file__).with_name("mini_cfpb.parquet")
    make().to_parquet(out, index=False)
    print(f"wrote {out}")
