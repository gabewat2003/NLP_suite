"""Text normalization applied before any training.

Order matters: dates first (they contain X runs), then amounts, then the
remaining X runs. Output is idempotent: normalize(normalize(s)) == normalize(s).
"""
import re
import unicodedata

# XX/XX/XXXX, XX/XX/2019, 3/4/21 ... any masked or real m/d/y date.
DATE_RE = re.compile(r"\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:XXXX|XX|\d{4}|\d{2})\b")
# {$1,234.56} as written by the CFPB redaction step; also masked ({$XXXX}),
# space-damaged ({$13 0.00}) and unclosed ({$300.00XXXX) variants.
AMT_RE = re.compile(r"\{\$[\dX,.]*(?: \d[\d,.]*)*(?: ?\})?")
# Runs of 2+ capital X not glued to other letters (so "EXXON" is left alone,
# but "XXXX1234" becomes "<REDACTED>1234").
REDACT_RE = re.compile(r"(?<![A-Za-z])X{2,}(?![A-WYZa-wyz])")
# "XXXX XXXX" (e.g. a masked first and last name) collapses to one token.
REPEAT_RE = re.compile(r"<REDACTED>(?:[\s/\-]*<REDACTED>)+")
WS_RE = re.compile(r"\s+")


def normalize(text):
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = DATE_RE.sub("<DATE>", s)
    s = AMT_RE.sub("<AMT>", s)
    s = REDACT_RE.sub("<REDACTED>", s)
    s = REPEAT_RE.sub("<REDACTED>", s)
    s = WS_RE.sub(" ", s).strip()
    return s
