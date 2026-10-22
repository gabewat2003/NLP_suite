from hypothesis import given, strategies as st

from corpus.normalize import normalize


def test_date_patterns():
    assert normalize("on XX/XX/XXXX and XX/XX/2021 and 3/4/21") == "on <DATE> and <DATE> and <DATE>"


def test_amount():
    assert normalize("I paid {$1,234.56} and {$50.00}.") == "I paid <AMT> and <AMT>."
    assert normalize("owed {$XXXX} or {$13 0.00} or {$300.00XXXX now") == "owed <AMT> or <AMT> or <AMT> now"


def test_redaction_runs_collapse():
    assert normalize("Mr. XXXX XXXX called") == "Mr. <REDACTED> called"
    assert normalize("account XXXX1234") == "account <REDACTED>1234"


def test_words_with_x_untouched():
    assert normalize("EXXON and Xerox and XXL") == "EXXON and Xerox and XXL"


def test_date_before_redaction():
    # if X-runs were replaced first, the date would become <REDACTED>/<REDACTED>/<REDACTED>
    assert "<DATE>" in normalize("XX/XX/XXXX")
    assert "<REDACTED>" not in normalize("XX/XX/XXXX")


def test_whitespace_and_unicode():
    assert normalize("  a\t\tb\r\nc  ") == "a b c"
    assert normalize("ﬁne") == "fine"  # NFKC ligature


def test_none_and_empty():
    assert normalize(None) == ""
    assert normalize("   ") == ""


@given(st.text())
def test_idempotent(s):
    once = normalize(s)
    assert normalize(once) == once


@given(st.text(alphabet="X /{}$0123456789,.abc<>", max_size=60))
def test_idempotent_on_redaction_alphabet(s):
    once = normalize(s)
    assert normalize(once) == once
