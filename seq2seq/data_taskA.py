"""Task A: spelled-out money amounts -> digits.

Convention (American, applied everywhere): no "and" inside the dollar part
("one hundred five"), "and" only before cents, singular "dollar"/"cent" for 1,
"zero dollars" when dollars are 0, cents omitted when 0. Hyphens and commas
from num2words are removed. Target: $1,234.05 (always two decimals).

    two hundred fifty dollars and thirty cents  ->  $250.30
"""
import numpy as np
from num2words import num2words

TRAIN_DIGITS = (1, 2, 3, 4)
EXTRAP_DIGITS = (5, 6, 7)


def spell(n):
    return num2words(n, lang="en").replace(" and ", " ").replace("-", " ").replace(",", "")


def source(dollars, cents):
    s = f"{spell(dollars)} {'dollar' if dollars == 1 else 'dollars'}"
    if cents:
        s += f" and {spell(cents)} {'cent' if cents == 1 else 'cents'}"
    return s


def target(dollars, cents):
    return f"${dollars:,}.{cents:02d}"


def n_digits(dollars):
    return len(str(dollars))


def sample_dollars(rng, d, n):
    lo, hi = (0, 10) if d == 1 else (10 ** (d - 1), 10 ** d)
    return rng.integers(lo, hi, size=n)


def unique_pool(rng, d, n):
    """Up to n distinct (dollars, cents) pairs whose dollar part has d digits."""
    space = (10 if d == 1 else 9 * 10 ** (d - 1)) * 100
    if space <= n:
        lo = 0 if d == 1 else 10 ** (d - 1)
        pool = [(x, c) for x in range(lo, 10 ** d) for c in range(100)]
        return [pool[i] for i in rng.permutation(len(pool))]
    seen = set()
    while len(seen) < n:
        for x, c in zip(sample_dollars(rng, d, n), rng.integers(0, 100, size=n)):
            seen.add((int(x), int(c)))
            if len(seen) >= n:
                break
    out = sorted(seen)
    return [out[i] for i in rng.permutation(len(out))]


def make_splits(seed=0, per_digit_train=15_000, per_digit_eval=200, per_digit_extrap=300):
    """Returns {split: [(src_words, tgt_chars, digits)]}, with no (dollars, cents)
    pair shared across splits."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test_in": [], "test_out": []}
    for d in TRAIN_DIGITS:
        pool = unique_pool(rng, d, per_digit_train + 2 * per_digit_eval)
        val, test, train = pool[:per_digit_eval], pool[per_digit_eval:2 * per_digit_eval], pool[2 * per_digit_eval:]
        for name, part in (("val", val), ("test_in", test), ("train", train)):
            splits[name] += [(x, c, d) for x, c in part]
    for d in EXTRAP_DIGITS:
        splits["test_out"] += [(x, c, d) for x, c in unique_pool(rng, d, per_digit_extrap)]
    return {k: [(source(x, c).split(), list(target(x, c)), d) for x, c, d in v] for k, v in splits.items()}


# ---- inverse functions, used by tests to check the generator ----------------
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def words_to_int(words):
    total = cur = 0
    for w in words:
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur *= 100
        elif w in _SCALES:
            total += cur * _SCALES[w]
            cur = 0
        else:
            raise ValueError(w)
    return total + cur


def parse_source(words):
    i = next(i for i, w in enumerate(words) if w in ("dollar", "dollars"))
    dollars = words_to_int(words[:i])
    cents = words_to_int(words[i + 2:-1]) if len(words) > i + 1 else 0
    return dollars, cents


def parse_target(chars):
    s = "".join(chars).lstrip("$").replace(",", "")
    d, c = s.split(".")
    return int(d), int(c)
