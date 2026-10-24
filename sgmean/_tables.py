"""Aligned tables (English / Spanish) for the inference and diagnostic results.

Same conventions as ``summary()``: labels follow ``set_language`` / ``SGMEAN_LANG``
or an explicit ``lang`` argument; numbers use 6 significant digits and a decimal
comma in Spanish.
"""
from .report import get_language


def num(v, lang, digits=6):
    text = f"{float(v):.{digits}g}"
    return text.replace(".", ",") if lang == "es" else text


def pval(p, lang):
    text = "<0.0001" if p < 1e-4 else f"{p:.4f}"
    return text.replace(".", ",") if lang == "es" else text


def pct(trim, lang):
    return num(trim * 100, lang) + "%"


def render(title, rows, lang):
    """rows: list of (english_label, spanish_label, value_text)."""
    i = 1 if lang == "es" else 0
    rows = [(r[i], r[2]) for r in rows]
    w1 = max(len(a) for a, _ in rows)
    w2 = max(len(b) for _, b in rows)
    total = max(len(title[i]), w1 + w2 + 2)
    w2 = total - w1 - 2
    lines = [title[i], "-" * total]
    lines += [f"{a:<{w1}}  {b:>{w2}}" for a, b in rows]
    return "\n".join(lines)


def lang_of(lang):
    return get_language(lang)
