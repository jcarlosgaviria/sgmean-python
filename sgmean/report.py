"""Printable table for ``summary()`` with English (default) or Spanish labels.

Only the *labels* of the printed table are translated; dictionary keys and function
names are the same in every language. Language resolution, from highest to lowest
priority: the ``lang`` argument, ``set_language()``, the ``SGMEAN_LANG`` environment
variable, and finally English. ``lang="auto"`` follows the system locale.
"""
import locale
import os

_LANGS = ("en", "es", "auto")
_current = None   # set by set_language()

# key -> (English, Spanish); {t} is replaced by the trimming percentage
_LABELS = {
    "count": ("Count", "Recuento"),
    "mean": ("Mean", "Promedio"),
    "median": ("Median", "Mediana"),
    "mode": ("Mode", "Moda"),
    "geometric_mean": ("Geometric Mean", "Media Geométrica"),
    "trimmed_mean": ("Trimmed Mean {t}%", "Media Recortada {t}%"),
    "winsorized_mean": ("Winsorized Mean {t}%", "Media Winsorizada {t}%"),
    "variance": ("Variance", "Varianza"),
    "std": ("Standard Deviation", "Desviación Estándar"),
    "cv": ("Coefficient of Variation", "Coeficiente de Variación"),
    "std_error": ("Standard Error", "Error Estándar"),
    "winsorized_sigma": ("Winsorized Sigma {t}%", "Sigma Winsorizada {t}%"),
    "dam": ("DAM (median abs. deviation)", "DAM"),
    "sbi": ("Sbi (biweight scale)", "Sbi"),
    "min": ("Minimum", "Mínimo"),
    "max": ("Maximum", "Máximo"),
    "range": ("Range", "Rango"),
    "q1": ("Lower Quartile", "Cuartil Inferior"),
    "q3": ("Upper Quartile", "Cuartil Superior"),
    "iqr": ("Interquartile Range", "Rango Intercuartílico"),
    "sextile_1_6": ("1/6 Sextile", "1/6 sextil"),
    "sextile_5_6": ("5/6 Sextile", "5/6 sextil"),
    "intersextile_range": ("Intersextile Range", "Rango Intersextil"),
    "skewness": ("Skewness", "Sesgo"),
    "std_skewness": ("Standardized Skewness", "Sesgo Estandarizado"),
    "kurtosis": ("Kurtosis", "Curtosis"),
    "std_kurtosis": ("Standardized Kurtosis", "Curtosis Estandarizada"),
    "sum": ("Sum", "Suma"),
    "sum_squares": ("Sum of Squares", "Suma de Cuadrados"),
}
_TITLE = ("Summary Statistics (n = {n})", "Resumen Estadístico (n = {n})")


def set_language(lang):
    """Set the default language of printed tables: ``"en"``, ``"es"`` or ``"auto"``."""
    global _current
    if lang is not None and lang not in _LANGS:
        raise ValueError(f"lang must be one of {_LANGS}")
    _current = lang


def get_language(lang=None):
    """Resolve the language to use ('en' or 'es')."""
    chosen = lang or _current or os.environ.get("SGMEAN_LANG") or "en"
    chosen = chosen.lower()
    if chosen not in _LANGS:
        raise ValueError(f"lang must be one of {_LANGS}")
    if chosen == "auto":
        loc = (locale.getlocale()[0] or "").lower()
        chosen = "es" if loc.startswith("es") else "en"
    return chosen


def _fmt(key, value, lang):
    if value is None:
        return ""
    if key == "count":
        return str(int(value))
    text = f"{float(value):.6g}"
    if key == "cv":
        text += "%"
    return text.replace(".", ",") if lang == "es" else text


class SummaryResult(dict):
    """Dict of the 29 results that prints as an aligned table.

    ``s["sbi"]`` works as with any dict; ``print(s)`` or just ``s`` shows the table;
    ``s.table(lang="es")`` gives the table in another language; ``s.to_dict()`` returns
    a plain dict.
    """

    def __init__(self, data, trim=0.05, lang=None):
        super().__init__(data)
        self.trim = trim
        self.lang = lang

    def to_dict(self):
        return dict(self)

    def table(self, lang=None):
        lg = get_language(lang if lang is not None else self.lang)
        i = 1 if lg == "es" else 0
        t = f"{self.trim * 100:g}"
        rows = [(_LABELS[k][i].format(t=t), _fmt(k, v, lg)) for k, v in self.items()]
        w1 = max(len(r[0]) for r in rows)
        w2 = max(len(r[1]) for r in rows)
        title = _TITLE[i].format(n=int(self["count"]))
        lines = [title, "-" * (w1 + w2 + 2)]
        lines += [f"{a:<{w1}}  {b:>{w2}}" for a, b in rows]
        return "\n".join(lines)

    __str__ = table

    def __repr__(self):
        return self.table()


class MultiSummaryResult(dict):
    """Several variables: ``{name: SummaryResult}`` printing as one table.

    ``print(m)`` shows one column per variable; ``m["x"]`` is that variable's
    ``SummaryResult``; ``m.table(lang="es")`` changes the language;
    ``m.to_dataframe()`` (needs pandas) and ``m.to_dict()`` export the numbers.
    """

    def __init__(self, data, trim=0.05, lang=None):
        super().__init__(data)
        self.trim = trim
        self.lang = lang

    def to_dict(self):
        return {k: dict(v) for k, v in self.items()}

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({k: dict(v) for k, v in self.items()})

    def table(self, lang=None):
        lg = get_language(lang if lang is not None else self.lang)
        i = 1 if lg == "es" else 0
        t = f"{self.trim * 100:g}"
        names = list(self)
        first = self[names[0]]
        labels = [_LABELS[k][i].format(t=t) for k in first]
        cols = [[_fmt(k, v, lg) for k, v in self[nm].items()] for nm in names]
        w0 = max(len(a) for a in labels)
        ws = [max(len(nm), max(len(c) for c in col)) for nm, col in zip(names, cols)]
        head = " " * w0 + "".join("  " + nm.rjust(w) for nm, w in zip(names, ws))
        title = (("Resumen Estadístico" if lg == "es" else "Summary Statistics")
                 + f" ({len(names)} variables)")
        lines = [title, "-" * len(head), head, "-" * len(head)]
        for r, lab in enumerate(labels):
            lines.append(f"{lab:<{w0}}" + "".join("  " + col[r].rjust(w)
                                                  for col, w in zip(cols, ws)))
        return "\n".join(lines)

    __str__ = table

    def __repr__(self):
        return self.table()
