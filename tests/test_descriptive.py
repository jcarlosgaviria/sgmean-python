import math
import pytest
from sgmean import (summary, geometric_mean, dam, sbi, skewness, kurtosis, cv, std_error, mode)

DATA = {8: [4, 6, 9, 12, 15, 22, 30, 71],
 9: [3, 8, 9, 14, 17, 25, 33, 41, 90],
 10: [2, 5, 7, 11, 13, 20, 24, 31, 45, 80],
 11: [0.125, 0.48, 0.731, 1.205, 1.61, 2.034, 2.388, 3.147, 4.56, 6.902, 9.318],
 12: [2.35, 3.1, 4.72, 5.05, 6.48, 7.91, 8.26, 9.63, 11.07, 12.84, 15.3, 21.75],
 14: [7.2, 3.4, 3.4, 9.8, 5.5, 5.5, 5.5, 12.1, 8.6, 4.9, 15.7, 6.3, 6.3, 10.2],
 20: [1.8, 2.4, 2.9, 3.3, 3.7, 4.1, 4.6, 5.2, 5.8, 6.5, 7.3, 8.2, 9.4, 10.9, 12.7, 15.1, 18.6, 23.4,
      31.9, 52.3]}

# Statgraphics Centurion 'Summary Statistics' output, 2026-10-21 (trim = 5 %).
SG = {8: {'count': 8,
     'cv': 103.663,
     'dam': 8,
     'geometric_mean': 14.3736,
     'intersextile_range': 24,
     'iqr': 18.5,
     'kurtosis': 4.58524,
     'max': 71,
     'mean': 21.125,
     'median': 13.5,
     'min': 4,
     'mode': None,
     'q1': 7.5,
     'q3': 26,
     'range': 67,
     'sbi': 11.3181,
     'sextile_1_6': 6,
     'sextile_5_6': 30,
     'skewness': 2.05883,
     'std': 21.8987,
     'std_error': 7.74236,
     'std_kurtosis': 2.64729,
     'std_skewness': 2.37733,
     'sum': 169,
     'sum_squares': 6927,
     'trimmed_mean': 19.3056,
     'variance': 479.554,
     'winsorized_mean': 21.125,
     'winsorized_sigma': 21.8987},
 9: {'count': 9,
     'cv': 100.396,
     'dam': 9,
     'geometric_mean': 17.5321,
     'intersextile_range': 33,
     'iqr': 24,
     'kurtosis': 4.13582,
     'max': 90,
     'mean': 26.6667,
     'median': 17,
     'min': 3,
     'mode': None,
     'q1': 9,
     'q3': 33,
     'range': 87,
     'sbi': 15.0166,
     'sextile_1_6': 8,
     'sextile_5_6': 41,
     'skewness': 1.91813,
     'std': 26.7722,
     'std_error': 8.92406,
     'std_kurtosis': 2.53266,
     'std_skewness': 2.34922,
     'sum': 240,
     'sum_squares': 12134,
     'trimmed_mean': 24.463,
     'variance': 716.75,
     'winsorized_mean': 26.6667,
     'winsorized_sigma': 26.7722},
 10: {'count': 10,
      'cv': 99.6821,
      'dam': 10.5,
      'geometric_mean': 14.8913,
      'intersextile_range': 40,
      'iqr': 24,
      'kurtosis': 2.99904,
      'max': 80,
      'mean': 23.8,
      'median': 16.5,
      'min': 2,
      'mode': None,
      'q1': 7,
      'q3': 31,
      'range': 78,
      'sbi': 17.431,
      'sextile_1_6': 5,
      'sextile_5_6': 45,
      'skewness': 1.68757,
      'std': 23.7243,
      'std_error': 7.5023,
      'std_kurtosis': 1.93587,
      'std_skewness': 2.17865,
      'sum': 238,
      'sum_squares': 10730,
      'trimmed_mean': 21.8889,
      'variance': 562.844,
      'winsorized_mean': 23.8,
      'winsorized_sigma': 23.7243},
 11: {'count': 11,
      'cv': 97.9795,
      'dam': 1.303,
      'geometric_mean': 1.71662,
      'intersextile_range': 6.422,
      'iqr': 3.829,
      'kurtosis': 1.1294,
      'max': 9.318,
      'mean': 2.95455,
      'median': 2.034,
      'min': 0.125,
      'mode': None,
      'q1': 0.731,
      'q3': 4.56,
      'range': 9.193,
      'sbi': 2.43752,
      'sextile_1_6': 0.48,
      'sextile_5_6': 6.902,
      'skewness': 1.33908,
      'std': 2.89485,
      'std_error': 0.872829,
      'std_kurtosis': 0.764604,
      'std_skewness': 1.81312,
      'sum': 32.5,
      'sum_squares': 179.824,
      'trimmed_mean': 2.75822,
      'variance': 8.38014,
      'winsorized_mean': 2.95455,
      'winsorized_sigma': 2.89485},
 12: {'count': 12,
      'cv': 61.7552,
      'dam': 3.2,
      'geometric_mean': 7.53483,
      'intersextile_range': 10.16,
      'iqr': 7.07,
      'kurtosis': 1.1234,
      'max': 21.75,
      'mean': 9.03833,
      'median': 8.085,
      'min': 2.35,
      'mode': None,
      'q1': 4.885,
      'q3': 11.955,
      'range': 19.4,
      'sbi': 5.16642,
      'sextile_1_6': 3.91,
      'sextile_5_6': 14.07,
      'skewness': 1.06137,
      'std': 5.58164,
      'std_error': 1.61128,
      'std_kurtosis': 0.79436,
      'std_skewness': 1.501,
      'sum': 108.46,
      'sum_squares': 1323.0,
      'trimmed_mean': 8.7037,
      'variance': 31.1547,
      'winsorized_mean': 9.03833,
      'winsorized_sigma': 5.58164},
 14: {'count': 14,
      'cv': 46.6784,
      'dam': 1.85,
      'geometric_mean': 6.78675,
      'intersextile_range': 5.3,
      'iqr': 4.3,
      'kurtosis': 0.995403,
      'max': 15.7,
      'mean': 7.45714,
      'median': 6.3,
      'min': 3.4,
      'mode': 5.5,
      'q1': 5.5,
      'q3': 9.8,
      'range': 12.3,
      'sbi': 3.20862,
      'sextile_1_6': 4.9,
      'sextile_5_6': 10.2,
      'skewness': 1.10485,
      'std': 3.48087,
      'std_error': 0.930303,
      'std_kurtosis': 0.760252,
      'std_skewness': 1.68769,
      'sum': 104.4,
      'sum_squares': 936.04,
      'trimmed_mean': 7.2246,
      'variance': 12.1165,
      'winsorized_mean': 7.45714,
      'winsorized_sigma': 3.48087},
 20: {'count': 20,
      'cv': 107.365,
      'dam': 3.8,
      'geometric_mean': 7.65947,
      'intersextile_range': 15.3,
      'iqr': 10.0,
      'kurtosis': 5.78183,
      'max': 52.3,
      'mean': 11.505,
      'median': 6.9,
      'min': 1.8,
      'mode': None,
      'q1': 3.9,
      'q3': 13.9,
      'range': 50.5,
      'sbi': 6.32414,
      'sextile_1_6': 3.3,
      'sextile_5_6': 18.6,
      'skewness': 2.28721,
      'std': 12.3523,
      'std_error': 2.76206,
      'std_kurtosis': 5.27806,
      'std_skewness': 4.17585,
      'sum': 230.1,
      'sum_squares': 5546.31,
      'trimmed_mean': 9.77778,
      'variance': 152.579,
      'winsorized_mean': 10.515,
      'winsorized_sigma': 9.76141}}

def _close(mine, sg):
    if sg is None:
        return mine is None
    return abs(mine - sg) <= 5.1e-6 * max(1, 10 ** math.floor(math.log10(abs(sg) or 1))) or abs(mine - sg) <= 1e-5 * abs(sg)

@pytest.mark.parametrize("n", sorted(SG))
def test_summary_matches_statgraphics(n):
    got = summary(DATA[n])
    bad = {k: (got[k], sg) for k, sg in SG[n].items() if not _close(got[k], sg)}
    assert not bad, bad
    assert set(got) >= set(SG[n])

def test_summary_has_29_results():
    assert len(summary(DATA[10])) == 29

def test_individual_functions_agree_with_summary():
    x = DATA[14]; s = summary(x)
    assert geometric_mean(x) == s["geometric_mean"] and dam(x) == s["dam"] and sbi(x) == s["sbi"]
    assert skewness(x) == s["skewness"] and kurtosis(x, True) == s["std_kurtosis"]
    assert cv(x) == s["cv"] and std_error(x) == s["std_error"] and mode(x) == 5.5

def test_edge_cases():
    with pytest.raises(ValueError): geometric_mean([1, 0, 2])
    with pytest.raises(ValueError): skewness([1, 2])
    with pytest.raises(ValueError): kurtosis([1, 2, 3])
    assert math.isnan(sbi([1, 1, 1, 5]))           # MAD = 0
    assert mode([1, 2, 3]) is None
    assert mode([1, 1, 2, 2, 3]) is None           # tie between modes: blank, as Statgraphics


def test_n3_matches_statgraphics():
    # Statgraphics Centurion, 2026-10-21: n = 3 leaves kurtosis (and its standardized value) blank.
    SG3 = {"count": 3, "mean": 11.0, "median": 9.0, "mode": None, "geometric_mean": 8.96281,
           "trimmed_mean": 10.8889, "winsorized_mean": 11.0, "variance": 67.0, "std": 8.18535,
           "cv": 74.4123, "std_error": 4.72582, "winsorized_sigma": 8.18535, "dam": 5.0, "sbi": 7.28664,
           "min": 4.0, "max": 20.0, "range": 16.0, "q1": 4.0, "q3": 20.0, "iqr": 16.0,
           "sextile_1_6": 4.0, "sextile_5_6": 20.0, "intersextile_range": 16.0,
           "skewness": 1.03388, "std_skewness": 0.731065, "kurtosis": None, "std_kurtosis": None,
           "sum": 33.0, "sum_squares": 497.0}
    got = summary([4, 9, 20])
    bad = {k: (got[k], v) for k, v in SG3.items() if not _close(got[k], v)}
    assert not bad, bad


def test_geometric_mean_blank_for_non_positive_data():
    # Statgraphics leaves the geometric mean blank with a zero or a negative value.
    assert summary([0, 1, 2, 3, 4, 5])["geometric_mean"] is None
    assert summary([-1, 2, 3, 4, 5, 6])["geometric_mean"] is None


@pytest.mark.parametrize("trim,want", [(0.15, 18.7143), (0.25, 17.4), (0.35, 16.8333)])
def test_trimmed_mean_other_percentages_set_a(trim, want):
    # Statgraphics Centurion, n=10 (set A), 2026-10-21. k = 1.5, 2.5, 3.5: fractional discount,
    # not truncation (scipy.stats.trim_mean would give 19.5, 17.66667 and 17.0).
    from sgmean import sgmean
    assert sgmean(DATA[10], trim=trim) == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("trim,want", [(0.13, 8.50811), (0.17, 8.07879)])
def test_trimmed_mean_other_percentages_set_g(trim, want):
    # Statgraphics Centurion, n=20 (set G), 2026-10-21. k = 2.6 and 3.4 (scipy: 8.85625 and 8.24286).
    from sgmean import sgmean
    assert sgmean(DATA[20], trim=trim) == pytest.approx(want, rel=1e-5)


def test_mad_zero_matches_statgraphics():
    # Statgraphics Centurion, 2026-10-21, data 1,1,1,1,5,9: DAM = 0 and Sbi blank.
    s = summary([1, 1, 1, 1, 5, 9])
    assert s["dam"] == 0 and s["sbi"] is None and s["median"] == 1


def test_mode_rule_matches_statgraphics():
    # Statgraphics Centurion, 2026-10-21: blank with a tie, and also blank with no repeats;
    # a unique mode is shown even if it occurs only twice.
    assert mode([1, 2, 2, 3, 3, 4, 5, 9]) is None     # two modes -> blank
    assert mode([1, 2, 2, 3, 4, 5, 9, 10]) == 2.0     # unique mode, twice -> 2.0


# ---- printed table and language ------------------------------------------------------
import locale
import sgmean as sg

G20 = DATA[20]


def test_summary_is_still_a_dict_and_plain_dict_available():
    s = summary(G20)
    assert isinstance(s, dict) and s["sbi"] == pytest.approx(6.32414, rel=1e-5)
    assert type(s.to_dict()) is dict and len(s.to_dict()) == 29


def test_default_table_is_english_and_rounded():
    sg.set_language(None)
    txt = str(summary(G20))
    assert "Summary Statistics (n = 20)" in txt and "Lower Quartile" in txt
    assert "13.9" in txt and "13.899999" not in txt
    assert repr(summary(G20)) == txt
    assert "Trimmed Mean 5%" in txt and "107.365%" in txt


def test_spanish_labels_and_decimal_comma():
    txt = summary(G20, lang="es").table()
    assert "Resumen Estadístico (n = 20)" in txt and "Cuartil Inferior" in txt
    assert "Media Recortada 5%" in txt and "107,365%" in txt and "11,505" in txt
    assert summary(G20).table(lang="es") == txt           # table(lang=...) gives the same


def test_language_priority(monkeypatch):
    sg.set_language(None)
    monkeypatch.delenv("SGMEAN_LANG", raising=False)
    assert "Lower Quartile" in str(summary(G20))                       # default: English
    monkeypatch.setenv("SGMEAN_LANG", "es")
    assert "Cuartil Inferior" in str(summary(G20))                     # environment variable
    sg.set_language("en")
    assert "Lower Quartile" in str(summary(G20))                       # set_language beats env
    assert "Cuartil Inferior" in str(summary(G20, lang="es"))          # argument beats all
    sg.set_language(None)


def test_auto_follows_system_locale(monkeypatch):
    sg.set_language(None)
    monkeypatch.setattr(locale, "getlocale", lambda *a, **k: ("es_CO", "UTF-8"))
    assert "Cuartil Inferior" in str(summary(G20, lang="auto"))
    monkeypatch.setattr(locale, "getlocale", lambda *a, **k: ("en_US", "UTF-8"))
    assert "Lower Quartile" in str(summary(G20, lang="auto"))


def test_invalid_language_and_blanks_in_table():
    with pytest.raises(ValueError):
        summary(G20, lang="fr").table()
    with pytest.raises(ValueError):
        sg.set_language("fr")
    row = [l for l in str(summary([1, 2, 3, 4, 5, 6, 7, 8])).splitlines() if l.startswith("Mode")][0]
    assert row.strip() == "Mode"                                        # blank value, as Statgraphics
