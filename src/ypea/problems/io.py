"""Readers for standard instance files: TSPLIB, QAPLIB and OR-Library (mknap).

The files are not bundled (except OR-Library mknap1). Download them from the libraries:

- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- QAPLIB: https://coral.ise.lehigh.edu/data-sets/qaplib/
- OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .knapsack import knapsack
from .qap import QAPLIB_OPTIMA, qap
from .tsp import TSPLIB_OPTIMA, tsp

__all__ = ["read_tsplib", "read_tsplib_tour", "tsplib_distances", "read_qaplib", "read_qaplib_solution",
           "read_orlib_mknap"]

_SECTIONS = {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION", "TOUR_SECTION",
             "DEPOT_SECTION", "DEMAND_SECTION", "FIXED_EDGES_SECTION"}


def _parse_tsplib(text):
    """Keywords (dict of str) and sections (dict of number lists) of a TSPLIB file."""
    spec, sections, current = {}, {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        head = line.split(":", 1)[0].strip().upper()
        if head == "EOF":
            break
        if head in _SECTIONS:
            current = sections.setdefault(head, [])
            continue
        if current is not None and _is_number(line.split()[0]):
            current.extend(float(t) for t in line.split())
            continue
        if ":" in line:
            current = None
            spec[head] = line.split(":", 1)[1].strip()
    return spec, sections


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _nint(x):
    return np.floor(x + 0.5)


def tsplib_distances(coords, kind):
    """TSPLIB distance matrix of ``coords`` for the edge weight type ``kind``.

    Supports EUC_2D, EUC_3D, MAN_2D, MAN_3D, MAX_2D, MAX_3D, CEIL_2D, ATT and GEO, with the
    rounding rules of the TSPLIB documentation.
    """
    c = np.asarray(coords, dtype=float)
    kind = kind.upper()
    diff = c[:, None, :] - c[None, :, :]
    if kind in ("EUC_2D", "EUC_3D"):
        return _nint(np.sqrt((diff**2).sum(-1)))
    if kind in ("MAN_2D", "MAN_3D"):
        return _nint(np.abs(diff).sum(-1))
    if kind in ("MAX_2D", "MAX_3D"):
        return _nint(np.abs(diff).max(-1))
    if kind == "CEIL_2D":
        return np.ceil(np.sqrt((diff**2).sum(-1)))
    if kind == "ATT":
        r = np.sqrt((diff**2).sum(-1) / 10.0)
        t = _nint(r)
        return np.where(t < r, t + 1, t)
    if kind == "GEO":
        deg = np.trunc(c)
        rad = 3.141592 * (deg + 5.0 * (c - deg) / 3.0) / 180.0
        lat, lon = rad[:, 0], rad[:, 1]
        q1 = np.cos(lon[:, None] - lon[None, :])
        q2 = np.cos(lat[:, None] - lat[None, :])
        q3 = np.cos(lat[:, None] + lat[None, :])
        arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
        d = np.trunc(6378.388 * np.arccos(arg) + 1.0)
        np.fill_diagonal(d, 0.0)
        return d
    raise ValueError(f"unsupported TSPLIB edge weight type {kind!r}")


def _explicit_matrix(values, n, fmt):
    fmt = fmt.upper()
    # for a symmetric matrix, each column format reads like the mirrored row format
    fmt = {"UPPER_COL": "LOWER_ROW", "LOWER_COL": "UPPER_ROW",
           "UPPER_DIAG_COL": "LOWER_DIAG_ROW", "LOWER_DIAG_COL": "UPPER_DIAG_ROW"}.get(fmt, fmt)
    v = np.asarray(values, dtype=float)
    if fmt == "FULL_MATRIX":
        return v[: n * n].reshape(n, n)
    upper, diag = fmt.startswith("UPPER"), "DIAG" in fmt
    if fmt not in ("UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW"):
        raise ValueError(f"unsupported TSPLIB edge weight format {fmt!r}")
    rows, cols = np.triu_indices(n, 0 if diag else 1) if upper else np.tril_indices(n, 0 if diag else -1)
    if len(v) < len(rows):
        raise ValueError(f"EDGE_WEIGHT_SECTION has {len(v)} numbers, {fmt} needs {len(rows)}")
    d = np.zeros((n, n))
    d[rows, cols] = v[: len(rows)]
    d[cols, rows] = v[: len(rows)]
    return d


def read_tsplib(path, optimum=None):
    """Read a TSPLIB file (TSP or ATSP) and return the TSP ``Problem``.

    ``optimum`` defaults to the TSPLIB value for the known instances in ``TSPLIB_OPTIMA``.
    Node coordinates (or display coordinates) are kept in ``info["coords"]`` for plotting.
    """
    spec, sections = _parse_tsplib(Path(path).read_text())
    n = int(spec["DIMENSION"])
    kind = spec.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    coords = None
    for key in ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"):
        if key in sections:
            table = np.asarray(sections[key]).reshape(n, -1)
            coords = table[:, 1:]
            break
    if kind == "EXPLICIT":
        d = _explicit_matrix(sections["EDGE_WEIGHT_SECTION"], n, spec.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"))
    else:
        d = tsplib_distances(np.asarray(sections["NODE_COORD_SECTION"]).reshape(n, -1)[:, 1:], kind)
    name = spec.get("NAME", Path(path).stem).removesuffix(".tsp").removesuffix(".atsp")
    if optimum is None:
        optimum = TSPLIB_OPTIMA.get(name)
    if coords is not None and coords.shape[1] > 2:
        coords = coords[:, :2]
    return tsp(coords, d, name=name, optimum=optimum, source=f"TSPLIB {name}")


def read_tsplib_tour(path):
    """Read a TSPLIB tour file (``.opt.tour``) and return the tour as a 0 based array."""
    _, sections = _parse_tsplib(Path(path).read_text())
    tour = np.asarray(sections["TOUR_SECTION"], dtype=np.int64)
    end = np.flatnonzero(tour == -1)
    return tour[: end[0] if len(end) else len(tour)] - 1


def read_qaplib(path, optimum=None):
    """Read a QAPLIB ``.dat`` file (n, then the matrices A and B) and return the QAP ``Problem``.

    A is the flow and B the distance, so the cost is ``sum A[i, j] * B[p[i], p[j]]``.
    ``optimum`` defaults to the QAPLIB value for the instances in ``QAPLIB_OPTIMA``.
    """
    tokens = np.array(Path(path).read_text().split(), dtype=float)
    n = int(tokens[0])
    if len(tokens) < 1 + 2 * n * n:
        raise ValueError(f"{path}: expected {2 * n * n} matrix entries after n = {n}")
    a = tokens[1:1 + n * n].reshape(n, n)
    b = tokens[1 + n * n:1 + 2 * n * n].reshape(n, n)
    name = Path(path).stem
    if optimum is None:
        optimum = QAPLIB_OPTIMA.get(name)
    return qap(a, b, name=name, optimum=optimum, source=f"QAPLIB {name}")


def read_qaplib_solution(path):
    """Read a QAPLIB ``.sln`` file. Returns ``(cost, p)`` with ``p`` 0 based."""
    tokens = Path(path).read_text().replace(",", " ").split()
    n, cost = int(tokens[0]), float(tokens[1])
    return cost, np.array(tokens[2:2 + n], dtype=np.int64) - 1


def read_orlib_mknap(path):
    """Read an OR-Library multidimensional knapsack file (mknap1, mknapcb1 to mknapcb9).

    Returns a list of ``Problem`` objects. The optimum is set when the file gives one (not 0).
    """
    t = np.array(Path(path).read_text().split(), dtype=float)
    count, pos, problems = int(t[0]), 1, []
    for k in range(count):
        n, m, opt = int(t[pos]), int(t[pos + 1]), t[pos + 2]
        pos += 3
        p = t[pos:pos + n]
        r = t[pos + n:pos + n + m * n].reshape(m, n)
        b = t[pos + n + m * n:pos + n + m * n + m]
        pos += n + m * n + m
        if len(b) < m:
            raise ValueError(f"{path}: problem {k + 1} is incomplete")
        name = f"{Path(path).stem}_{k + 1}"
        problems.append(knapsack(p, r, b, name=name, optimum=float(opt) if opt else None,
                                 source=f"OR-Library {name}"))
    return problems
