"""Constrained engineering design problems.

Each function returns a vectorized ``Problem`` (minimization) in the most cited formulation,
with constraints ``g <= 0``, the best known value as ``optimum``, and ``info`` holding the best
known solution (``x_opt``), the variant and the source. The best known points are stored with
full precision: each was computed on the active constraints of the formulation below, and it
rounds to the values published in the cited papers.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from ..variables import Integer, Real

__all__ = ["pressure_vessel", "welded_beam", "spring", "speed_reducer", "three_bar_truss", "gear_train"]

_SQRT2 = np.sqrt(2.0)


def _out(f, g):
    """Stack the constraints on the last axis. NaN values (from 0/0 at a bound) count as violated."""
    g = np.stack(g, axis=-1)
    return f, np.where(np.isnan(g), np.inf, g)


def _cols(x):
    return [x[..., i] for i in range(x.shape[-1])]


# pressure vessel

def _vessel(ts, th, r, length):
    with np.errstate(all="ignore"):
        f = 0.6224 * ts * r * length + 1.7781 * th * r**2 + 3.1661 * ts**2 * length + 19.84 * ts**2 * r
        return _out(f, [-ts + 0.0193 * r, -th + 0.00954 * r,
                        -np.pi * r**2 * length - 4 / 3 * np.pi * r**3 + 1296000, length - 240])


def pressure_vessel_obj(sol):
    return _vessel(*_cols(np.asarray(sol.x, dtype=float)))


def pressure_vessel_mixed_obj(sol):
    t = 0.0625 * np.asarray(sol.thickness, dtype=float)
    rl = np.asarray(sol.x, dtype=float)
    return _vessel(t[..., 0], t[..., 1], rl[..., 0], rl[..., 1])


def pressure_vessel(mixed=True):
    """Pressure vessel design (Sandgren 1990, Kannan and Kramer 1994).

    Minimize the cost of material, forming and welding of a cylindrical vessel with
    hemispherical heads. The design is the shell thickness Ts, the head thickness Th, the
    inner radius R and the length L of the cylinder:

    f = 0.6224 Ts R L + 1.7781 Th R^2 + 3.1661 Ts^2 L + 19.84 Ts^2 R, subject to
    g1 = -Ts + 0.0193 R, g2 = -Th + 0.00954 R, g3 = -pi R^2 L - 4/3 pi R^3 + 1296000,
    g4 = L - 240, and 10 <= R, L <= 200.

    mixed=True (default, the original problem): Ts and Th are integer multiples of 0.0625 inch,
    given by ``Integer("thickness", 2, lb=1, ub=99)`` in those units, and ``Real("x")`` holds
    (R, L). Best known f = 6059.714335 at Ts = 0.8125, Th = 0.4375, R = 42.098446,
    L = 176.636596.

    mixed=False: the continuous version, with ``Real("x")`` = (Ts, Th, R, L) and
    0 <= Ts, Th <= 99. Best known f = 5885.332774 at (0.778169, 0.384649, 40.319619, 200),
    often reported as 5885.3328.

    References: E. Sandgren, "Nonlinear integer and discrete programming in mechanical design
    optimization", Journal of Mechanical Design 112 (1990) 223-229. B. K. Kannan and
    S. N. Kramer, "An augmented Lagrange multiplier based method for mixed integer discrete
    continuous optimization", Journal of Mechanical Design 116 (1994) 405-411.
    """
    source = "Sandgren (1990), Kannan and Kramer (1994)"
    if mixed:
        x_opt = {"thickness": np.array([13, 7]), "x": np.array([42.09844559585492, 176.63659584243936])}
        return Problem([Integer("thickness", 2, lb=1, ub=99), Real("x", 2, lb=10.0, ub=200.0)],
                       pressure_vessel_mixed_obj, vectorized=True, name="pressure_vessel",
                       optimum=6059.714335048434,
                       info={"x_opt": x_opt, "source": source,
                             "variant": "mixed, thicknesses in multiples of 0.0625, four constraints"})
    x_opt = np.array([0.7781686413751053, 0.3846491626279018, 40.31961872409872, 200.0])
    return Problem(Real("x", lb=[0.0, 0.0, 10.0, 10.0], ub=[99.0, 99.0, 200.0, 200.0]),
                   pressure_vessel_obj, vectorized=True, name="pressure_vessel_continuous",
                   optimum=5885.332773616459,
                   info={"x_opt": x_opt, "source": source, "variant": "continuous, four constraints"})


# welded beam

def welded_beam_obj(sol):
    h, length, t, b = _cols(np.asarray(sol.x, dtype=float))
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    with np.errstate(all="ignore"):
        tau1 = P / (_SQRT2 * h * length)
        M = P * (L + length / 2)
        R = np.sqrt(length**2 / 4 + ((h + t) / 2) ** 2)
        J = 2 * (_SQRT2 * h * length * (length**2 / 12 + ((h + t) / 2) ** 2))
        tau2 = M * R / J
        tau = np.sqrt(tau1**2 + 2 * tau1 * tau2 * length / (2 * R) + tau2**2)
        sigma = 6 * P * L / (b * t**2)
        delta = 4 * P * L**3 / (E * t**3 * b)
        pc = 4.013 * E * np.sqrt(t**2 * b**6 / 36) / L**2 * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))
        f = 1.10471 * h**2 * length + 0.04811 * t * b * (14 + length)
        return _out(f, [tau - 13600, sigma - 30000, h - b,
                        0.10471 * h**2 + 0.04811 * t * b * (14 + length) - 5,
                        0.125 - h, delta - 0.25, P - pc])


def welded_beam():
    """Welded beam design, in the seven constraint version of Coello (2000).

    Minimize the fabrication cost f = 1.10471 h^2 l + 0.04811 t b (14 + l) of a beam welded to a
    support, with weld thickness h, weld length l, bar height t and bar thickness b. The
    constraints bound the shear stress (tau <= 13600 psi), the bending stress
    (sigma <= 30000 psi), the end deflection (delta <= 0.25 inch) and the buckling load
    (P <= Pc), with h <= b, h >= 0.125 and a cost limit of 5. The polar moment of inertia is
    J = 2 (sqrt(2) h l (l^2 / 12 + ((h + t) / 2)^2)) and the buckling load is
    Pc = 4.013 E sqrt(t^2 b^6 / 36) / L^2 (1 - t / (2 L) sqrt(E / (4 G))), with P = 6000 lb,
    L = 14 inch, E = 30e6 psi and G = 12e6 psi. Bounds: 0.1 <= h, b <= 2 and 0.1 <= l, t <= 10.

    Best known f = 1.724852 at (h, l, t, b) = (0.205730, 3.470489, 9.036624, 0.205730).
    Other versions of this problem use other constants or constraints, with other best values.

    Reference: C. A. C. Coello, "Use of a self adaptive penalty approach for engineering
    optimization problems", Computers in Industry 41 (2000) 113-127.
    """
    x_opt = np.array([0.20572963978607842, 3.4704886656280074, 9.036623910357712, 0.2057296397861004])
    return Problem(Real("x", lb=[0.1, 0.1, 0.1, 0.1], ub=[2.0, 10.0, 10.0, 2.0]), welded_beam_obj,
                   vectorized=True, name="welded_beam", optimum=1.7248523085975365,
                   info={"x_opt": x_opt, "source": "Coello (2000)", "variant": "seven constraints"})


# tension/compression spring

def spring_obj(sol):
    d, D, N = _cols(np.asarray(sol.x, dtype=float))
    with np.errstate(all="ignore"):
        f = (N + 2) * D * d**2
        return _out(f, [1 - D**3 * N / (71785 * d**4),
                        (4 * D**2 - d * D) / (12566 * (D * d**3 - d**4)) + 1 / (5108 * d**2) - 1,
                        1 - 140.45 * d / (D**2 * N),
                        (d + D) / 1.5 - 1])


def spring():
    """Tension/compression spring design (Belegundu 1982, Arora 1989).

    Minimize the weight f = (N + 2) D d^2 of a spring with wire diameter d, mean coil diameter D
    and N active coils, subject to limits on deflection, shear stress, surge frequency and
    outer diameter (four constraints). Bounds: 0.05 <= d <= 2, 0.25 <= D <= 1.3, 2 <= N <= 15.
    N is continuous, as in most papers.

    Best known f = 0.012665233 at (d, D, N) = (0.051689, 0.356718, 11.288966). The rounded
    point violates the first constraint by about 4e-6, the stored ``x_opt`` is feasible.

    Reference: J. S. Arora, Introduction to Optimum Design, McGraw-Hill, 1989. A. D. Belegundu,
    "A study of mathematical programming methods for structural optimization", PhD thesis,
    University of Iowa, 1982.
    """
    x_opt = np.array([0.051689060909504585, 0.3567177356312907, 11.288965995978813])
    return Problem(Real("x", lb=[0.05, 0.25, 2.0], ub=[2.0, 1.3, 15.0]), spring_obj, vectorized=True,
                   name="spring", optimum=0.012665232788319181,
                   info={"x_opt": x_opt, "source": "Belegundu (1982), Arora (1989)",
                         "variant": "continuous N, four constraints"})


# speed reducer

def speed_reducer_obj(sol):
    x1, x2, x3, x4, x5, x6, x7 = _cols(np.asarray(sol.x, dtype=float))
    with np.errstate(all="ignore"):
        f = (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
             - 1.508 * x1 * (x6**2 + x7**2) + 7.4777 * (x6**3 + x7**3) + 0.7854 * (x4 * x6**2 + x5 * x7**2))
        return _out(f, [27 / (x1 * x2**2 * x3) - 1,
                        397.5 / (x1 * x2**2 * x3**2) - 1,
                        1.93 * x4**3 / (x2 * x3 * x6**4) - 1,
                        1.93 * x5**3 / (x2 * x3 * x7**4) - 1,
                        np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110 * x6**3) - 1,
                        np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85 * x7**3) - 1,
                        x2 * x3 / 40 - 1,
                        5 * x2 / x1 - 1,
                        x1 / (12 * x2) - 1,
                        (1.5 * x6 + 1.9) / x4 - 1,
                        (1.1 * x7 + 1.9) / x5 - 1])


def speed_reducer():
    """Speed reducer design (Golinski 1973), with the eleven constraints of Mezura-Montes and Coello (2005).

    Minimize the weight of a gear box. The variables are the face width x1, the tooth module x2,
    the number of pinion teeth x3 (continuous, as in most papers), the shaft lengths x4 and x5
    between bearings, and the shaft diameters x6 and x7. The constraints bound the bending and
    surface stress of the teeth, the deflection and stress of the shafts, and the dimensions.
    Bounds: 2.6 <= x1 <= 3.6, 0.7 <= x2 <= 0.8, 17 <= x3 <= 28, 7.3 <= x4, x5 <= 8.3,
    2.9 <= x6 <= 3.9, 5.0 <= x7 <= 5.5.

    Best known f = 2994.471066 at (3.5, 0.7, 17, 7.3, 7.715320, 3.350215, 5.286654). Some papers
    use 7.8 as the lower bound of x5, which makes this point infeasible.

    References: J. Golinski, "An adaptive optimization system applied to machine synthesis",
    Mechanism and Machine Theory 8 (1973) 419-436. E. Mezura-Montes and C. A. C. Coello,
    "Useful infeasible solutions in engineering optimization with evolutionary algorithms",
    MICAI 2005, LNAI 3789, 652-662.
    """
    x_opt = np.array([3.5, 0.7, 17.0, 7.3, 7.715319911478245, 3.3502146660964467, 5.286654464980222])
    return Problem(Real("x", lb=[2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], ub=[3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
                   speed_reducer_obj, vectorized=True, name="speed_reducer", optimum=2994.47106614682,
                   info={"x_opt": x_opt, "source": "Golinski (1973), Mezura-Montes and Coello (2005)",
                         "variant": "continuous x3, x5 >= 7.3, eleven constraints"})


# three bar truss

def three_bar_truss_obj(sol):
    x1, x2 = _cols(np.asarray(sol.x, dtype=float))
    P, sigma, length = 2.0, 2.0, 100.0
    with np.errstate(all="ignore"):
        den = _SQRT2 * x1**2 + 2 * x1 * x2
        f = (2 * _SQRT2 * x1 + x2) * length
        return _out(f, [(_SQRT2 * x1 + x2) / den * P - sigma,
                        x2 / den * P - sigma,
                        1 / (x1 + _SQRT2 * x2) * P - sigma])


def three_bar_truss():
    """Three bar truss design (Nowacki 1973, Ray and Saini 2001).

    Minimize the volume f = (2 sqrt(2) x1 + x2) l of a symmetric truss with bar cross sections
    x1 (outer bars) and x2 (middle bar), subject to three stress constraints, with l = 100 cm,
    P = 2 kN/cm^2 and sigma = 2 kN/cm^2. Bounds: 0 <= x1, x2 <= 1.

    Best known f = 263.8958434 at (0.7886751, 0.4082483).

    Reference: T. Ray and P. Saini, "Engineering design optimization using a swarm with an
    intelligent information sharing among individuals", Engineering Optimization 33 (2001)
    735-748.
    """
    x_opt = np.array([0.7886751347347404, 0.40824829006808844])
    return Problem(Real("x", 2, lb=0.0, ub=1.0), three_bar_truss_obj, vectorized=True, name="three_bar_truss",
                   optimum=263.89584337646846,
                   info={"x_opt": x_opt, "source": "Nowacki (1973), Ray and Saini (2001)",
                         "variant": "three stress constraints"})


# gear train

def gear_train_obj(sol):
    t = np.asarray(sol.teeth, dtype=float)
    return (1 / 6.931 - t[..., 0] * t[..., 1] / (t[..., 2] * t[..., 3])) ** 2


def gear_train():
    """Gear train design (Sandgren 1990).

    Choose the numbers of teeth (Ta, Tb, Td, Tf) of four gears, integers from 12 to 60, so that
    the gear ratio Ta Tb / (Td Tf) is as close as possible to 1/6.931:
    f = (1/6.931 - Ta Tb / (Td Tf))^2. Only the bounds constrain the problem.

    Best known f = 2.700857e-12 at (16, 19, 43, 49), which is the global optimum (checked by
    enumeration). Swapping Ta with Tb, or Td with Tf, gives the same value.

    Reference: E. Sandgren, "Nonlinear integer and discrete programming in mechanical design
    optimization", Journal of Mechanical Design 112 (1990) 223-229.
    """
    return Problem(Integer("teeth", 4, lb=12, ub=60), gear_train_obj, vectorized=True, name="gear_train",
                   optimum=2.7008571488865134e-12,
                   info={"x_opt": np.array([16, 19, 43, 49]), "source": "Sandgren (1990)",
                         "variant": "integer teeth from 12 to 60, unconstrained"})
