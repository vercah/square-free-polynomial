#!/usr/bin/env python3

import argparse
from fractions import Fraction
from math import gcd
from functools import reduce

def to_fraction_poly(p):
    return [Fraction(c) for c in p]

def strip(poly):
    while len(poly) > 1 and poly[-1] == 0: # oříznutí vedoucích nul
        poly.pop()
    return poly

def pol_add(a, b): # součet dvou polynomů
    print(f"Adding polynomials: {poly_to_string(a)} + {poly_to_string(b)}")
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append(ai + bi)
    return strip(result)

def pol_sub(a, b): # rozdíl dvou polynomů
    print(f"Subtracting polynomials: {poly_to_string(a)} - {poly_to_string(b)}")
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append(ai - bi)
    return strip(result)

def pol_mul(a, b): # součin dvou polynomů
    print(f"Multiplying polynomials: {poly_to_string(a)} * {poly_to_string(b)}")
    result = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return strip(result)

def pol_derivative(a): # derivace polynomu
    print(f"Calculating derivative of polynomial: {poly_to_string(a)}")
    if len(a) <= 1:
        return [0]
    result = []
    for i in range(1, len(a)):
        result.append(a[i] * i)
    return strip(result)

def pol_div(a, b): # dělení polynomů v Q, vrací (podíl, zbytek)
    #print(f"Dividing polynomials: {poly_to_string(a)} / {poly_to_string(b)}")
    a = [Fraction(x) for x in a]
    b = [Fraction(x) for x in b]

    a = strip(a)
    b = strip(b)

    if b == [0]:
        raise ZeroDivisionError("Division by zero polynomial")

    result = [Fraction(0)] * (len(a) - len(b) + 1)

    while len(a) >= len(b):
        #print(f"Current dividend: {poly_to_string(a)}, leading divisor: {b[-1]}, leading dividend: {a[-1]}")
        if a == [Fraction(0)]:
            break
        coef = a[-1] / b[-1] #přesné dělení v Q
        shift = len(a) - len(b)
        result[shift] = coef

        for i in range(len(b)):
            a[shift + i] -= coef * b[i]
        a = strip(a)
        #print(f"New dividend after subtraction: {poly_to_string(a)}")

    return strip(result), strip(a)

def pol_gcd(a, b): #eukleidův algoritmus v Q[x], na konci normalizace do Z[x]
    a = to_fraction_poly(a)
    b = to_fraction_poly(b)

    while b != [Fraction(0)]:
        _, r = pol_div(a, b)
        a, b = b, r
    return normalize_to_Z(a)

def lcm(a, b):
    return a * b // gcd(a, b)

def normalize_to_Z(p):
    print(f"Normalizing polynomial to Z[x]: {poly_to_string(p)}")
    denoms = [c.denominator for c in p]    
    L = reduce(lcm, denoms, 1) #vrati lcm všech jmenovatelů
    #print(f"Least common multiple of denominators: {L}")

    q = [int(c * L) for c in p]
    G = reduce(gcd, [abs(x) for x in q], 0)
    #print(f"GCD of coefficients before normalization: {G}")
    
    if G > 1:
        q = [x // G for x in q]

    if q[-1] < 0:
        q = [-x for x in q]
    print(f"Normalized polynomial: {poly_to_string(strip(q))}")
    return strip(q)

def square_free_factorization(f): #square-free faktorizace
    print(f"Starting square-free factorization for polynomial: {poly_to_string(f)}")
    f = strip(f[:])
    f_der = pol_derivative(f)

    f_j = pol_gcd(f, f_der)
    g_j, _ = pol_div(f, f_j)

    factors = []
    j = 1
    while g_j != [1]:
        g_jj = pol_gcd(f_j, g_j)
        f_jj, _ = pol_div(f_j, g_jj)
        h_j, _ = pol_div(g_j, g_jj)
        if h_j != [1]:
            factors.append((j, h_j))
        f_j = f_jj
        g_j = g_jj
        j += 1

    return factors

def poly_to_string(p): #lepší výpis polynomu
    terms = []
    for i, coef in enumerate(p):
        if coef == 0:
            continue

        if coef > 0 and terms:
            sign = " + "
        elif coef < 0:
            sign = " - "
        else:
            sign = ""
        c = abs(coef)

        if i == 0:
            term = f"{c}"
        elif i == 1:
            term = f"{'' if c == 1 else c}x"
        else:
            term = f"{'' if c == 1 else c}x^{i}"

        if terms:
            terms.append(sign + term)
        else:
            if coef < 0:
                terms.append("-" + term)
            else:
                terms.append(term)
    if not terms:
        return "0"
    return "".join(terms)


def main():
    parser = argparse.ArgumentParser(
        description="Square-free factorization of polynomial in Z[x]."
    )
    parser.add_argument(
        "coefficients",
        metavar="c",
        type=int,
        nargs="+",
        help="Coefficients of the polynomial f(x) from the lowest degree term. For example, for x^3 - 5x^2 + 2 enter: 2 0 -5 1"
    )

    args = parser.parse_args()
    f = args.coefficients
    factors = []
    #factors = square_free_factorization(f)
    printfactors = ""
    if factors:
        for i in range(len(factors)):
            exp, poly = factors[i]
            if exp > 1:
                printfactors += f"({poly_to_string(poly)})^{exp}"
            else:
                printfactors += f"({poly_to_string(poly)})"
            if i < len(factors) - 1:
                printfactors += " * "


    f0 = [1, 1, -1, -1, -1, -1, 1, 1]
    fder = pol_derivative(f0)
    f1 = pol_gcd(f0, fder)
    print(f"GCD of {poly_to_string(f0)} and its derivative {poly_to_string(fder)} is f1 = {poly_to_string(f1)}")
    g1, _ = pol_div(f0, f1)
    print(f"Quotient g1 = {poly_to_string(g1)}")
    g2 = pol_gcd(f1, g1)
    print(f"GCD of f1 and g1 is g2 = {poly_to_string(g2)}")
    f2, _ = pol_div(f1, g2)
    print(f"Quotient f2 = {poly_to_string(f2)}")
    g3 = pol_gcd(g2, f2)
    print(f"GCD of g2 and f2 is g3 = {poly_to_string(g3)}")
    f3, _ = pol_div(f2, g3)
    print(f"Quotient f3 = {poly_to_string(f3)}")
    g4 = pol_gcd(g3, f3)
    print(f"GCD of g3 and f3 is g4 = {poly_to_string(g4)}")
    f4, _ = pol_div(f3, g4)
    print(f"Quotient f4 = {poly_to_string(f4)}")


if __name__ == "__main__":
    main()
