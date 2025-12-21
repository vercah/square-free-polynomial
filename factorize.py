#!/usr/bin/env python3

import argparse

def strip(poly):
    while len(poly) > 1 and poly[-1] == 0: # oříznutí vedoucích nul
        poly.pop()
    return poly

def pol_add(a, b): # součet dvou polynomů
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append(ai + bi)
    return strip(result)

def pol_sub(a, b): # rozdíl dvou polynomů
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append(ai - bi)
    return strip(result)

def pol_mul(a, b): # součin dvou polynomů
    result = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return strip(result)

def pol_derivative(a): # derivace polynomu
    if len(a) <= 1:
        return [0]
    result = []
    for i in range(1, len(a)):
        result.append(a[i] * i)
    return strip(result)

def pol_div(a, b): # dělení polynomů, vrací (podíl, zbytek)
    a = a[:]  # kopie
    b = strip(b[:])
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError("Division by zero polynomial")

    result = [0] * (len(a) - len(b) + 1)
    div_lead = b[-1]

    while len(a) >= len(b):
        coef = a[-1] // div_lead
        shift = len(a) - len(b)
        result[shift] = coef

        for i in range(len(b)):
            a[shift + i] -= coef * b[i]
        strip(a)

    return strip(result), strip(a)

def pol_gcd(a, b): # gcd dvou polynomů, euklidův algoritmus
    a = strip(a[:])
    b = strip(b[:])
    while b != [0]:
        _, r = pol_div(a, b)
        a, b = b, r
    # normalizace - vedoucí koeficient musí být kladný
    if a[-1] < 0:
        a = [-x for x in a]
    return a

def square_free_factorization(f): #square-free faktorizace
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

    factors = square_free_factorization(f)
    printfactors = ""
    for i in range(len(factors)):
        exp, poly = factors[i]
        if exp > 1:
            printfactors += f"({poly_to_string(poly)})^{exp}"
        else:
            printfactors += f"({poly_to_string(poly)})"
        if i < len(factors) - 1:
            printfactors += " * "

    print("\nSquare-free factors:")
    print(poly_to_string(f), "=", printfactors)

if __name__ == "__main__":
    main()
