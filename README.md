# Square‑Free Factorization in ℤ[x]

This script computes the square‑free factorization of a polynomial with integer coefficients:

\[
f(x) = h_1(x)\, h_2(x)^2 \cdots h_k(x)^k,
\]

where each \(h_i(x)\) is square‑free.

Polynomial arithmetic (addition, subtraction, multiplication, division, gcd) is implemented manually via python lists in ℤ[x].  
Division and GCD are computed in **ℚ[x]** using exact rational arithmetic (`fractions.Fraction`), then normalized back to **ℤ[x]**.


## Usage

### Run from terminal

```bash
python3 factorize.py <coefficients>
```

Coefficients are given **from lowest degree to highest**.

Example for  
\[
f(x) = x^3 - 5x^2 + 2
\]

```bash
python3 factorize.py 2 0 -5 1
```

## Requirements

Uses only Python standard library modules:

- `argparse`
- `fractions`
- `math`
- `functools`
