# Test Data

The files in this directory were created with https://github.com/pyamg/pyamg.

There are several tests:

| `aniso`       | anisotropic diffusion |
| `laplacian27` | 27-point Laplacian |
| `random`      | random sparse matrix |
| `rss`         | Classical AMG for rotated anisotropic diffusion |

With several setups:

| `_inc_b`      | Right-hand side `b = Ax` for `x` ranging from 1 to n |
| `_inc_b_T`    | Right-hand side `b = A^T x` for `x` ranging from 1 to n |
| `_ones_b`     | Right-hand side `b = A x` for `x` equal to 1 |
| `_ones_b_T`   | Right-hand side `b = A^T x` for `x` equal to 1 |

And several portions of an AMG hierarchy:

| `_A1`         | refers to the matrix on level 1 |
| `_P1`         | refers to interpolation on level 1 |
| `_S1`         | refers to strength on level 1 |
| `_cf1`        | refers to strength on level 1 |

And a weight file for matching CLJP with pyamg:
| `weights.txt` | Weights corresponding to random value generator used in Pyamg |

And several python files:
| `mod_strength.py`     | Modified pyamg classical strength (matches hypre) |
| `mod_class_interp.py` | Modified classical interpolation (matched hypre) |



