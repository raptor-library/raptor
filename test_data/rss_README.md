Files `rss_*.pm`: Matrices created in Ruge Stuben AMG 
    - classical strength measure (modified to match Hypre)
            -threshold = 0.25
    - CLJP coarsening
    - Direct interpolation (as in pyamg)
    - Fine matrix: rotated anisotropic diffusion
            - epsilon = 0.001
            - theta = pi / 8
            - grid = {50, 50}
    - Saved Ruge Stuben CF splittings
    - Saved Modified Classical Interpolation operators

Corresponding Python Files
| `mod_strength.py`       | Python file for modified strength of connection |
| `mod_class_interp.py'   | Python file for modified classical interpolation |
