# Documentation

Documentation is found at https://raptor-library.github.io/raptor/.
To build it locally (with Doxygen and Graphviz):

```bash
cmake -Bbuild
cmake --build build --target docs
```

The HTML output is written to `build/docs/doxygen/html`.

## Contents of `docs/`

| Path | Description |
|---|---|
| `doxygen/Doxyfile.in` | Doxygen configuration (filled in by CMake) |
| `doxygen/raptor_main.dox` | Main page of the API documentation |
| `doxygen/doxygen-awesome/` | [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme |
| `bsr_guide.tex` | LaTeX guide to the block sparse row (BSR) format and the `BSRMatrix` / `ParBSRMatrix` classes (attributes, constructors, methods, tests, usage examples) |
| `cpp_docstring_example.cpp` | Example documentation comment layout for functions |
| `copyright.cpp` | Standard SPDX header for new source files |
| `checkstyle.sh` | Runs `cppcheck` static analysis on `raptor/` (run from the repository root) |
| `logo/` | RAPtor logos and the notebook used to generate them |
