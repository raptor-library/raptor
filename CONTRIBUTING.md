RAPtor is distributed under LICENSE.txt:

Copyright (c) 2015-2026, RAPtor Core Developers
License: BSD 2-clause

RAPtor source files (`.cpp`, `.hpp`, `.tpp`) begin with this header
(also in `docs/copyright.cpp`):

```cpp
// SPDX-FileCopyrightText: RAPtor Core Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
```

The copyright years are maintained in LICENSE.txt.
Code adapted from third-party sources also include an attribution
line after the header citing the original source, copyright, and license.

Development workflow
---

Build and run the tests using:

   ```bash
   cmake -Bbuild
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```

Register new tests in the associated
`CMakeLists.txt`. For example:

   ```cmake
   add_executable(test_par_example test_par_example.cpp)
   target_link_libraries(test_par_example raptor ${MPI_LIBRARIES} googletest)
   add_test(ParExampleTest ${MPIRUN} -n 4 ${HOST} ./test_par_example)
   ```

   (note: for a non-parallel test, omit `${MPIRUN} -n <N> ${HOST}`).

Check spelling with [typos](https://github.com/crate-ci/typos)
   (project exceptions are in `_typos.toml`):

   ```bash
   typos
   ```

Contributing
---

By submitting a pull request or other contribution to this project, you agree that:

  1. You have reviewed and understand the contribution in its entirety;

  2. To the best of your knowledge, you are entitled to contribute it under this project’s license (see LICENSE.txt); and

  3. You transfer the copyright of your contribution to the RAPtor Core Developers, where possible.
