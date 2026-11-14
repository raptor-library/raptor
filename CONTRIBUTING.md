RAPtor is distributed under LICENSE.txt:

Copyright (c) 2015-2026, RAPtor Developers
License: BSD 2-clause

Source file headers
---

RAPtor source files (`.cpp`, `.hpp`, `.tpp`) begin with this header
(also in `docs/copyright.cpp`):

```cpp
// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
```

Do not add years to file headers; the copyright years are maintained in
LICENSE.txt.  Code adapted from third-party sources also include an attribution
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
   target_link_libraries(test_par_example raptor ${MPI_LIBRARIES} googletest pthread)
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

Contributing code to this project assumes that
    - your contributions are not significant enough to claim copyright or
      that you implicitly transfer the copyright of your contribution to
      the RAPtor Developers
    - you agree to the Developer Certificate of Origin for
      your contributing portion as described here (and below):
      http://developercertificate.org/

Certificate of Origin
---

"""
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
"""
