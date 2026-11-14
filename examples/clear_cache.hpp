// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef CLEAR_CACHE
#define CLEAR_CACHE

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include "raptor/core/types.hpp"

using namespace raptor;

void clear_cache(std::vector<double>& cache_list)
{
    srand(time(NULL));
    for (size_t i = 0; i < cache_list.size(); i++)
    {
        cache_list[i] = rand()%10;
    }
}


#endif
