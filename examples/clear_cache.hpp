#ifndef CLEAR_CACHE
#define CLEAR_CACHE

#include <stdlib.h>
#include <time.h>
#include <math.h>

void clear_cache(int size, double* cache_list)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        cache_list[i] = rand()%10;
    }
}


#endif
