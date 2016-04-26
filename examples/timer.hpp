#ifndef INIT_TIMER_HPP
#define INIT_TIMER_HPP

#include <time.h>
#include <sys/time.h>
#include <mpi.h>


/* Currently set to use MPI_Wtime()... uncomment for more precision on BW*/
//#define CLOCKTYPE CLOCK_MONOTONIC    
struct timespec now_ts;


#define CT
#ifdef CT

#ifdef CLOCKTYPE
    #define get_ctime(tval) \
    clock_gettime(CLOCKTYPE, &now_ts); \
    tval = (double)(now_ts.tv_sec) + (double) (now_ts.tv_nsec/1000000000.0);
#else 
    #define get_ctime(tval) \
    tval = MPI_Wtime();
#endif

#endif


#endif
