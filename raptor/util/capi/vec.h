#ifndef CEDAR_INTERFACE_VEC_H
#define CEDAR_INTERFACE_VEC_H

#include <cedar/capi.h>
#include <cedar/2d/mpi/grid_func.h>
#ifdef ENABLE_3D
#include <cedar/3d/mpi/grid_func.h>
#endif

struct cedar_vec_cont
{
	unsigned short nd;
	cedar_topo topo;
	std::unique_ptr<cedar::cdr2::mpi::grid_func> gfunc2;
	#ifdef ENABLE_3D
	std::unique_ptr<cedar::cdr3::mpi::grid_func> gfunc3;
	#endif
};


cedar_vec_cont *cedar_vec_getobj(cedar_vec vec);

#endif
