/******** ParVector.cpp **********/
#include "ParVector.hpp"

ParVector::ParVector(int gblN, int lclN)
{
    this.globalN = N;
    this.localN = n;
    this.local = new VectoXd(n);
}

ParVector::ParVector(ParVector* x)
{
    this.globalN = x->globalN;
    this.localN = x->localN;
    this.local = new VectorXd(x->localN);
}

ParVector::~ParVector()
{
    delete local;
}
double ParVector::norm(double p)
{
    double result = local->lpNorm<p>();
    
    result = pow(tmp, p); // undoing root of p from local operation
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return pow(result, 1./p);
}
