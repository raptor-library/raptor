#ifndef PARVECTOR_HPP
#define PARVECTOR_HPP

#include <mpi.h>
#include <math.h>

#include <Eigen/Dense>
using Eigen::VectorXd;

class ParVector
{
    public:
        ParVector(N, n);
        ParVector(ParVector* x);
        ~ParVector();

        double norm(double p);
        VectorXd* getLocalVector()
            { return local; }
        void axpy(ParVector* x, double alpha) 
            { local += x->local * alpha; }
        void scale(double alpha)
            { local *= alpha; }
        void setConstValue(double alpha)
            { local = VectorXd::Constant(alpha); }
        void setValues(double* values)
            { local << values; }
    private:
        int globalN;
        int localN;
        VectorXd* local;
};
#endif
