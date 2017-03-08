// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_LEVEL_H
#define RAPTOR_CORE_LEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/prolongation.hpp"

// Coarse Matrices (A) are CSC
// Prolongation Matrices (P) are CSC
// P^T*A*P is then CSR*(CSC*CSC) -- returns CSC Ac
namespace raptor
{
    class Level
    {
        public:

            // Create level using fine A
            Level(CSRMatrix& Af, double theta = 0.0, double omega = 4.0/3, 
                    int num_smooth_steps = 1, int max_coarse = 50)
            {
                // Copy fine level matrix to level
                A = CSRMatrix(Af);
                A.sort();

                // If n_rows > max_coarse, create P
                if (A.n_rows > max_coarse)
                {
                    // Create strength of connection matrix
                    CSRMatrix S;
                    A.symmetric_strength(&S, theta);

                    // Create tentative interpolation
                    CSCMatrix T;
                    std::vector<int> c_points;
                    standard_aggregation(S, T, c_points);

                    // Smooth T to form prolongation (P)
                    jacobi_prolongation(A, T, P, omega, num_smooth_steps);
                }

                // Initalize space for x, b, tmp
                x = Vector(A.n_rows);
                b = Vector(A.n_rows);
                tmp = Vector(A.n_rows);
            }


            // Create level using finer A, P
            Level(CSRMatrix& Af, CSCMatrix& Pf, double theta = 0.0, 
                    double omega = 4.0/3, int num_smooth_steps = 1, 
                    int max_coarse = 50)
            {
                // Create coarse matrix (A = Pf^(T)*Af*Pf)
                //CSRMatrix Atmp;
                //Af.mult(Pf, &Atmp);
                //Pf.mult_T(Atmp, &A);
                Af.RAP(Pf, &A);
                A.sort();

                // If n_rows > max_coarse, create P
                if (A.n_rows > max_coarse)
                {
                    // Create strength of connection matrix
                    CSRMatrix S;
                    A.symmetric_strength(&S, theta);

                    // Create tentative interpolation
                    CSCMatrix T;
                    std::vector<int> c_points;
                    standard_aggregation(S, T, c_points);

                    // Smooth T to form prolongation (P)
                    jacobi_prolongation(A, T, P, omega, num_smooth_steps);
                }
                                
                // Initalize space for x, b, tmp
                x = Vector(A.n_rows);
                b = Vector(A.n_rows);
                tmp = Vector(A.n_rows);
            }

            CSRMatrix A;
            CSCMatrix P;
            Vector x;
            Vector b;
            Vector tmp;
    };
}
#endif
