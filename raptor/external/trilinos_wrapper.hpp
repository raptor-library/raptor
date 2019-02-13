// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_TRILINOS_WRAPPER_H
#define RAPTOR_TRILINOS_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/comm_pkg.hpp"

#include <MueLu_ConfigDefs.hpp>
#include <Epetra_CrsMatrix.h>
#include <ml_MultiLevelPreconditioner.h>
#include <Xpetra_EpetraCrsMatrix.hpp>
#include <AztecOO.h>
#include <MueLu_EpetraOperator.hpp>
#include <MueLu_CreateEpetraPreconditioner.hpp>

#include <MueLu.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>

#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>


typedef double Scalar;
typedef int LocalOrdinal;
typedef int GlobalOrdinal;
typedef Xpetra::EpetraNode Node;
#include <MueLu_UseShortNames.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

using namespace raptor;

Epetra_Vector* epetra_convert(raptor::ParVector& x_rap,
                       RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
Epetra_CrsMatrix* epetra_convert(raptor::ParCSRMatrix* A_rap,
                       RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);
AztecOO* create_ml_hierarchy(Epetra_CrsMatrix* Ae, Epetra_Vector* xe, Epetra_Vector* be, int dim = 2, int* coords = NULL);


#endif

