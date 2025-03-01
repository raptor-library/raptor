#ifndef RAPTOR_UTIL_WRITER_HPP
#define RAPTOR_UTIL_WRITER_HPP

#include "raptor/core/par_matrix.hpp"

namespace raptor
{

void write(const char *fname, const ParCSRMatrix &mat);
void write(const char *fname, const ParBSRMatrix &mat);
void write(const char *fname, const BSRMatrix &mat);
void write(const char *fname, const CSRMatrix &mat);

}
#endif
