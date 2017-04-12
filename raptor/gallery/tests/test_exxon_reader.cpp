#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/exxon_reader.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    // Create A from diffusion stencil
    char* folder = "";
    char* iname = "";
    char* fname = "";
    char* suffix = "";

    int* global_num_rows;

    ParCSRMatrix* A = exxon_reader(folder, iname, fname, suffix, &global_num_rows);
    

    delete A;

}   

