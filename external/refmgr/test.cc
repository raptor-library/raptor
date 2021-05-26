#include <iostream>

#include "sample_api.h"

int main(int argc, char *argv[])
{
	int status, size;
	raptor_vec vec;

	status = raptor_vec_create(7, &vec);
	status = raptor_vec_getsize(vec, &size);
    std::cout << "size: " << size << std::endl;
	status = raptor_vec_free(&vec);

    std::cout << vec << " " << raptor_vec_NULL << std::endl;

    return 0;
}
