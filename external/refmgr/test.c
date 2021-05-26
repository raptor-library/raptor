#include <stdio.h>

#include "sample_api.h"

int main(int argc, char *argv[])
{
	int status, size;
	raptor_vec vec;

	status = raptor_vec_create(7, &vec);
	status = raptor_vec_getsize(vec, &size);
	printf("size: %d\n", size);
	status = raptor_vec_free(&vec);

	printf("%d vs %d\n", vec, raptor_vec_NULL);

	return 0;
}
