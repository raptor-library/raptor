#include <iostream>

int main(int argc, char* argv[])
{
    srand(2448422);
    for(int i=0; i < 15625; i++){
      printf("%e\n", double(rand())/RAND_MAX);
    }

    return 0;
}
