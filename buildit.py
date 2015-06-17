import os

ipath = '-I/Users/lukeo/repos/raptor/'
MPICC = 'mpicxx'

# [1/4] Compiling raptor/src/core/ParVector.cpp
cmd1 = ' '.join([MPICC, '-O2', '--std=c++11',
                 ipath+'external',
                 './raptor/core/ParVector.cpp',
                 '-c', '-o', 'ParVector.cpp.1.o'])

# [2/4] Compiling examples/vecnorm.cpp
cmd2 = ' '.join([MPICC, '-O2', '--std=c++11',
                 ipath+'external',
                 ipath,
                 './examples/vecnorm.cpp',
                 '-c', '-o', 'vecnorm.cpp.1.o'])

# [3/4] Linking build/raptor/src/libraptor.a
cmd3 = ' '.join(['/usr/bin/ar', 'rcs',
                 'libraptor.a', 'ParVector.cpp.1.o'])

# [4/4] Linking build/examples/vecnorm
cmd4 = ' '.join([MPICC,
                 'vecnorm.cpp.1.o',
                 '-o', 'vecnorm',
                 '-L.',
                 '-lraptor'])

print(cmd1)
os.system(cmd1)
print(cmd2)
os.system(cmd2)
print(cmd3)
os.system(cmd3)
print(cmd4)
os.system(cmd4)
