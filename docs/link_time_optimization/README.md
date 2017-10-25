The information in this directory summarize our experience compiling raptor 
with link time optimization (LTO) using the gnu compilers (-flto).

Link time optimization has been observed to produce significant speed-up
in raptor.


We found that the procedure to effectively use LTO varies from system to system. 
Factors as compiler version, linker version, mpi implementation seems to have an 
effect in the results, therefore we include information on three system:
    1.- Blue Waters
    2.- Campus Cluster (university of Illinois)
    3.- Our own MachineShop (https://andreask.cs.illinois.edu/MachineShop/UserNotes)
    
Please look at the README.md file of the corresponding directory.

Important: One think that seems to have an effect in all cases is the 
linker (ld) version; LTO seems to work only when the version of the
linker used 2.21 and older. So check your linker version (ld --version).

Update: 10/25/2017 

A new directory (Automatic_LTO) was added discussing automatic link time optimization.


