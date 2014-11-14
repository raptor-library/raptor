import os
import scipy.io as io
import numpy as np

class thinmultilevel:

    class thinlev:
        def __init__(self):
            pass

    def __init__(self,ml):
        self.levels = []
        for l in ml.levels:
            thinlevel = self.thinlev()
            thinlevel.A = l.A
            if hasattr(l,'P'):
                thinlevel.P = l.P
                thinlevel.R = l.R
            self.levels.append(thinlevel) 

#from scipy.io import savemat
#savemat('pyamg_basic.mat', {'b': b, 'ml':thinml})
#savemat('pyamg_basic.mat', {'b':b, 'ml':thinml})

def savehierarchy(name, ml):
    """
    Saves a thin ml hierarchy into directory name as
    A0, P0, R0,
    A1, P1, R1,
    ...
    Ak
    info.txt

    The format of info.txt is just a single integer: number of levels
    """
    thinml = thinmultilevel(ml)
    try:
        os.makedirs(name)
    except OSError:
        raise
    os.chdir(name)
    nlevels = len(thinml.levels)
    np.savetxt('info.txt',[nlevels],fmt='%d')

    for i in range(0,nlevels):
        io.mmwrite('A%d.mtx'%i,thinml.levels[i].A,comment='level %d matrix A'%i, field='real')
        if i < (nlevels - 1):
            io.mmwrite('P%d.mtx'%i,thinml.levels[i].P,comment='level %d matrix P'%i, field='real')
            io.mmwrite('R%d.mtx'%i,thinml.levels[i].R,comment='level %d matrix R'%i, field='real')
