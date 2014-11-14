import numpy as np
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d

n=500
X,Y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
stencil = diffusion_stencil_2d(type='FE',epsilon=0.001,theta=np.pi/3)
A = stencil_grid(stencil, (n,n), format='csr')

from pyamg.gallery.laplacian import poisson
A = poisson( (n,n) , format ='csr')
b = np.random.rand(A.shape[0])
ml = smoothed_aggregation_solver(A,
        B=X.reshape(n*n,1),
        strength='symmetric',
        aggregate='standard',
        smooth=('jacobi', {'omega': 4.0/3.0,'degree':2}),
        presmoother=('jacobi', {'omega': 4.0/3.0}), 
        postsmoother=('jacobi', {'omega': 4.0/3.0}), 
        Bimprove=None,
        max_levels=10,
        max_coarse=5,
        keep=False)

res = []
x = ml.solve(b, tol=1e-8, residuals=res)
import pyamg_util
#pyamg_util.savehierarchy('pyamg_basic_%d_%d'%(n,n),ml)
pyamg_util.savehierarchy('poisson_%d_%d'%(n,n),ml)
print res
