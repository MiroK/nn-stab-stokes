from dolfin import *
from dolfin_adjoint import *
from helpers import plot
from numpy.random import rand, seed
seed(21)

# For problem
#   -Delta u = f
#
# A consistent FEM discretization requires piecewise continuous elements. 
# This does not hold for DG1 elements. A suitable stabilization term is added 
# over the element interfaces.
#
# Let NN1: R x R^2 -> R
# 
# Consider min(uS - ud)**2 + alpha*min(pS - pd)**2
#
# subject to -Delta u + grad(p) = 0
#                  div(u) + NN1 = 0
#
# 
# Where uS, pS are STABLE data but our FEM discretization of stokes
# is in terms of unstable elements. Can NN1 learn the stabilization?


def poisson(W, nn=None):
    '''
    u = 0 ---------------------------- u = 1
    '''
    # NOTE: nice thing about this setup is that P1 element get the exact solution
    # of the problem
    bcs = [DirichletBC(W, Constant(0), 'near(x[0], 0)'),
           DirichletBC(W, Constant(1), 'near(x[0], 1)')]
    
    u = Function(W)
    v = TestFunction(W)

    F = inner(grad(u), grad(v))*dx

    if nn:
        Fnn, reg, _ = nn(u, p, v, q)
        F += Fnn
    
    solve(F == 0, u, bcs)
    
    if nn:
        return u, assemble(reg)
    else:
        return u



if __name__== "__main__":
    mesh = UnitIntervalMesh(16)

    stable = FiniteElement('Lagrange', interval, 1)
    W = FunctionSpace(mesh, stable)

    u = poisson(W)

    # Add noise
    eps_noise = 0
    u.vector()[:] += eps_noise*rand(W.dim())

    plot(u, "out/u_stab.png")

    with HDF5File(MPI.comm_world, "out/u_stab.h5", "w") as xdmf:
        xdmf.write(u, "u")
