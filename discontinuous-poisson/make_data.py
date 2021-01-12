from dolfin import *
from dolfin_adjoint import *
from helpers import plot
from numpy.random import rand, seed
seed(21)

# For problem
#   -Delta u = f
#
# A consistent FEM discretization requires piecewise continuous elements. 
# This does not hold for DG1 elements. Instead, a suitable stabilization term 
# must be added over the element interfaces, e.g. as in the discontinuous Galerkin 
# formulation (interior penalty method).
# Such a formulation is for instance:
#
# dot(grad(v), grad(u))*dx \
#   - dot(avg(grad(v)), jump(u, n))*dS \
#   - dot(jump(v, n), avg(grad(u)))*dS \
#   + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS 
#
# In the Neural Network formulation we use:
#
# dot(grad(v), grad(u))*dx \
#   + dot(NN(avg(u), jump(u), n), jump(u, n) + avg(u, n))*dS 
#
# For more info, see doi:10.1007/978-3-642-22980-0 
#
# Let NN1: R x R^2 -> R
# 
# Consider min(uS - ud)**2 + alpha*min(pS - pd)**2
#
# subject to -Delta u + NN1 = 0


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
        Fnn, reg = nn(u, v)
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
