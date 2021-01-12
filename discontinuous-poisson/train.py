from fenics import *
from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from make_data import poisson
from helpers import plot
seed(21)

# Load observations

def train(u_stab, element):
    mesh = u_stab.function_space().mesh()
    n = FacetNormal(mesh)

    W = FunctionSpace(mesh, element)
    # Now solve the Poisson with an inconsistent element, 
    # but with the NN as a source term
    
    # Define a neural network that will be added as a source term to the Poisson eqn
    # We pass into the network the mesh size :_
    R = VectorFunctionSpace(mesh, "R", 0, dim=50)
    R2 = FunctionSpace(mesh, "R", 0)
    W_1 = [Function(R)]*5
    W_2 = Function(R)
    b_1 = Function(R)
    b_2 = Function(R2)
    

    eps = 1e1
    for w in W_1:
        w.vector()[:] = eps*rand(R.dim())
    W_2.vector()[:] = eps*rand(R.dim())
    b_1.vector()[:] = eps*rand(R.dim())
    b_2.vector()[:] = eps*rand(R2.dim())

    def nn(u, v):

        inp = as_vector([avg(u), jump(u), *grad(avg(u)), *grad(jump(u)), *n('+')])
        
        def sigma_(vec, func=ufl.tanh):
            v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
            return ufl.as_vector(v)
        relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
        sigma = lambda vec: sigma_(vec, func=relu)

        nn = dot(W_2, sigma(ufl.transpose(as_vector(W_1)) * inp + b_1)) + b_2

        return inner(nn, jump(v) + avg(v))*dS, inner(nn, nn)*dS

    # Now solve the Stokes-NN forward problem
    u, reg = poisson(W, nn)
    plot(u, "out/u_nn0.png")

    J = assemble((u - u_stab)**2*dx)         
    print(f"J={J}")

    # L2 regularisation
    #J += 1e4*reg
    #print(f"reg={1e4*reg}")

    # l2 regularisation
    reg = 0
    for W in [*W_1, W_2, b_1, b_2]:
        reg += 1e4*assemble(W**2*dx)
    J += reg
    print(f"reg={reg}")

    ctrls = [Control(W) for W in W_1]
    ctrls += [Control(b_1), Control(W_2), Control(b_2)]

    Jhat = ReducedFunctional(J, ctrls)
    C_u = Control(u)

    set_log_level(LogLevel.ERROR)
    
    minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 20})

    print("|U - d| = ", assemble(inner(C_u.tape_value() - u_stab, C_u.tape_value() - u_stab)*dx)**0.5)

    u_nn = C_u.tape_value()
    plot(u_nn, "out/u_nn.png")

    return nn

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    
    mesh = UnitIntervalMesh(16)
    
    stable = FiniteElement('Lagrange', interval, 1)
    W = FunctionSpace(mesh, stable)

    u_stab = Function(W)
    with HDF5File(MPI.comm_world, "out/u_stab.h5", "r") as xdmf:
        xdmf.read(u_stab, "u")

    ustab_elm = FiniteElement('DG', interval, 1)
    
    nn = train(u_stab, ustab_elm)    
