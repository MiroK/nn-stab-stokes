from fenics import *
from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from ad_make_data import advection_diffusion, make_data
from helpers import plot
import dolfin as df
import matplotlib.pyplot as plt
    
seed(21)

# Load observations

def train_ad(u_stab, V, alpha):
    mesh = V.mesh()

    # Define a neural network that will be added as a source term to the Stokes eqn
    R = VectorFunctionSpace(mesh, "R", 0, dim=50)
    W_1, W_2, b_1, W_3_1, W_3_2 = Function(R), Function(R), Function(R), Function(R), Function(R)
    W_3 = as_vector([W_3_1, W_3_2])
    R2 = VectorFunctionSpace(mesh, "R", 0, dim=2)
    b_2 = Function(FunctionSpace(mesh, 'R', 0))
    
    eps = 1e1
    W_1.vector()[:] = eps*rand(R.dim())
    W_2.vector()[:] = eps*rand(R.dim())
    W_3[0].vector()[:] = eps*rand(R.dim())
    W_3[1].vector()[:] = eps*rand(R.dim())
    b_1.vector()[:] = eps*rand(R.dim())
    b_2.vector()[:] = eps*rand(b_2.function_space().dim())

    def nn(u, v):
        #return inner(grad(p), grad(q)) * dx 
        
        def sigma_(vec, func=ufl.tanh):
            v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
            return ufl.as_vector(v)
        relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
        sigma = lambda vec: sigma_(vec, func=relu)#lambda x:x)

        #from IPython import embed
        #embed()
        
        nl = dot(W_2, sigma(W_1 * (u + u.dx(0))+ b_1)) + b_2

        return inner(nl, v.dx(0))*dx, inner(nl, nl)*dx, nl

    # Now solve the Stokes-NN forward problem
    uh, reg = advection_diffusion(V, alpha, nn)
    plot(uh, "out/ad0.png")

    J = assemble((uh - u_stab)**2*dx)         
    print(f"J={J}")

    reg = 0
    for W in [W_1, W_2, b_1, W_3_1, W_3_2, b_2]:
        reg += 1E-4*assemble(W**2*dx)
    J += reg
    print(f"reg={reg}")

    Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3_1), Control(W_3_2)])
    C_u = Control(uh)

    set_log_level(LogLevel.ERROR)
    
    minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 50})

    print("|U - d| = ", assemble(inner(C_u.tape_value() - u_stab, C_u.tape_value() - u_stab)*dx)**0.5)

    u_nn = C_u.tape_value()

    
    df.plot(u_nn, label='learned')
    df.plot(interpolate(u_stab, V), label='data alpha {}'.format(alpha))
    df.plot(advection_diffusion(V, alpha=alpha, nn=None), label='no nn')
    plt.legend()
    plt.savefig("out/ad_nn.png")    

    return nn

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np


    eps = 1E-2
    mesh = UnitIntervalMesh(16)
    elm = FiniteElement('Lagrange', interval, 1)
    V = FunctionSpace(mesh, elm)
    
    u_data, u_true = make_data(eps, mesh, elm)

    df.plot(u_data)
    df.plot(interpolate(u_true, FunctionSpace(UnitIntervalMesh(10000), 'CG', 1)))
    df.plot(advection_diffusion(V, alpha=eps, nn=None))
    plt.show()

    nn = train_ad(u_true, V, alpha=eps)    
