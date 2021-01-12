from fenics import *
from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from mp_make_data import mixed_poisson
seed(21)

# Load observations

def train_mixed_poisson(data, W):
    '''Fit data with state in W(ideally unstable)'''
    
    # Define a neural network that will be added as a source term to the Stokes eqn
    R = VectorFunctionSpace(mesh, "R", 0, dim=50)
    W_1, W_2, b_1, W_3_1, W_3_2 = Function(R), Function(R), Function(R), Function(R), Function(R)
    W_3 = as_vector([W_3_1, W_3_2])
    R2 = VectorFunctionSpace(mesh, "R", 0, dim=2)
    b_2 = Function(R2)
    
    eps = 1e1
    W_1.vector()[:] = eps*rand(R.dim())
    W_2.vector()[:] = eps*rand(R.dim())
    W_3[0].vector()[:] = eps*rand(R.dim())
    W_3[1].vector()[:] = eps*rand(R.dim())
    b_1.vector()[:] = eps*rand(R.dim())
    b_2.vector()[:] = eps*rand(R2.dim())

    def nn(u, p, v, q):
        # return inner(grad(p), grad(q)) * dx, None, None
        
        def sigma_(vec, func=ufl.tanh):
            v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
            return ufl.as_vector(v)
        relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
        sigma = lambda vec: sigma_(vec, func=relu)#lambda x:x)

        nn_p = dot(W_3, sigma(ufl.transpose(as_vector([W_1, W_2])) * u + b_1)) + b_2

        #nn_q = dot(W_3, sigma(ufl.transpose(as_vector([W_1, W_2])) * grad(q) + b_1)) + b_2

        return inner(nn_p, v)*dx, inner(nn_p, nn_p)*dx, nn_p

    sigma0, u0 = data.split(deepcopty=True)
    # Now solve the Stokes-NN forward problem
    w, reg = mixed_poisson(W, u0, nn)
    sigma_nn, u_nn = w.split(deepcopy=True)
    plot(sigma_nn, "out/mp_sigma_nn0.png")
    plot(u_nn, "out/mp_u_nn0.png")

    J = assemble((data - w)**2*dx)
    print(f"J={J}")

    reg = 0
    for W in [W_3, W_1, W_2, b_1, b_2]:
        reg += 1e4*assemble(W**2*dx)
    J += reg
    print(f"reg={reg}")

    Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3_1), Control(W_3_2)])
    C_w = Control(w)

    set_log_level(LogLevel.ERROR)
    
    minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 20})

    print("|U - d| = ", assemble(inner(C_w.tape_value() - data, C_w.tape_value() - data)*dx)**0.5)

    sigma_nn, u_nn = C_w.tape_value().split(deepcopy=True)

    
    File("out/mp_sigma_nn.pvd") << sigma_nn
    File("out/mp_u_nn.pvd") << u_nn

    File("out/mp_sigma0.pvd") << sigma0
    File("out/mp_u0.pvd") << u0

    return nn

# --------------------------------------------------------------------

if __name__ == '__main__':
    from mp_make_data import make_data    
    import numpy as np
    
    mesh = RectangleMesh(Point(0.1, 0.1), Point(1.1, 1.1), 16, 16)
    
    elm = [VectorElement('Lagrange', triangle, 1),
           FiniteElement('Discontinuous Lagrange', triangle, 0)]
    W = FunctionSpace(mesh, MixedElement(elm))

    sigma0, u0 = make_data()
    elm = [FiniteElement('Raviart-Thomas', triangle, 1),
           FiniteElement('Discontinuous Lagrange', triangle, 0)]
    W = FunctionSpace(mesh, MixedElement(elm))
    data = Function(W)
    assign(data, [interpolate(sigma0, W.sub(0).collapse()),
                  interpolate(u0, W.sub(1).collapse())])

    nn = train_mixed_poisson(data, W)    
