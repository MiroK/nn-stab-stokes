from fenics import *
#from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from make_data import stokes
seed(21)


# Generate data with a stable element pair
mesh = UnitSquareMesh(32, 32, 'crossed')

stable = [VectorElement('Lagrange', triangle, 2),
          FiniteElement('Lagrange', triangle, 1)]
W = FunctionSpace(mesh, MixedElement(stable))

up_stab = stokes(W)

# Now solve the Stokes with an unstable element pair, 
# but with the NN as a source term

# This one yield checker-board pattern of reasonable magnitude
unstable = [VectorElement('Lagrange', triangle, 1),
            FiniteElement('Lagrange', triangle, 1)]
W = FunctionSpace(mesh, MixedElement(unstable))


def rhs(u, p, v, q):
    #return inner(grad(p), grad(q)) * dx 

    # Define a neural network that will be added as a source term to the Stokes eqn
    R = VectorFunctionSpace(mesh, "R", 0, dim=50)
    W_1, W_2, b_1, W_3 = Function(R), Function(R), Function(R), Function(R)
    W_4 = as_vector([Function(R), Function(R)])
    R2 = VectorFunctionSpace(mesh, "R", 0, dim=2)
    b_2 = Function(R2)

    W_1.vector()[:] = 2*rand(R.dim())
    W_2.vector()[:] = 2*rand(R.dim())
    W_3.vector()[:] = 2*rand(R.dim())
    W_4[0].vector()[:] = 2*rand(R.dim())
    W_4[1].vector()[:] = 2*rand(R.dim())
    b_1.vector()[:] = 2*rand(R.dim())
    b_2.vector()[:] = 2*rand(R2.dim())

    def sigma_(vec, func=ufl.tanh):
        v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
        return ufl.as_vector(v)
    relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
    sigma = lambda vec: sigma_(vec, func=relu)
    return inner(dot(W_4, sigma(ufl.transpose(as_vector([W_1, W_2])) * grad(p) + b_1)) + b_2, grad(q)) * dx 


# Now solve the Stokes-NN forward problem
up = stokes(W, rhs)

J = assemble((up - up_stab)**2*dx)
print(J)

exit()
Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3), Control(W_4)])
C_up = Control(up)

set_log_level(LogLevel.ERROR)

W_1, b_1, W_2, b_2, W_3, W_4 = minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 20})



print("|U - d| = ", assemble(abs(C_u.tape_value() - d)*dx))
print("|kappa(x)*d - N(U, X)| = ", assemble(abs(cos(X[0]+X[1])*d - (inner(W_4, sigma(ufl.transpose(as_vector([W_1, W_2, W_3])) * as_vector([C_u.tape_value(), *X]) + b_1)) + b_2))*dx))
