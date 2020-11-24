from fenics import *
from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from make_data import stokes
seed(21)


# Generate data with a stable element pair
mesh = UnitSquareMesh(10, 10, 'crossed')

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


# Define a neural network that will be added as a source term to the Stokes eqn
R = VectorFunctionSpace(mesh, "R", 0, dim=50)
W_1, W_2, b_1, W_3_1, W_3_2 = Function(R), Function(R), Function(R), Function(R), Function(R)
W_3 = as_vector([W_3_1, W_3_2])
R2 = VectorFunctionSpace(mesh, "R", 0, dim=2)
b_2 = Function(R2)

W_1.vector()[:] = 2*rand(R.dim())
W_2.vector()[:] = 2*rand(R.dim())
W_3[0].vector()[:] = 2*rand(R.dim())
W_3[1].vector()[:] = 2*rand(R.dim())
b_1.vector()[:] = 2*rand(R.dim())
b_2.vector()[:] = 2*rand(R2.dim())

def rhs(u, p, v, q):
    #return inner(grad(p), grad(q)) * dx 

    def sigma_(vec, func=ufl.tanh):
        v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
        return ufl.as_vector(v)
    relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
    sigma = lambda vec: sigma_(vec, func=relu)
    return inner(dot(W_3, sigma(ufl.transpose(as_vector([W_1, W_2])) * grad(p) + b_1)) + b_2, grad(q)) * dx 
   (NN(p), q) --  (p - Pi*p, q - Pi*q)

   # 
   # h**2*       (NN(grad(p)), q)
   # [A_h, B_h';     [-Delta, 0
   #   B_h, NN]            0, I]
   #

   # Delta(Delta(u)) = f
   # P2
   # inner(div(grad(u)), div(grad(v)))*dx + (NN(u, grad(u), n), q)*dS
   # 

   #
   # -div(mu*grad(u) + lambda*div(u)*I) + NN(u, div(u)) = 0

   # -div([eps+O(h)]*grad(u)) - v*grad(u) + NN(u)

# Now solve the Stokes-NN forward problem
up = stokes(W, rhs)

J = assemble((up - up_stab)**2*dx)

Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3_1), Control(W_3_2)])
C_up = Control(up)

set_log_level(LogLevel.ERROR)

minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 5})

print("|U - d| = ", assemble(inner(C_up.tape_value() - up_stab, C_up.tape_value() - up_stab)*dx)**0.5)

u_nn, p_nn = C_up.tape_value().split(deepcopy=True)
File("u_nn.pvd") << u_nn
File("p_nn.pvd") << p_nn
