from fenics import *
from fenics_adjoint import *
import ufl
from numpy.random import rand, seed
from make_data import stokes
from helpers import plot
seed(21)

# Load observations
mesh = UnitSquareMesh(16, 16, 'crossed')

stable = [VectorElement('Lagrange', triangle, 2),
          FiniteElement('Lagrange', triangle, 1)]
W = FunctionSpace(mesh, MixedElement(stable))

up_stab = Function(W)
with HDF5File(MPI.comm_world, "out/up_stab.h5", "r") as xdmf:
    xdmf.read(up_stab, "up")

# Now solve the Stokes with an unstable element pair, 
# but with the NN as a source term

# This one yield checker-board pattern of reasonable magnitude
unstable = [VectorElement('Lagrange', triangle, 2),
            FiniteElement('Lagrange', triangle, 1)]
W = FunctionSpace(mesh, MixedElement(unstable))


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
    #return inner(grad(p), grad(q)) * dx 

    def sigma_(vec, func=ufl.tanh):
        v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
        return ufl.as_vector(v)
    relu = lambda vec: conditional(ufl.gt(vec, 0), vec, (ufl.exp(vec) - 1))
    sigma = lambda vec: sigma_(vec, func=relu)
    nn = dot(W_3, sigma(ufl.transpose(as_vector([W_1, W_2])) * grad(p) + b_1)) + b_2

    return inner(nn, grad(q))*dx, inner(nn, nn)*dx


# Now solve the Stokes-NN forward problem
up, reg = stokes(W, nn)
u_nn, p_nn = up.split(deepcopy=True)
plot(u_nn, "out/u_nn0.png")
plot(p_nn, "out/p_nn0.png")

J = assemble((up - up_stab)**2*dx) 
print(f"J={J}")
#print(f"reg={reg}")
#J += 1e4*reg

reg = 0
for W in [W_1, W_2, b_1, W_3_1, W_3_2, b_2]:
    reg += 1e0*assemble(W**2*dx)
print(f"reg={reg}")
#J += reg

Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3_1), Control(W_3_2)])
C_up = Control(up)

set_log_level(LogLevel.ERROR)

minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 20})

print("|U - d| = ", assemble(inner(C_up.tape_value() - up_stab, C_up.tape_value() - up_stab)*dx)**0.5)

u_nn, p_nn = C_up.tape_value().split(deepcopy=True)
plot(u_nn, "out/u_nn.png")
plot(p_nn, "out/p_nn.png")
