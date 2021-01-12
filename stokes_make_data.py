from dolfin import *
from dolfin_adjoint import *
from helpers import plot
from numpy.random import rand, seed
seed(21)

# For problem
#   -Delta u + grad(p) = f
#               div(u) = 0
#
# Stable FEM discretization requires inf-sup condition. Is satisied
# by them system
#
#   [A, B';
#    B, 0]
# 
# This does not hold for P1-P1 elements. For stability, a suitable positive
# operator C is added to the system so that
#
#   [A, B';
#    B, -C]
#
# satisfies the modified inf-sup condition.
#
# Possible stabilizations are to e.g. to let C = -eps(h)*Delta(p) (Brezzi-Pitkaranta)
# Other one is based on L^2 project (Burman), C = M-Lump(M)
#
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


def stokes(W, nn=None):
    '''
           no-slip
    ---------------------------
    \                          |
     \ parabolic inflow        |p = 0
     /                         |
    /                          |
    ----------------------------
          no-slip
    '''
    # NOTE: nice thing about this setup is that P2-P1 get the exact solution
    # of the problem
    bcs = [DirichletBC(W.sub(0), Constant((0, 0)), 'near(x[1]*(1-x[1]), 0)'),
           DirichletBC(W.sub(0), Expression(('x[1]*(1-x[1])', '0'), degree=2), 'near(x[0], 0)')]
    
    up = Function(W)
    u, p = split(up)
    v, q = TestFunctions(W)

    F = (inner(grad(u), grad(v))*dx + inner(p, div(v))*dx + 
         inner(q, div(u))*dx)

    if nn:
        Fnn, reg, _ = nn(u, p, v, q)
        F += Fnn
    
    solve(F == 0, up, bcs)
    
    #solve(F == 0, up, bcs, solver_parameters={"nonlinear_solver": "snes", "snes_solver": {"line_search": "bt", "linear_solver": "lu", "report": False}},  # [basic,bt,cp,l2,nleqerr]
    #              form_compiler_parameters={"optimize": True})

    if nn:
        return up, assemble(reg)
    else:
        return up



if __name__== "__main__":
    mesh = UnitSquareMesh(16, 16, 'crossed')

    stable = [VectorElement('Lagrange', triangle, 2),
              FiniteElement('Lagrange', triangle, 1)]
    W = FunctionSpace(mesh, MixedElement(stable))

    up = stokes(W)

    # Add noise
    eps_noise = 0
    up.vector()[:] += eps_noise*rand(W.dim())

    u_stab, p_stab = up.split(deepcopy=True)
    plot(u_stab, "out/u_stab.png")
    plot(p_stab, "out/p_stab.png")

    with HDF5File(MPI.comm_world, "out/up_stab.h5", "w") as xdmf:
        xdmf.write(up, "up")
