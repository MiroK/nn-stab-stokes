from dolfin import *
from dolfin_adjoint import *

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


def stokes(W, rhs=None):
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

    if rhs:
       F -= rhs(u, p, v, q)
    
    solve(F == 0, up, bcs)

    return up



if __name__== "__main__":
    mesh = UnitSquareMesh(32, 32, 'crossed')

    stable = [VectorElement('Lagrange', triangle, 2),
              FiniteElement('Lagrange', triangle, 1)]
    W = FunctionSpace(mesh, MixedElement(stable))

    u_stab, p_stab = stokes(W).split(deepcopy=True)
    File('u_stab.pvd') << u_stab
    File('p_stab.pvd') << p_stab

    # This one yield checker-board pattern of reasonable magnitude
    unstable = [VectorElement('Lagrange', triangle, 1),
                FiniteElement('Lagrange', triangle, 1)]
    W = FunctionSpace(mesh, MixedElement(unstable))

    u_stab, p_stab = stokes(W).split(deepcopy=True)

    File('u_ustab.pvd') << u_stab
    File('p_ustab.pvd') << p_stab
