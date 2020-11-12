from dolfin import *

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


def stokes(mesh, Velm, Qelm):
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
    W = FunctionSpace(mesh, MixedElement([Velm, Qelm]))
    bcs = [DirichletBC(W.sub(0), Constant((0, 0)), 'near(x[1]*(1-x[1]), 0)'),
           DirichletBC(W.sub(0), Expression(('x[1]*(1-x[1])', '0'), degree=2), 'near(x[0], 0)')]
    
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = (inner(grad(u), grad(v))*dx + inner(p, div(v))*dx + 
         inner(q, div(u))*dx)
    x, y = SpatialCoordinate(mesh)
    f = Constant((0, 0))

    n = FacetNormal(mesh)
    L = inner(f, v)*dx
    
    wh = Function(W)
    solve(a == L, wh, bcs)

    uh, ph = wh.split(deepcopy=True)

    return uh, ph


mesh = UnitSquareMesh(32, 32, 'crossed')

stable = [VectorElement('Lagrange', triangle, 2),
          FiniteElement('Lagrange', triangle, 1)]

u_stab, p_stab = stokes(mesh, *stable)
File('u_stab.pvd') << u_stab
File('p_stab.pvd') << p_stab

# This one yield checker-board pattern of reasonable magnitude
unstable = [VectorElement('Lagrange', triangle, 1),
            FiniteElement('Lagrange', triangle, 1)]

u_stab, p_stab = stokes(mesh, *unstable)
File('u_ustab.pvd') << u_stab
File('p_ustab.pvd') << p_stab


