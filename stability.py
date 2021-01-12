from dolfin import *
import numpy as np


def preconditioned_stokes(W, stab=None):
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
    
    up = TrialFunction(W)
    u, p = split(up)
    v, q = TestFunctions(W)

    F = (inner(grad(u), grad(v))*dx + inner(p, div(v))*dx + 
         inner(q, div(u))*dx)

    if stab is not None:
        h = CellDiameter(mesh)
        F = F + stab(h, p, q)

    a, L = lhs(F), rhs(F)
    if L.empty():
        L = inner(Constant(0), q)*dx
    A, _ = assemble_system(a, L, bcs)

    # Preconditioner form is
    a_prec = inner(grad(u), grad(v))*dx + inner(p, q)*dx - stab(h, p, q)
    B, _ = assemble_system(a_prec, L, bcs)
    
    return A, B


def is_symmetric(mat, tol=1E-10):
    return np.linalg.norm(mat - mat.T) < tol

# --------------------------------------------------------------------

if __name__ == '__main__':
    from scipy.linalg import eigvals, eigvalsh
    from numpy.linalg import cond
    
    stable = [VectorElement('Lagrange', triangle, 1),
              FiniteElement('Lagrange', triangle, 1)]

    stab = lambda h, p, q: -(h**2)*inner(grad(p), grad(q))*dx
    for n in (2, 4, 8, 16):
        mesh = UnitSquareMesh(n, n, 'crossed')
        W = FunctionSpace(mesh, MixedElement(stable))
        A, B = preconditioned_stokes(W, stab=stab)
        A, B = A.array(), B.array()

        print(A.shape)
        if is_symmetric(A):
            eigw = eigvalsh(A, B)
        else:
            eigw = eigvals(A, B)

        lmin, lmax = np.sort(np.abs(eigw))[[0, -1]]
        print (lmin, lmax, lmax/lmin)

    



    
