from dolfin import *
from dolfin_adjoint import *
from helpers import plot
from numpy.random import rand, seed
seed(21)


def make_data(eps, mesh, elm=None):
    '''On (0, 1) solve -eps u'' + u' = 0 with u(0) = 0, u(1) = 1'''
    if elm is None:
        elm = FiniteElement('Lagrange', interval, 1)
    assert mesh.ufl_cell() == interval

    u_true = Expression('(1-std::exp((x[0]-1)/epsilon))/(1-std::exp(-1./epsilon))', degree=5,
                        epsilon=eps)

    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(u_true, V)

    return u, u_true


def advection_diffusion(V, alpha, nn=None):
    '''XXX'''
    bcs = [DirichletBC(V, Constant(1), 'near(x[0], 0)'),
           DirichletBC(V, Constant(0), 'near(x[0], 1)')]
        
    u, v = Function(V), TestFunction(V)

    F = Constant(alpha)*inner(grad(u), grad(v))*dx + inner(u.dx(0), v)*dx
    print('Alpha', alpha)
    if nn is not None:
        Fnn, reg, _ = nn(u, v)
        F += Fnn
    
    solve(F == 0, u, bcs)
    
    if nn:
        return u, assemble(reg)
    else:
        return u

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    eps = 1E-2
    
    _, u_true = make_data(eps, UnitIntervalMesh(2))

    e0, h0 = None, None
    for n in (2**k for k in range(5, )):
        mesh = UnitIntervalMesh(n)

        V = FunctionSpace(mesh, 'CG', 2)
        bcs = [DirichletBC(V, Constant(1), 'near(x[0], 0)'),
               DirichletBC(V, Constant(0), 'near(x[0], 1)')]
        
        u, v = TrialFunction(V), TestFunction(V)

        a = Constant(eps)*inner(grad(u), grad(v))*dx + inner(u.dx(0), v)*dx
        L = inner(Constant(0), v)*dx

        uh = Function(V)
        solve(a == L, uh, bcs)

        e = errornorm(u_true, uh, 'H1')
        h = mesh.hmin()
        
        if e0 is None:
            rate = -1
        else:
            rate = ln(e/e0)/ln(h/h0)

        e0, h0 = e, h
        print(h0, e0, rate)
        
    import matplotlib.pyplot as plt

    df.plot(uh)
    df.plot(interpolate(u_true, uh.function_space()), label='true')

    plt.legend()
    plt.show()
