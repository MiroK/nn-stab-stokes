from dolfin import *
from dolfin_adjoint import *
from numpy.random import rand, seed
seed(21)


def make_data():
    '''Poisson problem on square 

    -div(grad(u)) = 0 in [0.1, 1.1]^2
                u = g on bdry
    '''
    u_true = Expression('std::log(x[0]*x[0]+x[1]*x[1])', degree=5)
    sigma_true = Expression(('2*x[0]/(x[0]*x[0]+x[1]*x[1])',
                             '2*x[1]/(x[0]*x[0]+x[1]*x[1])'), degree=5)
    return sigma_true, u_true


def mixed_poisson(W, u0, nn=None):
    '''Mixed Poisson solver for -div(sigma) = 0, sigma = grad(u), u=u0 on bdry'''
    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)

    n = FacetNormal(W.mesh())
    F = inner(sigma, tau)*dx + inner(u, div(tau))*dx + inner(v, div(sigma))*dx - inner(u0, dot(tau, n))*ds
    
    if nn is not None:
        Fnn, reg, _ = nn(sigma, u, tau, v)
        F += Fnn
    
    solve(F == 0, w)
    
    if nn:
        return w, assemble(reg)
    else:
        return w

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    eps = 1E-2
    
    sigma_true, u_true = make_data()

    Selm = FiniteElement('Raviart-Thomas', triangle, 1)  # Stable with Velm
    # Selm = VectorElement('Lagrange', triangle, 1)        # Unstable
    
    Velm = FiniteElement('Discontinuous Lagrange', triangle, 0)
    Welm = MixedElement([Selm, Velm])

    e_u0, e_sigma0, h0 = None, None, None
    for n in (2**k for k in range(4, 5)):
        mesh = RectangleMesh(Point(0.1, 0.1), Point(1.1, 1.1), n, n)

        W = FunctionSpace(mesh, Welm)
        
        wh = mixed_poisson(W, u_true, nn=None)
        sigmah, uh = wh.split(deepcopy=True)
        e_sigma = errornorm(sigma_true, sigmah, 'Hdiv')
        e_u = errornorm(u_true, uh, 'L2')
        h = mesh.hmin()
        
        if e_u0 is None:
            rate_u, rate_sigma = -1, -1
        else:
            rate_u = ln(e_u/e_u0)/ln(h/h0)
            rate_sigma = ln(e_sigma/e_sigma0)/ln(h/h0)

        e_u0, e_sigma0, h0 = e_u, e_sigma, h
        print(h0, e_sigma0, rate_sigma, e_u0, rate_u)

    df.File('out/mp_uh.pvd') << uh
    df.File('out/mp_sigmah.pvd') << sigmah
    # import matplotlib.pyplot as plt

    df.plot(uh)
    # df.plot(interpolate(u_true, uh.function_space()), label='true')

    # plt.legend()
    # plt.show()
