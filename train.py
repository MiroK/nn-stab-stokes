from fenics import *
from fenics_adjoint import *
import ufl
import sys
import numpy


generate = "--gen" in sys.argv

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

U = Function(V)
X = SpatialCoordinate(mesh)

if generate:
    X = SpatialCoordinate(mesh)

    kappa = cos(X[0]+X[1])
    a = inner(kappa*u, v)*dx + inner(grad(u), grad(v))*dx
    L = Constant(1)*v*dx

    solve(a == L, U)

    out_f = XDMFFile("test_poisson_nn_u.xdmf")
    out_f.write_checkpoint(U, "U", 0.0, XDMFFile.Encoding.HDF5, False)
    out_f.close()

    p = plot(U)
    import matplotlib.pyplot as plt
    plt.colorbar(p)
    plt.show()
else:
    d = Function(V)
    obs = XDMFFile("test_poisson_nn_u.xdmf")
    obs.read_checkpoint(d, "U", -1)

    R = VectorFunctionSpace(mesh, "R", 0, dim=50)
    W_1, W_2, b_1, W_3, W_4 = Function(R), Function(R), Function(R), Function(R), Function(R)
    R2 = FunctionSpace(mesh, "R", 0)
    b_2 = Function(R2)
    rhs = Function(V)

    from numpy.random import rand, seed
    seed(21)

    W_1.vector()[:] = 2*rand(R.dim())
    W_2.vector()[:] = 2*rand(R.dim())
    W_3.vector()[:] = 2*rand(R.dim())
    W_4.vector()[:] = 2*rand(R.dim())
    b_1.vector()[:] = 2*rand(R.dim())
    b_2.vector()[:] = 2*rand(R2.dim())

    c_vars = [W_1, b_1, W_2, b_2, W_3, W_4]

    c_values = []
    #for i, c in enumerate(c_vars):
    #    c.vector()[:] = numpy.load("test_poisson_nn_fail_c_{}.npy".format(i))

    def sigma_(vec, func=ufl.tanh):
        v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
        return ufl.as_vector(v)
    a = 1.0
    relu = lambda vec: conditional(ufl.gt(vec, 0), vec, a*(ufl.exp(vec) - 1))
    sigma = lambda vec: sigma_(vec, func=relu)

    U_ = Function(V)
    from pyadjoint.placeholder import Placeholder
    p = Placeholder(U_)

    a1 = inner(inner(W_4, sigma(ufl.transpose(as_vector([W_1, W_2, W_3])) * as_vector([U_, *X]) + b_1)) + b_2, v) * dx + inner(
        grad(U), grad(v)) * dx - Constant(
        1) * v * dx   #+ inner(inner(W_4, sigma(as_vector(W_3, W_4)*X + b_3)) + b_4, v)*dx

    dt = 0.01
    for i in range(100):
        a2 = (U-U_)/dt*v*dx + a1
        solve(a2 == 0, U)
        U_.assign(U)

    a1 = inner(inner(W_4, sigma(ufl.transpose(as_vector([W_1, W_2, W_3])) * as_vector([U, *X]) + b_1)) + b_2,
               v) * dx + inner(
        grad(U), grad(v)) * dx - Constant(
        1) * v * dx
    solve(a1 == 0, U)

    J = assemble((U - d)**2*dx)
    #p.set_value(U)
    #print("J = ", J)
    #exit()
    Jhat = ReducedFunctional(J, [Control(W_1), Control(b_1), Control(W_2), Control(b_2), Control(W_3), Control(W_4)])
    C_u = Control(U)

    set_log_level(LogLevel.ERROR)
    
    W_1, b_1, W_2, b_2, W_3, W_4 = minimize(Jhat, tol=1e-200, options={"disp": True, "gtol": 1e-12, "maxiter": 20})
    
    

    #print(project(inner(c_opt[1], sigma(c_opt[0])), V).vector().get_local())
    print("|U - d| = ", assemble(abs(C_u.tape_value() - d)*dx))
    print("|kappa(x)*d - N(U, X)| = ", assemble(abs(cos(X[0]+X[1])*d - (inner(W_4, sigma(ufl.transpose(as_vector([W_1, W_2, W_3])) * as_vector([C_u.tape_value(), *X]) + b_1)) + b_2))*dx))

    p = plot(d)
    import matplotlib.pyplot as plt
    plt.colorbar(p)
    plt.savefig("OBS.png")
    plt.clf()
    p = plot(C_u.tape_value())
    plt.colorbar(p)
    plt.savefig("U_OPT.png")

    from IPython import embed; embed()

