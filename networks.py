import dolfin as df
import numpy as np
import ufl

# TODO: more ...
linear = lambda x: x
ReLu = lambda x: ufl.Max(x, df.Constant(0))

class DenseLayer(object):
    def __init__(self, input_dim, output_dim, mesh, use_bias=True, activation=linear):
        self.input_dim = input_dim  # Remeber for consistency checks
        # Represent (odim, idim) matrix by rows
        W_space = df.VectorFunctionSpace(mesh, 'R', 0, input_dim)
        self.weights = [self.init_weights(W_space) for row in range(output_dim)]

        if use_bias:
            b_space = df.VectorFunctionSpace(mesh, 'R', 0, output_dim)
            self.bias = self.init_bias(b_space)
        else:
            self.bias = None

        self.rho = activation

    def init_weights(self, V):
        '''Randomize'''
        w = df.Function(V)
        values = w.vector().get_local()
        values = np.random.rand(len(values))
        w.vector().set_local(values)

        return w

    def init_bias(self, V):
        '''Constant'''
        b = df.Function(V)
        values = b.vector().get_local()
        values = 0.5*np.ones_like(values)
        b.vector().set_local(values)

        return b

    # ----------------------------------------------------------------

    def __call__(self, x):
        '''Apply the layer'''
        assert x.ufl_shape == (self.input_dim, )

        Wx = as_vector([dot(Wi, x) for Wi in self.weights])
        if self.bias is not None:
            Wx = Wx + self.bias

        # Pointwise nonlinearity
        output = as_vector([self.rho(Wx[i]) for i in range(len(self.weights))])

        return output

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(3, 3)
    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, 'CG', 1)
    p = interpolate(Expression('x[0]+x[1]', degree=1), V)
    # TODO: auto flatten non-scalars
    inputs = as_vector([x, y, p, grad(p)[0], grad(p)[1]])

    out = DenseLayer(input_dim=5, output_dim=4, mesh=mesh)(inputs)
    out = DenseLayer(input_dim=4, output_dim=2, mesh=mesh)(out)
    out = DenseLayer(input_dim=2, output_dim=2, mesh=mesh, activation=ReLu)(out)
    out = DenseLayer(input_dim=2, output_dim=1, mesh=mesh, activation=ReLu)(out)
        
    print(assemble(inner(out, out)*dx))
