from numpy.polynomial.legendre import legval
from dolfin import mpi_comm_world, Vector
from scipy.linalg import eigh
from block.object_pool import vec_pool
from block.block_base import block_base
from block import block_transpose
from math import sqrt
import numpy as np

COMM = mpi_comm_world()


def legendre(k):
    '''k-th Legendre polynomial'''
    return lambda x: legval(x, np.r_[np.zeros(k), 1.])


def shen(k):
    '''Shen basis foos from

        Efficient Spectral-Galerkin Method I. Direct Solvers for the
        Second and Fourth Order Equations Using Legendre Polynomials
    '''
    return lambda x: (legendre(k)(x)-legendre(k+2)(x))/sqrt(4*k+6)


def as_expression(expr, degree, **kwargs):
    '''Wrap function as Expression'''
    class FooExpr(Expression):
        def eval(self, values, x):
            values[:] = expr(x)
    return FooExpr(degree=degree, **kwargs)


def block_op(f, shape):
    '''
    Operator whose action is in f is wrapped from cbc.block
    f is expected to map Vector to Vector (I mean types)
    '''
    class OP(block_base):
        def matvec(self, b): 
            return f(b)

        @vec_pool
        def create_vec(self, dim=1):
            return Vector(COMM, shape[dim])
    
    return OP()


def numpy_op(A):
    '''Wrapping numpy matrix as operator'''
    def f(b):
        b = b.array()
        x_values = A.dot(b)
        x = Vector(COMM, A.shape[0])
        x.set_local(x_values)
        x.apply('insert')
        return x
    return block_op(f, A.shape)


def fractional_laplace(V, s, bcs=[]):
    '''(M*U)*Lambda^s*(MU)'''
    try:
        u, v = V
    except TypeError:
        u, v = TrialFunction(V), TestFunction(V)

    if bcs:
        if u.ufl_shape:
            L = Constant((0, )*u.ufl_shape)
        else:
            L = Constant(0)
        L = inner(L, v)*dx

        A, _ = assemble_system(inner(grad(u), grad(v))*dx, L, bcs)
        M, _ = assemble_system(inner(u, v)*dx, L, bcs)
    else:
        A = assemble(inner(grad(u), grad(v))*dx)
        M = assemble(inner(u, v)*dx)

    A = A.array()
    M = M.array()

    lmbda, U = eigh(A, M)
    L = np.diag(lmbda**s)
    U = M.dot(U)
    Ut = U.T

    B = np.array(U.dot(L.dot(Ut)))

    return numpy_op(B)


class Transformation(object):
    '''Tranforming between V and W spaces by L2 projection'''
    def __init__(self, V, (p, q)):
        # Setup solver for getting to V
        u, v = TrialFunction(V), TestFunction(V)
        M = assemble(inner(u, v)*dx)
        Minv = LUSolver(COMM, M); Minv.parameters['reuse_factorization'] = True
   
        # Setup solver for getting to W
        M = assemble(inner(p, q)*dx)
        Ninv = LUSolver(COMM, M); Ninv.parameters['reuse_factorization'] = True

        # Compute the tranformation
        T = assemble(inner(p, v)*dx)

        self.MV = Minv
        self.MW = Ninv
        self.T = T

    def to_V(self):
        '''From W to V'''
        def f(b):
            x = Vector(COMM, self.T.size(0))
            self.MV.solve(x, self.T*b)
            return x
        return block_op(f, (self.T.size(0), self.T.size(1)))

    def to_W(self):
        '''From V to W'''
        def f(b):
            x = Vector(COMM, self.T.size(1))
            self.MW.solve(x, block_transpose(self.T)*b)
            return x
        return block_op(f, (self.T.size(1), self.T.size(0)))

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dolfin import *
    # So how many m(legendre functions) to get the approximation of fractional
    # Laplacian right?
    ncells = 20
    s = 0.5

    mesh = IntervalMesh(ncells, -1, 1)
    # FEM space
    V = FunctionSpace(mesh, 'CG', 1)
    fh = interpolate(Expression('(1-x[0])*(1+x[0])', degree=2), V)

    # Applying the laplacian in FEM 
    Dh = fractional_laplace(V, s=s, 
                            bcs=DirichletBC(V, Constant(0), 'on_boundary'))
    d_fh = Function(V, Dh*fh.vector())

    nspectral = 10
    # Spactral space
    fs = [as_expression(shen(k), k+1, cell=interval) for k in range(nspectral)]

    W = VectorFunctionSpace(mesh, 'R', 0, dim=len(fs))
    ps, qs = TrialFunction(W), TestFunction(W)
    p = sum([ui*fi for ui, fi in zip(ps, fs)])
    q = sum([vi*fi for vi, fi in zip(qs, fs)])

    # Applying the laplacian in spect
    T = Transformation(V, (p, q))
    to_W = T.to_W()
    to_V = T.to_V()

    d_N = fractional_laplace((p, q), s=s)
    y = toV*d_N*to_W*fh.vector()

    FN = to_W*fh.vector()
    FN = FN.array()
    fN = as_expression(lambda y: sum(ci*fi(y) for ci, fi in zip(FN, fs)),
                       degree=len(FN)+2)

    x = np.linspace(-1, 1, 100)
    plt.figure()
    plt.plot(x, map(fh, x), label='f')
    plt.plot(x, map(d_fh, x), label='Df')
    plt.plot(x, map(fN, x))

    plt.legend(loc='best')
    plt.show()
    
