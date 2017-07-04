from numpy.polynomial.legendre import legval
from dolfin import mpi_comm_world, Vector
from scipy.sparse import diags
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

        print A.array()
        print M.array()

    A = A.array()
    M = M.array()

    lmbda, U = eigh(A, M)
    L = np.diag(lmbda**s)
    U = M.dot(U)
    Ut = U.T

    lmbda, U = eigh(A, M)
    L = np.diag(lmbda**s)
    U = M.dot(U)
    Ut = U.T

    B = np.array(U.dot(L.dot(Ut)))

    return numpy_op(B)
    B = np.array(U.dot(L.dot(Ut)))

    return numpy_op(B)


def fractional_laplace_shen(n, s):
    '''A is I, M is tridiag'''
    weight = lambda k: 1/sqrt(4*k + 6)
    # The matrix is tridiagonal and symmetric
    # Main
    main_diag = np.array([weight(i)**2*((2./(2*i+1) + 2./(2*(i+2)+1)))
                          for i in range(n)])
    # Upper
    up_diag = np.array([-weight(i)*weight(i+2)*(2./(2*(i+2)+1))
                        for i in range(n-2)])
   
    if n < 3:
        M = diags(main_diag, 0)
    else:
        M = diags([up_diag, main_diag, up_diag], [-2, 0, 2])
    M = M.todense()

    lmbda, U = eigh(np.eye(n), M)
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
    ncells = 256
    s = 0.5

    mesh = IntervalMesh(ncells, -1, 1)
    # FEM space
    V = FunctionSpace(mesh, 'CG', 1)
    bcs=DirichletBC(V, Constant(0), 'on_boundary')
    fh = interpolate(Expression('sin(k*pi*x[0])', degree=2, k=2), V)
    # fh = Function(V)
    # fh.vector().set_local(np.random.rand(fh.vector().local_size()))
    # bcs.apply(fh.vector())

    # Applying the laplacian in FEM 
    Dh = fractional_laplace(V, s=s, bcs=bcs)
    d_fh = Function(V, Dh*fh.vector())

    for nspectral in range(2, 10, 2):
        # Spectral space
        fs = [as_expression(shen(k), k+1, cell=interval) for k in range(nspectral)]

        W = VectorFunctionSpace(mesh, 'R', 0, dim=len(fs))
        ps, qs = TrialFunction(W), TestFunction(W)
        p = sum([ui*fi for ui, fi in zip(ps, fs)])
        q = sum([vi*fi for vi, fi in zip(qs, fs)])

        # Applying the laplacian in spect
        T = Transformation(V, (p, q))
        toW = T.to_W()
        toV = T.to_V()

        d_N = fractional_laplace_shen(W.dim(), s=s)
        xx = d_N*toW*fh.vector()
        y = toV*xx
        fy = Function(V, y)

        FN = toW*fh.vector()
        FN = FN.array()
        fN = as_expression(lambda y: sum(ci*fi(y) for ci, fi in zip(FN, fs)),
                           degree=len(FN)+2)
        g = as_expression(lambda y: sum(ci*fi(y) for ci, fi in zip(xx, fs)),
                          degree=len(xx)+2)

        # print y.array()
        print nspectral
        # 
        print 'norm(fh-fN)', sqrt(assemble(inner(fh-fN, fh-fN)*dx))
        print 'norm(fh)', sqrt(assemble(inner(fh, fh)*dx)),
        print 'norm(fN)', sqrt(assemble(inner(fN, fN)*dx(domain=mesh)))

        print 'norm(dfh-dfN)', sqrt(assemble(inner(d_fh-g, d_fh-g)*dx))
        print 'norm(dfh)', sqrt(assemble(inner(d_fh, d_fh)*dx)),
        print 'norm(dfN)', sqrt(assemble(inner(g, g)*dx(domain=mesh)))

        print

        x = np.linspace(-1, 1, 100)
        # plt.figure()
        # plt.plot(x, map(fh, x), label='fh')
        # plt.plot(x, map(fN, x), label='fN')
        # plt.legend(loc='best')

        plt.figure()
        plt.plot(x, map(d_fh, x), label='$-\Delta^{0.5} fh, %d$' % nspectral)
        plt.plot(x, map(fy, x), label='$T(-\Delta^{0.5} fN)T$')
        plt.legend(loc='best')
    plt.show()
