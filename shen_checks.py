from common import fractional_laplace_shen, mass_matrix_shen
from common import as_expression, shen
import scipy.sparse.linalg as LA
from dolfin import *
import numpy as np

COMM = mpi_comm_world()

s = 1.0
k = 2

# On (-1, 1) eigen basis in uk cos(k*pi/2 x) and sin(k*pi* x)
# Exact
f = Expression('2*cos(l*pi/2*x[0])+sin(k*pi*x[0])', degree=6, k=k, l=k+1)
df0 = Expression('2*pow(l*pi/2, 2*s)*cos(l*pi/2*x[0])+pow(k*pi, 2*s)*sin(k*pi*x[0])',
                 degree=6, k=k, l=k+1, s=s)

mesh = IntervalMesh(200, -1, 1)
# Different basis
for nspectral in range(3, 16):
    fs = [as_expression(shen(k), k+1, cell=interval) for k in range(nspectral)]

    W = VectorFunctionSpace(mesh, 'R', 0, dim=len(fs))
    qs = TestFunction(W)
    q = sum([vi*fi for vi, fi in zip(qs, fs)])
    # Represent in coefficient space
    b = assemble(inner(q, f)*dx).array()
    M = mass_matrix_shen(W.dim())

    fh_values = LA.spsolve(M, b)
    fh = Vector(COMM, len(fh_values))
    fh.set_local(fh_values)
    # Derivative in coef space
    d = fractional_laplace_shen(W.dim(), s)

    dfh = d*fh
    dfh = dfh.array()

    df = as_expression(lambda x: sum(Ck*fk(x) for Ck, fk in zip(dfh, fs)), degree=6)
    fh = as_expression(lambda x: sum(Ck*fk(x) for Ck, fk in zip(fh_values, fs)), degree=6)

    print sqrt(assemble(inner(df-df0, df-df0)*dx(domain=mesh)))

# Finally something more challenging. Note that in the basis coefs of f should
# be just 1, zeros(...)
f = Expression('(1-x[0]*x[0])', degree=1, k=k, l=k+1)
# Based on representation of f in the basis, no sine terms
Vfine = FunctionSpace(mesh, 'CG', 6)
df0 = Function(Vfine)
values = df0.vector().get_local()
x = Vfine.tabulate_dof_coordinates()
for k in range(1, 20000, 2):
    C = 32*sin(k*pi/2)/((pi*k)**3)  # Repr
    C *= (k*pi/2)**(2*s)            # What the derivative brings
    values += C*np.cos(k*pi/2*x)    # Basis
df0.vector().set_local(values)

for nspectral in range(3, 16):
    fs = [as_expression(shen(k), k+1, cell=interval) for k in range(nspectral)]

    W = VectorFunctionSpace(mesh, 'R', 0, dim=len(fs))
    qs = TestFunction(W)
    q = sum([vi*fi for vi, fi in zip(qs, fs)])
    # Represent in coefficient space
    b = assemble(inner(q, f)*dx).array()
    M = mass_matrix_shen(W.dim())

    fh_values = LA.spsolve(M, b)
    fh = Vector(COMM, len(fh_values))
    fh.set_local(fh_values)
    # Derivative in coef space
    d = fractional_laplace_shen(W.dim(), s)

    dfh = d*fh
    dfh = dfh.array()

    df = as_expression(lambda x: sum(Ck*fk(x) for Ck, fk in zip(dfh, fs)), degree=6)
    fh = as_expression(lambda x: sum(Ck*fk(x) for Ck, fk in zip(fh_values, fs)), degree=6)

    print sqrt(assemble(inner(df-df0, df-df0)*dx(domain=mesh)))

import matplotlib.pyplot as plt

x = mesh.coordinates()

plt.figure()
plt.plot(x, map(f, x))
plt.plot(x, map(fh, x))

plt.figure()
plt.plot(x, map(df0, x))
plt.plot(x, map(df, x))

plt.show()
