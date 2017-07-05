# On (0, 1) eigen basis in uk sin(k*pi*x)*sqrt(2)
# Fractional laplacian d^s is then d^s u_k = (k*pi)^(2s) u_k
# We check this property here with fem
from common import fractional_laplace
from dolfin import *
import numpy as np

s = 1.0
k = 2
# Exact
f = Expression('sin(k*pi*x[0])', degree=5, k=k)
df0 = Expression('pow(k*pi, 2*s)*sin(k*pi*x[0])', degree=4, k=k, s=s)

for ncells in (2**i for i in range(3, 11)):


    mesh = UnitIntervalMesh(ncells)
    V = FunctionSpace(mesh, 'CG', 1)
    bcs = DirichletBC(V, Constant(0), 'on_boundary')

    fh = interpolate(f, V).vector()
    # Numeric
    d = fractional_laplace(V, bcs=bcs, s=s)

    dfh = d*fh
    # As function
    u, v = TrialFunction(V), TestFunction(V)
    M = assemble(inner(u, v)*dx)
    df = Function(V)

    solve(M, df.vector(), dfh)

    print errornorm(df0, df, 'L2')
print

# On (-1, 1) eigen basis in uk cos(k*pi/2 x) and sin(k*pi* x)
# Exact
f = Expression('2*cos(l*pi/2*x[0])+sin(k*pi*x[0])', degree=1, k=k, l=k+1)
df0 = Expression('2*pow(l*pi/2, 2*s)*cos(l*pi/2*x[0])+pow(k*pi, 2*s)*sin(k*pi*x[0])',
                 degree=4, k=k, l=k+1, s=s)

for ncells in (2**i for i in range(3, 11)):
    mesh = IntervalMesh(ncells, -1, 1)
    V = FunctionSpace(mesh, 'CG', 1)
    bcs = DirichletBC(V, Constant(0), 'on_boundary')

    fh = interpolate(f, V).vector()
    # Numeric
    d = fractional_laplace(V, bcs=bcs, s=s)

    dfh = d*fh
    # As function
    u, v = TrialFunction(V), TestFunction(V)
    M = assemble(inner(u, v)*dx)
    df = Function(V)

    solve(M, df.vector(), dfh)

    print errornorm(df0, df, 'L2')
print

# Finally something more challenging
f = Expression('(1-x[0]*x[0])', degree=1, k=k, l=k+1)
# Based on representation of f in the basis, no sine terms
Vfine = FunctionSpace(mesh, 'CG', 4)
df0 = Function(Vfine)
values = df0.vector().get_local()
x = Vfine.tabulate_dof_coordinates()
for k in range(1, 20000, 2):
    C = 32*sin(k*pi/2)/((pi*k)**3)  # Repr
    C *= (k*pi/2)**(2*s)            # What the derivative brings
    values += C*np.cos(k*pi/2*x)    # Basis
df0.vector().set_local(values)

for ncells in (2**i for i in range(3, 11)):
    mesh = IntervalMesh(ncells, -1, 1)
    V = FunctionSpace(mesh, 'CG', 1)
    bcs = DirichletBC(V, Constant(0), 'on_boundary')

    fh = interpolate(f, V).vector()
    # Numeric
    d = fractional_laplace(V, bcs=bcs, s=s)

    dfh = d*fh
    # As function
    u, v = TrialFunction(V), TestFunction(V)
    M = assemble(inner(u, v)*dx)
    df = Function(V)

    solve(M, df.vector(), dfh)

    print errornorm(df0, df, 'L2')
print
plot(df0, mesh=mesh)
plot(df, title='numeric')
interactive()
