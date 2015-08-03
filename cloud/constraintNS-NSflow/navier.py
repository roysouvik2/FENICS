from dolfin import *

# Create mesh and define function space.

mesh = UnitSquare(128,128)
File("mesh.xml") << mesh

out_file = File("exact_velocity.pvd")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

Re = 1000
nu = 1.0/Re

# Define Variational Problem

(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
u0 = interpolate(Constant((0.0,0.0)),V)
w = Function(W)
u1 = Function(V)
p1 = Function(Q)


# Boundary Conditions

noslip  = DirichletBC(W.sub(0), (0,0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS")

lid  = DirichletBC(W.sub(0), (1,0), "x[1] > 1.0 - DOLFIN_EPS")

pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [noslip, lid, pref]


g = Constant((0.0,0.0))
eps = 1.0
tol  = 1.0E-7
iter = 0
maxiter = 100
F = nu*inner(grad(u), grad(v))*dx \
    -div(v)*p*dx - q*div(u)*dx\
+ inner(grad(u)*u0, v)*dx \
- inner(g,v)*dx
   
a = lhs(F)
L = rhs(F)
while eps > tol and iter < maxiter:
     iter += 1
     solve(a == L, w, bcs)
     (u1,p1) = w.split()
     eps = errornorm(u0, u1, 'L2')
     print "L2error =", eps
     print "Iterations = ", iter
     u0.assign(u1)
out_file << u1
plot(u1, interactive = 'True')
File("exact.xml") << w.vector()


