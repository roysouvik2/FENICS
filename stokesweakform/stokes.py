from dolfin import *

# Create mesh and define function space.

mesh = UnitSquare(50,50)
File("mesh.xml") << mesh

out_file = File("exact_velocity.pvd")

# Define function spaces (P2-P1)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
R = VectorFunctionSpace(mesh, "CG", 2)
system = MixedFunctionSpace([V,Q,R])

Re = 10
nu = 1.0/Re

# Define Variational Problem

(u,p,psi) = TrialFunctions(system)
(v,q,lam) = TestFunctions(system)

w = Function(system)
u1 = Function(V)
p1 = Function(Q)
lam1 = Function(R)


# Boundary Conditions


g = Constant((1.0,1.0))

F = nu*inner(grad(u), grad(v))*dx \
     -div(v)*p*dx - q*div(u)*dx \
    + inner((u-g),lam)*ds + inner(psi,v)*ds
   

a = lhs(F)
L = rhs(F)

solve(a==L, w)

(u1,p1, lam1) = w.split()

out_file << u1
plot(u1, interactive = 'True')
File("exact.xml") << w.vector()


