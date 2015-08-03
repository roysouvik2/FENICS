from dolfin import *
from math import *

k = 0.01
Kappa = 1.0/k

# Defining Mesh.

nx = 100
nt = 100

mesh = Rectangle(0,0,pi,pi, nx, nt)

h = CellSize(mesh)
print(mesh.hmax())

# Defining Function Space.
yex = Expression(" sin(x[1])*sin(x[0])")
uex = Expression("-sin(x[1])")


V = FunctionSpace(mesh,"CG",2)
Q = FunctionSpace(mesh,"CG",2)
W = V * Q

yex = interpolate(yex, V)
#plot(yex, interactive = 'True')

# Defining Boundaries

class Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], pi)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 0.0) 

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], pi)

noslip = Constant(0.0)

bc0 = DirichletBC(W.sub(0), noslip, Bottom())
bc1 = DirichletBC(W.sub(1), noslip, Top())
bcs = [bc0,bc1]

# Defining Boundary Measures

boundaries = FacetFunction("uint", mesh)
boundaries.set_all(0)
left = Left()
left.mark(boundaries, 4)
right = Right()
right.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
bottom = Bottom()
bottom.mark(boundaries, 1)

dss = Measure("ds")[boundaries]


gradx = as_vector((1.0,0.0))
gradt = as_vector((0.0,1.0))

(y,p) = TrialFunction(W)
(v,q) = TestFunction(W)
u = Function(V)
w = Function(W)
# Defining Exact Solutions.

# Defining the target control and target state expressions

ud1 = Expression("-sin(x[1]) - Kappa*(pi-x[1])", Kappa = Kappa)

ud2 = Expression("-sin(x[1]) + Kappa*(pi-x[1])", Kappa = Kappa)

yd = Expression("sin(x[0])*sin(x[1])- cos(x[0]) - cos(x[0])*(pi-x[1])")

# Defining Neumann Boundary Conditions

f = Expression("sin(x[1])*sin(x[0])+ cos(x[1])*sin(x[0])") 

# Defining Bilinear and Linear forms.


F = inner(grad(p), gradx) * inner(grad(q),gradx)*dx\
    + inner(grad(y), gradx) * inner(grad(v),gradx)*dx\
    + inner(grad(q),gradt) * p *dx\
    - inner(grad(v), gradt) * y * dx\
    - f * v * dx\
    - y * q * dx  + yd * q * dx\
    + p * q * dss(1) - (ud1 - Kappa*p) * v * dss(2)\
    + y * v * dss(3) - (ud2 - Kappa*p) * v * dss(4)

a = lhs(F)
L = rhs(F)

solve(a == L, w, bcs)

(y,p) = w.split()
u1 = ud1 - Kappa*p
u2 = ud2 - Kappa*p

L2error = sqrt(assemble(inner((u1-uex),(u1-uex))*dss(2))
		+ assemble(inner((u2-uex),(u2-uex))*dss(4)))

print(L2error)


