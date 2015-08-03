from dolfin import *
from math import *

k = 1.0
Kappa = 1.0

# Defining Mesh.

nx = 100
nt = 100

mesh = Rectangle(0,0,pi,pi, nx, nt)

h = CellSize(mesh)
print(mesh.hmax())

# Defining Function Space.


V = FunctionSpace(mesh,"CG",2)
Q = FunctionSpace(mesh,"CG",2)
W = V * Q

# Exact Solutions.

yex = Expression(" sin(x[1])*sin(x[0])")
uex = Expression("sin(x[1])*sin(x[0])+ cos(x[1])*sin(x[0])")

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

# Defining the target control and target state expressions

ud = Expression("sin(x[0])*(sin(x[1])+cos(x[1]))+ Kappa*cos(x[0])*(pi-x[1])", Kappa = Kappa)

yd = Expression("sin(x[0])*sin(x[1])- cos(x[0]) - cos(x[0])*(pi-x[1])")

# Defining Neumann Boundary Conditions

g = Expression("-sin(x[1])") 

# Defining Bilinear and Linear forms.


F = inner(grad(p), gradx) * inner(grad(q),gradx)*dx\
    + inner(grad(y), gradx) * inner(grad(v),gradx)*dx\
    + inner(grad(q),gradt) * p *dx\
    - inner(grad(v), gradt) * y * dx\
    + Kappa * p * v * dx - ud * v * dx\
    - y * q * dx  + yd * q * dx\
    + p * q * dss(1) - g * v * dss(2)\
    + y * v * dss(3) - g * v * dss(4)

a = lhs(F)
L = rhs(F)

solve(a == L, w, bcs)

(y,p) = w.split()
u = ud - Kappa*p
u = project(u, V)
ud = project(ud, V)
#plot(u, interactive = 'True')
yL2error = errornorm(y,yex,'L2')
print(yL2error)
uL2error = errornorm(u,uex,'L2')
print(uL2error)
L2error = errornorm(uex,ud,'L2')
print(L2error)
#plot(ud, interactive = 'True')
#plot(yex, interactive = 'True')
#plot(yd, interactive = 'True')
plot(y, interactive = 'True')
#plot(p, interactive = 'True')
    


