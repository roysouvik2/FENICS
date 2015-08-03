from dolfin import *
from math import *

k = 1.0
Kappa = 1.0

# Defining Mesh.

nx = 100
nt = 100

mesh = Rectangle(0,0,pi,pi, nx, nt)

# Defining Function Space.

V = FunctionSpace(mesh,"CG",1)

# Exact y
yex = Expression(" sin(x[1])*sin(x[0])")
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


# Defining the target control and target state expressions

ud = Expression("sin(x[0])*(sin(x[1])+cos(x[1]))+ Kappa*cos(x[0])*(pi-x[1])", Kappa = Kappa)

yd = Expression("sin(x[0])*sin(x[1])- cos(x[0]) - cos(x[0])*(pi-x[1])")

# Defining Neumann Boundary Conditions

g = Expression("sin(x[1])") 


L2error = 1.0
tol = 1e-9

# Starting with an initial value of u
umod = Function(V)
umod = interpolate(Constant(0.0),V)
u = Function(V)
# Setting up iterative procedure

while(L2error > tol):
	u.assign(umod)
	# Step 1:- Solving equation for y
	# Define Variational Problem
	y = TrialFunction(V)
	v = TestFunction(V)
	w = Function(V)
	F =  inner(grad(y), gradx) * inner(grad(v),gradx)*dx\
		+ inner(grad(y), gradt) * v * dx\
		- u * v * dx\
		- g * v * dss(2)\
    		- g * v * dss(4)
	
	bc = DirichletBC(V,noslip, Bottom())

	a = lhs(F)
	L = rhs(F)

	solve(a==L, w, bc)
	
	# Step 2:- Solving equation for p
	# Define Variational Problem
	p = TrialFunction(V)
	q = TestFunction(V)
	r = Function(V)

	F = inner(grad(p), gradx) * inner(grad(q),gradx)*dx\
		- inner(grad(p),gradt) * q *dx\
		- w * q * dx  + yd * q * dx

	bc1 = DirichletBC(V,noslip, Top())

	a = lhs(F)
	L = rhs(F)

	solve(a==L, r, bc1)
	
	# Control Step:- Checking if solution obtained is correct 
	
	umod = ud - Kappa*r
	umod = project(umod, V)
	L2error = sqrt(assemble(inner(u-umod,u-umod)*dx))
	print(L2error)

plot(u,interactive = 'True')