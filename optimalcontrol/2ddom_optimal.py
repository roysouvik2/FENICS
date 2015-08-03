from dolfin import *

# Creating mesh.

nx = 100
ny = 100
nt = 5
mesh  = Box(0,0,0,pi,pi,pi,nx,ny,nt)

print(mesh.hmax())

# Defining Function Space

V = FunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Defining Boundaries

class Gamma1(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)
class Gamma2(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0],pi)
class Gamma3(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 0.0)
class Gamma4(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], pi)
class Gamma5(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[2], 0.0)
class Gamma6(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[2], pi)

noslip = Constant(0.0)

bc0 = DirichletBC(W.sub(0), noslip, Gamma5())
bc1 = DirichletBC(W.sub(1), noslip, Gamma6())
bcs = [bc0,bc1]

# Defining Boundary Measures

boundaries = FacetFunction("uint", mesh)
boundaries.set_all(0)
g1 = Gamma1()
g1.mark(boundaries, 1)
g2 = Gamma2()
g2.mark(boundaries, 2)
g3 = Gamma3()
g3.mark(boundaries, 3)
g4 = Gamma4()
g4.mark(boundaries, 4)
g5 = Gamma5()
g5.mark(boundaries, 5)
g6 = Gamma6()
g6.mark(boundaries, 6)


dss = Measure("ds")[boundaries]

gradx = as_vector((1.0,0.0,0.0))
grady = as_vector((0.0,1.0,0.0))
gradt = as_vector((0.0,0.0,1.0))

(y,p) = TrialFunction(W)
(v,q) = TestFunction(W)

u = Function(V)
w = Function(W)

# Defining the target control and state expressions

ud = Expression("sin(x[0])*sin(x[1]) + 2*sin(x[0])*sin(x[1])*sin(x[2])+ cos(x[0])*cos(x[1])*(pi-x[2])")

yd = Expression("sin(x[0])*sin(x[1])*sin(x[2]) - cos(x[0])*cos(x[1])-2*cos(x[0])*cos(x[1])*(pi-x[2])")

# Defining Neumann Boundary Conditions

g1 = Expression("-sin(x[0])*sin(x[2])")
g2 = Expression("-sin(x[1])*sin(x[2])")

yex = Expression("sin(x[0])*sin(x[1])*sin(x[2])")

# Defining Bilinear and Linear forms

F = inner(grad(p), gradx) * inner(grad(q),gradx)*dx\
    + inner(grad(p), grady) * inner(grad(q),grady)*dx\
    + inner(grad(y), gradx) * inner(grad(v),gradx)*dx\
    + inner(grad(y), grady) * inner(grad(v),grady)*dx\
    + inner(grad(q),gradt) * p *dx\
    - inner(grad(v), gradt) * y * dx\
    +  p * v * dx - ud * v * dx\
    - y * q * dx  + yd * q * dx\
    + p * q * dss(5) + y * v * dss(6)\
    - g1 * v * dss(3) - g1 * v * dss(4)\
    - g2 * v * dss(1) - g2 * v * dss(2)

a = lhs(F)
L = rhs(F)

solve(a == L, w, bcs)

(y,p) = w.split()
u = ud - p
L2error = errornorm(y,yex, 'L2')
print(L2error)



