from dolfin import *
from math import *
from numpy import *

mesh =  Interval(100, 0, pi)

h = CellSize(mesh)
n = FacetNormal(mesh)
# Create Function Space

Q = FunctionSpace(mesh, "CG", 2)
y1 = Function(Q)
y2 = Function(Q)
u = Function(Q)
un = Function(Q)

# Parameters
T = pi
nt = 10
dt = T/nt
t = dt


# Time Stepping
time_series = TimeSeries("Forward_data")
time_p = TimeSeries("Backward_data")
time_u = TimeSeries("U")
time_un = TimeSeries("Un")

# Boundaries

class Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)
class Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], pi)
boundaries = FacetFunction("uint", mesh)
boundaries.set_all(0)
left = Left()
left.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 2)
dss = Measure("ds")[boundaries]

g0 = Expression('-sin(t)', t = t)
g1 = Expression('-sin(t+dt)', t = t, dt = dt)

# Initialise source function and previous solution function
while t < T+dt:
	u = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=t)
	u = project(u, Q)
	time_u.store(u.vector(),t) 
	t += dt

yinit = Constant(0.0)
y0 = Function(Q)
y = interpolate(yinit, Q)
t = dt
while t < T+dt:
	y = project(y, Q)
	time_series.store(y.vector(),t)
	t += dt

pfinal = Constant(0.0)
pf = Function(Q)
pf = interpolate(pfinal, Q)
p = Function(Q)

t = dt
while t < T+dt:
	p = project(p, Q)
	time_p.store(p.vector(),t)
	t += dt

# Test and Trial Functions

v, w = TrialFunction(Q), TestFunction(Q)


# Output file
out_file = File("state.pvd","compressed")

L2error = 10
tol = 1e-5
while L2error > tol:
	t = 2*dt
	while t < T+dt:
		time_series.retrieve(y.vector(),t-dt)
		time_u.retrieve(u.vector(),t)
		F = w*v*dx + (dt)*(inner(grad(w), grad(v)))*dx\
		-w*y*dx \
		- dt*u*w*dx - dt*g0*ds
		a = lhs(F)
		L = rhs(F)
		solve(a==L, y)	
		time_series.store(y.vector(),t)
		t += dt
	yd0 = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)',t = t+dt)	
	yd1 = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)',t=t)
	

# Test and Trial Functions
	v, w = TrialFunction(Q), TestFunction(Q)
# Galerkin Variational Formulation
	t = T-dt
	while t > 0:
		time_series.retrieve(y.vector(),t)
		time_p.retrieve(p.vector(),t+dt)
		F = w*v*dx + (dt/2.0)*(inner(grad(w), grad(v)))*dx\
		- w*p*dx + (dt/2.0)*(inner(grad(w), grad(p)))*dx\
	    	+ (dt/2.0)* ((yd0-y))*w*dx
		time_series.retrieve(y.vector(),t)
		F +=  + (dt/2.0)* ((yd1-y))*w*dx
	# Creating Bilinear and Linear Forms.
		a = lhs(F)
		L = rhs(F)
		solve(a==L, p)
		time_p.store(p.vector(), t)
		pf.assign(p)
		t -= dt
	t = dt
	while t < T+dt:
		time_series.retrieve(p.vector(),t)
		p = interpolate(p,Q)
		ud = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=t)
		un = ud - p
		un = project(un, Q)
		time_un.store(un.vector(),t) 
		t += dt
		time_u.retrieve(u.vector(),t)
		L2error += assemble(inner(u-un,u-un)*dx)
	L2error = sqrt(L2error)
	print(L2error)
	t = dt
	while t < T+dt:
		time_un.retrieve(un.vector(),t)
		u.vector()[:] = un.vector()
		time_u.store(u.vector(),t)
		t += dt
