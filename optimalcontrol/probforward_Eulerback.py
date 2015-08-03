"""
Solving the forward problem by minimizing the functional 
J(y,u) given by:-

J(y,u) = 0.5*||y-yd||^2 + 0.5*||u-ud||^2
for (x,t) in Q = (0,pi) x (0,pi)

s.t
y_t - y_xx = u in Q
dy/dn = -sin(t) for x = 0 and x = pi
y(x,0) = 0

using discretization in time by Backward Euler method.

"""

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
n = FacetNormal(mesh)
# Parameters
T = pi
nt = 10
dt = T/nt
t = dt


# Time Stepping
time_y = TimeSeries("Forward_data")
time_p = TimeSeries("Backward_data")
time_u = TimeSeries("u")
time_ud = TimeSeries("ud")
time_yd = TimeSeries("yd")
# Boundaries

class Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)
class Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], pi)


g0 = Expression('-sin(t)', t = t)

ud = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=t)

yd = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)',t = t)

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
	time_y.store(y.vector(),t)
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

# Solving the forward problem.

def forw(u,y,t,dt,g0):
	v,w = TrialFunction(Q), TestFunction(Q)
	t = 2*dt
	while t < T +dt:
		time_y.retrieve(y.vector(), t-dt)
		time_u.retrieve(u.vector(),t)
		F = w*v*dx + dt*(inner(grad(w),grad(v)))*dx\
			-w*y*dx - dt*u*w*dx - dt*g0*n*ds
		a = lhs(F)
		L = rhs(F)
		solve(a == L, y)
		time_y.store(y.vector(),t)
		t += dt
forw(u,y,t,dt,g0)
# Calculating the value of the functional J
t = dt
J = 0.0
while t < T+dt:
	time_y.retrieve(y.vector(),t)
	time_p.retrieve(p.vector(),t)
	J += 0.5*assemble(inner(u-ud,u-ud)*dx+\
		inner(y-yd,y-yd)*dx)
	t += dt
print "J= ", J
	