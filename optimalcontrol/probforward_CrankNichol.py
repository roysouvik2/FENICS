"""
Solving the forward problem by minimizing the functional 
J(y,u) given by:-

J(y,u) = 0.5*||y-yd||^2 + 0.5*||u-ud||^2
for (x,t) in Q = (0,pi) x (0,pi)

s.t
y_t - y_xx = u in Q
dy/dn = -sin(t) for x = 0 and x = pi
y(x,0) = 0

using discretization in time by Crank Nicholson method.

"""

from dolfin import *
from math import *
from numpy import *

mesh =  Interval(100, 0, pi)

Q = FunctionSpace(mesh, "CG", 2)

# Parameters
T = pi
nt = 10
dt = T/nt

g = Expression('-sin(t)', t=0)

ud = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=0)
yd = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)', t=0)

yinit = Constant(0.0)
y0    = Function(Q)
y1    = Function(Q)
y0    = project(yinit, Q)
u     = Function(Q)

v,w = TrialFunction(Q), TestFunction(Q)
yavg= 0.5*(v + y0)

# Solving the forward problem.
F = (1.0/dt)*(v - y0)*w*dx \
  + inner(grad(yavg), grad(w))*dx \
  - g*w*ds \
  - u*w*dx
a = lhs(F)
L = rhs(F)
A = assemble(a)

def forward():
   J = 0.0
   t = 0
   while t < T:
      g.t  = t + 0.5*dt
      ud.t = t + 0.5*dt
      u    = project(ud, Q)
      yd.t = t + 0.5*dt
      b = assemble(L)
      solve(A, y1.vector(), b)
      J += assemble( (0.5*(y0+y1)-yd)**2*dx + (u-ud)**2*dx )
      y0.assign(y1)
      t += dt
   return J*dt

J = forward()

print "J= ", J
	
