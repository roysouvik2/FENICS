from dolfin import *

mesh =  Interval(100, 0, pi)

# Create Function Space

Q = FunctionSpace(mesh, "CG", 2)


# Parameters
T = pi
nt = 50
dt = T/nt
t = 0
eps = 0.0001

# Time Stepping
time_y = TimeSeries("Forward_data")
time_p = TimeSeries("Backward_data")
time_u = TimeSeries("U")

g = Expression('-sin(t)', t=0)

ud = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=0)
yd = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)', t=0)


y0    = Function(Q)
y     = Function(Q)
u     = Function(Q)

pf   = Function(Q)
p    = Function(Q)


while t < T:
   ud.t = t
   u = project(ud, Q)
   time_u.store(u.vector(),t)
   t += dt

def J():
   # Solving the forward problem.

   v,w = TrialFunction(Q), TestFunction(Q)
   u0 = Function(Q)
   u1 = Function(Q)
   y0 = interpolate(Constant(0), Q)
   yavg= 0.5*(v + y0)
   t = 0
   time_y.store(y0.vector(),t)
   while t < T:
      g.t  = t + 0.5*dt
      time_u.retrieve(u0.vector(),t)
      time_u.retrieve(u1.vector(),t+dt)
      F = (1.0/dt)*(v - y0)*w*dx \
          + inner(grad(yavg), grad(w))*dx \
          - g*w*ds \
          - 0.5*(u0+u1)*w*dx
      a = lhs(F)
      L = rhs(F)
      solve(a==L, y)
      t += dt
      y0.assign(y)
      time_y.store(y.vector(),t)

def dJ():
   # Solving the backward problem.
   v, w = TrialFunction(Q), TestFunction(Q)
   pf   = interpolate(Constant(0), Q)
   pavg= 0.5*(v + pf)
   t = T
   time_p.store(pf.vector(), t)
   while t > 0:
      yd.t = t-0.5*dt
      time_y.retrieve(y.vector(), t)
      time_y.retrieve(y0.vector(),t-dt)
      F = -(1.0/dt)*(pf - v)*w*dx\
          + inner(grad(pavg),grad(w))*dx\
          - (0.5*(y0+y) - yd)*w*dx
      a = lhs(F)
      L = rhs(F)
      solve(a==L, p)
      pf.assign(p)
      time_p.store(p.vector(), t)
      t -= dt

gradnorm  = 1.0
tol = 1e-10
it  = 0

# Setting up iterative step using steepest descent method.
while gradnorm > tol and it < 2:
   J()
   dJ()
   # Steepest Descent Method
   gradnorm = 0
   t = 0
   while t <= T:
      ud.t = t
      time_p.retrieve(p.vector(),t)
      time_u.retrieve(u.vector(),t)
      gradnorm += assemble((u-ud+p)**2*dx)
      un = u - eps*(u-ud+p)
      un = project(un, Q)
      time_u.store(un.vector(),t) 
      t += dt
   gradnorm = sqrt( dt * gradnorm )
   it += 1
   print "Iter=", it, " grad norm =", gradnorm
