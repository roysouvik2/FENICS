from dolfin import *

mesh =  Interval(100, 0, pi)

# Create Function Space

Q = FunctionSpace(mesh, "CG", 1)


# Parameters
T = pi
nt = 5
dt = T/nt
t = 0
eps = 0

# Time Stepping
time_y = TimeSeries("Forward_data")
time_p = TimeSeries("Backward_data")
time_u = TimeSeries("U")

g = Expression('-sin(t)', t=0)

ud = Expression("sin(x[0])*(sin(t)+cos(t)) + cos(x[0])*(pi-t)", t=0)
yd = Expression('sin(x[0])*sin(t)-cos(x[0])- cos(x[0])*(pi-t)', t=0)
uex = Expression("sin(t)*sin(x[0])+cos(t)*sin(x[0])",t=0)
yex = Expression("sin(t)*sin(x[0])", t=0)
pex = Expression("cos(x[0])*(pi-t)",t=0)
y0   = Function(Q)
y    = Function(Q)
u    = Function(Q)
un   = Function(Q)
pf   = Function(Q)
p    = Function(Q)


while t <= T:
   uex.t = t
   u = project(uex, Q)
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
      yex.t = t
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
      gradnorm1 = assemble((y-yex)**2*dx)
      print "Yerr",gradnorm1
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
   gradnorm2 = 0.0
   gradnorm = 0.0
   while t > 0:
      yd.t = t-0.5*dt
      ud.t = t
      time_u.retrieve(u.vector(),t)
      time_y.retrieve(y.vector(), t)
      time_y.retrieve(y0.vector(),t-dt)
      F = -(1.0/dt)*(pf - v)*w*dx\
          + inner(grad(pavg),grad(w))*dx\
          - (0.5*(y+y0) - yd)*w*dx
      a = lhs(F)
      L = rhs(F)
      solve(a==L, p)
      t -= dt
      pf.assign(p)
      time_p.store(p.vector(), t)
   t = 0
   while t <= T:
      ud.t=t
      time_p.retrieve(p.vector(),t)
      time_u.retrieve(u.vector(),t)
      gradnorm2 = assemble((p+u-ud)**2*dx)
      gradnorm += assemble((p+u-ud)**2*dx)
      plot(u, interactive = True)
      print "perr=", gradnorm2
      t += dt
   gradnorm = sqrt( dt * gradnorm )
   return gradnorm

gradnorm = 1.0
it = 0
tol = 1e-3
while gradnorm > tol and it < 3:
    J()
    gradnorm  = dJ()
    print "Iter=", it, " grad norm =", gradnorm
    t = 0
    while t <= T:
       yex.t = t
       time_y.retrieve(y.vector(),t)
       ud.t = t
       time_p.retrieve(p.vector(),t)
       time_u.retrieve(u.vector(),t)
       u = project(u,Q)
       un.assign(u)
       #plot(un, interactive = True)
       time_u.store(un.vector(),t) 
       t += dt
    it += 1
   

"""
t = 0
while t <= T:
    yex.t = t
    uex.t = t
    time_y.retrieve(y.vector(),t)
    time_u.retrieve(u.vector(),t)
    gradnorm1 = assemble((y-yex)**2*dx)
    gradnorm2 = assemble((u-uex)**2*dx)
    print "Yerr",gradnorm1
    print "uerr",gradnorm2
    t += dt
"""
