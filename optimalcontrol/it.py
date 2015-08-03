from dolfin import *

mesh =  Interval(100, 0, pi)

# Create Function Space

Q = FunctionSpace(mesh, "CG", 1)
u    = Function(Q)

# Parameters
T = pi
nt = 5
dt = T/nt
t = 0


# Time Stepping
time_y = TimeSeries("Forward_data")
time_p = TimeSeries("Backward_data")
time_u = TimeSeries("U")

uex = Expression("sin(t)*sin(x[0])+cos(t)*sin(x[0])",t=0)

while t <= T:
   uex.t = t
   u = project(uex, Q)
   time_u.store(u.vector(),t)
   t += dt
it = 0
while it < 3:
    t = 0
    while t <= T:
       time_u.retrieve(u.vector(),t)
       time_u.store(u.vector(),t) 
       print "Iter=", it
       plot(u, interactive = True)
       t += dt
    it += 1
