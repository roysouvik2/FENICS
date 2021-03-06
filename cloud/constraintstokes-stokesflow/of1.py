"""
This is a program to solve the Stokes type system for 
estimating the motion of clouds with velocity coming from 
a stokes flow in a lid driven cavity.
"""
from dolfin import *
import math

# Create mesh and define function space.

mesh = UnitSquare(100,100)
V = VectorFunctionSpace(mesh,"CG", 2)
Q = FunctionSpace(mesh,"CG", 1)
W = V * Q

# Determining the exact velocity with which the image will be 
# moved by solving a stokes problem with some given boundary 
# conditions.


# Output file
ex_file = File("Exact.pvd")
out_file = File("velocity.pvd")

# Boundaries

def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def bottom(x, on_boundary): return x[1] < DOLFIN_EPS
def top(x, on_boundary):return x[1] > 1.0 - DOLFIN_EPS 

nstep= 50
a = 0.1
# Exact Velocity.
class vel1(Expression):
   def eval(self, value, x):
      r = pow((x[0]-0.5),2.0)+0.00000001 +pow((x[1]-0.5),2.0)
      theta = math.atan2((x[1]-0.5),(x[0]-0.5))
      value[0] = 0.25*(2*r-3*a + pow(a,3.0)/(r*r))*sin(2*theta)
      #value[0] = 1
ue1 = vel1()

class vel2(Expression):
   def eval(self, value, x):
      r = pow((x[0]-0.5),2.0)+0.00000001 +pow((x[1]-0.5),2.0)
      theta = math.atan2((x[1]-0.5),(x[0]-0.5))
      value[0] = 0.25*(4*r-3*a - pow(a,3.0)/(r*r))*sin(theta)*sin(theta)
      #value[0] = 1
ue2 = vel2()
      
ue = as_vector((ue1, ue2))
ue = project(ue,V)
ex_file << ue

L2norm = norm(ue, 'L2')
# Boundary Conditions.

noslip  = DirichletBC(W.sub(0), (1,0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS ")

lid  = DirichletBC(W.sub(0), (0,0), "x[1] > 1.0 - DOLFIN_EPS|| x[1] < DOLFIN_EPS")

pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [noslip, lid, pref]
conv=[]
# Image data and its derivatives
class Image(Expression):
    def eval(self, value, x):
        value[0] = exp(-50*(pow((x[0]-0.5),2.0) + \
                        pow((x[1]-0.5),2.0)))
E = Image()

class MyExpression0(Expression):
    def eval(self, value, x):
        value[0] = -2*50*(x[0]-0.5)*E(x)
Ex = MyExpression0()

class MyExpression1(Expression):
    def eval(self, value, x):
        value[0] = -2*50*(x[1]-0.5)*E(x)
Ey = MyExpression1()

Et = -(Ex*ue[0] + Ey*ue[0])
 
gradE = as_vector((Ex,Ey))


j = 0

while j < nstep:

      nu  = 0.001 + 0.005 * j**3
      print "nu = ", nu
    
      (u,p) = TrialFunctions(W)
      (v,q) = TestFunctions(W)
      sol = Function(W)

    
      # Solving Linear Variational Problem

      F = nu * inner(grad(u), grad(v))*dx \
          - div(v)*p*dx - q*div(u)*dx \
          +(inner(gradE,u))*(inner(gradE,v))*dx\
          +(Et*Ex*v[0] + Et*Ey*v[1])*dx
   
      a = lhs(F)
      L = rhs(F)

      solve(a == L, sol, bcs)
      (vel, p) = sol.split()
      out_file << vel
      L2error = errornorm(vel, ue, 'L2')
      print "L2error = ", L2error/L2norm
      conv.append([nu, L2error/L2norm])
      j += 1

print "---------------------------------------"
f = open('conv.dat','w')
for j in range(nstep):
   fmt='{0:14.6e} {1:14.6e}'
   print fmt.format(conv[j][0], conv[j][1])
   f.write(str(conv[j][0])+' '+str(conv[j][1])+'\n')
print "---------------------------------------"
f.close()


 
