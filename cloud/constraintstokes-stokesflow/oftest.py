"""
This is a program to solve the Stokes type system for 
estimating the motion of clouds with velocity coming from 
a stokes flow in a lid driven cavity.
"""
from dolfin import *


# Create mesh and define function space.

#mesh = UnitSquare(100,100)
mesh = Mesh("mesh.xml")
sub_domains = MeshFunction("uint", mesh, "subdomains.xml");
plot(mesh)
V = VectorFunctionSpace(mesh,"CG", 2)
Q = FunctionSpace(mesh,"CG", 1)
W = V * Q

# Determining the exact velocity with which the image will be 
# moved by solving a stokes problem with some given boundary 
# conditions.


# Output file
ex_file = File("Exacttest.pvd")
out_file = File("velocitytest.pvd")


def boundary_value(n):
    if n < 10:
        return float(n)/10.0
    else:
        return 1.0

# Set up boundary condition
g = Constant((boundary_value(0),0))
vbc = DirichletBC(V, g, sub_domains, 1)


pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [vbc, pref]


alpha = 1


# Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
test = Function(W)

F = Constant((1.0,100.0))


a = alpha * inner(grad(u), grad(v))*dx \
    - div(v)*p*dx - q*div(u)*dx \
    + 1e-10*p*q*dx
L = inner(F, v)*dx

solve(a==L, test, bcs)

(ue, pe) = test.split()
ex_file << ue

nstep= 50

L2norm = norm(ue, 'L2')


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

Et = -(Ex*ue[0] + Ey*ue[1])
 
gradE = as_vector((Ex,Ey))


L2norm = norm(ue, 'L2')
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


 
