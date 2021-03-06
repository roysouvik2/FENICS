# This is a program to solve the Stokes type system for 
# estimating the motion of clouds with velocity coming from 
# a stokes flow in a lid driven cavity.

from dolfin import *

# Create mesh and define function space.

mesh = UnitSquare(50,50)
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q
# Determining the exact velocity with which the image will be 
# moved by solving a stokes problem with some given boundary 
# conditions.

Re = 1000
nu = 1.0/Re
alpha = 1.0/ 50
# Output file
out_file = File("velocity.pvd")

# Define test functions
(v,q) = TestFunctions(W)

# Define trial functions
w = Function(W)
(u,p) = (as_vector((w[0], w[1])), w[2])

noslip  = DirichletBC(W.sub(0), (0,0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS")

lid  = DirichletBC(W.sub(0), (1,0), "x[1] > 1.0 - DOLFIN_EPS")

pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [noslip, lid, pref]

# Tentative velocity step
F =   inner(grad(u)*u, v)*dx \
 + nu*inner(grad(u), grad(v))*dx \
 - div(v)*p*dx \
 - q*div(u)*dx

dw = TrialFunction(W)
dF = derivative(F, w, dw)
nsproblem = NonlinearVariationalProblem(F, w, bcs, dF)
solver = NonlinearVariationalSolver(nsproblem)
solver.solve()
(ue,pe) = w.split()

plot(ue)
 # plot(pe)


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

Et = -(Ex*w[0] + Ey*w[1])
gradE = as_vector((Ex,Ey))

 # Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w = Function(W)

class MyExpression3(Expression):
     def eval(self, value, x):
         value[0] = -Et(x)*Ex(x)
         value[1] = -Et(x)*Ey(x)
     def value_shape(self):
        return (2,)
f = MyExpression3()

a = alpha*inner(grad(u), grad(v))*dx \
      - div(v)*p*dx - q*div(u)*dx \
      +(inner(gradE,u))*(inner(gradE,v))*dx\
    
L = -(Et*Ex*v[0] + Et*Ey*v[1])*dx


solve(a==L, w, bcs)
(vel, p) = w.split()
plot(vel)
 #interactive()
 # Save the solution to file
out_file << vel
L2error = errornorm(ue,vel,'L2')
L2norm = norm(ue, 'L2')
print(L2error)
print(L2norm)
print(sqrt(L2error)/sqrt(L2norm))
 # plot(p)
interactive()
 
 

