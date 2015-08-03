# This is a program to solve the Stokes type system for 
# estimating the motion of clouds.

from dolfin import *

# Create mesh and define function space.

mesh = UnitSquare(10,10)
V1= FunctionSpace(mesh,"CG", 2)
V = VectorFunctionSpace(mesh,"CG", 2)
Q = FunctionSpace(mesh,"CG", 1)
W = V * Q
n = FacetNormal(mesh)

# Determining the exact velocity with which the image will be 
# moved by solving a stokes problem with some given boundary 
# conditions.

# Boundaries
def bdry(x, on_boundary): return on_boundary

def bottom(x, on_boundary): return x[1] < DOLFIN_EPS and on_boundary
def top(x, on_boundary): return x[1] > 1 - DOLFIN_EPS and on_boundary
def left(x, on_boundary): return x[0] < DOLFIN_EPS and on_boundary
def right(x, on_boundary): return x[0] > 1-DOLFIN_EPS and on_boundary

 # Defining the normal as a function on the whole space.
 # Boundary condition is specified as the normal.
 
bc_bot = DirichletBC(V, ("0","-1"), bottom)
bc_top = DirichletBC(V, ("0","1"), top)
bc_left = DirichletBC(V, ("-1","0"), left)
bc_right = DirichletBC(V, ("+1","0"), right)
bcn = [bc_bot, bc_top, bc_left, bc_right]
 
U = TrialFunction(V)
R = TestFunction(V)
eta = Function(V)
b = inner(grad(U), grad(R))*dx 
F = Constant((0.0,0.0))
m = inner(F, R)*dx

solve(b==m, eta, bcn)
#plot(eta)
interactive()

unit = Function(V1)
unit = interpolate(Constant(1), V1)

t = 0
dt = 0.01
T  = dt

# Output file
out_file = File("velocity.pvd")

# Boundary condition for velocity
class Boundary(Expression):
    def eval(self, value, x):
        value[0] = x[1]*t + 1.0
        value[1] = -x[0]*t + 1.0
    def value_shape(self):
        return (2,)
noslip = Boundary()

bcs = DirichletBC(W.sub(0), noslip, bdry)

alpha = 1.0

# Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
test = Function(W)

a = pow(alpha, 2.0)* inner(grad(u), grad(v))*dx \
    - div(v)*p*dx - q*div(u)*dx \
    + 1e-10*p*q*dx
L = inner(F, v)*dx

solve(a==L, test, bcs)
(ue, pe) = test.split()
#plot(ue)
# plot(pe)
#interactive()
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

Et = -(Ex*test[0] + Ey*test[1])

L2error = 1.0;
tol = 1e-9;
K = 1.0

# Starting of with an initial value of g assuming it to be 0.
#gmod = Expression(("0","0"))
g = Function(V)
g = ue
g = project(g, V)
plot(g)

# Setting up the iterative procedure.
while(L2error >tol):

   # Step 1:- Solving the Stokes problem with g = 0 initially.
   # Boundary condition for velocity
   bcs = DirichletBC(W.sub(0), g, bdry)

   # Define Variational Problem
   (u, p) = TrialFunctions(W)
   (v, q) = TestFunctions(W)
   w = Function(W)

   gradE = as_vector((Ex,Ey))

   a = pow(alpha, 2.0)* inner(grad(u), grad(v))*dx \
    - div(v)*p*dx - q*div(u)*dx \
    +(inner(gradE,u))*(inner(gradE,v))*dx\
    + 1e-10*p*q*dx
   L = -(Et*Ex*v[0] + Et*Ey*v[1])*dx

   solve(a==L, w, bcs)
   (vel, p) = w.split(True)
  #plot(vel)
   plot(p)
   interactive()
   print "Done"

   # Step 2:- Now determining the value of lambda.
   lam = Function(V)
   lam =  p*eta - pow(alpha,2.0)*(grad(vel)*eta)
   
   #mu = assemble(unit*ds)
   mu = assemble(inner(eta,eta)*ds)
   print "mu =", mu
   # Step 3:- Determining the new value of g
   nu = assemble(inner(lam,eta)*ds)
   print "nu=", nu

   gmod = (1.0/K) * (lam-(mu/nu)*eta)
   gmod = project(gmod, V)
   plot(gmod)
   L2error = assemble(inner(g-gmod,g-gmod)*ds)
   
   print "L2error =", L2error
   g.vector()[:] = gmod.vector()
   File("u.pvd") << vel
   File("gmod.pvd") << gmod
   File("g.pvd") << g
  
# plot(vel)
# Save the solution to file
out_file << vel

# plot(p)
#interactive()
 
