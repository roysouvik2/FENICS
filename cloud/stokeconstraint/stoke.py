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
#interactive()

unit = Function(V1)
unit = interpolate(Constant(1), V1)


# Output file
out_file = File("velocity.pvd")

# Boundary Conditions.

noslip  = DirichletBC(W.sub(0), (0,0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS")

lid  = DirichletBC(W.sub(0), (1,0), "x[1] > 1.0 - DOLFIN_EPS")

pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [noslip, lid, pref]

mu = 1.0/1000

# Exact Velocity

# Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
test = Function(W)

a = mu * inner(grad(u), grad(v))*dx \
    - div(v)*p*dx - q*div(u)*dx 
L = inner(F, v)*dx

solve(a==L, test, bcs)
(ue, pe) = test.split()

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
tol = 1e-8;
K = 1.0

# Starting of with an initial value of g assuming it to be 0.
#gmod = Expression(("0","0"))
g = Function(V)
g = ue
g = project(g, V)
#plot(g)

# Setting up the iterative procedure.
while(L2error >tol):

   # Step 1:- Solving the Stokes problem for u  with g = 0 initially.
   # Boundary condition for velocity

   bc = DirichletBC(W.sub(0), g, bdry)
   pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

   bcs = [bc, pref]

   # Define Variational Problem
   (u, p) = TrialFunctions(W)
   (u1, p1) = TestFunctions(W)
   w = Function(W)

   F = mu * inner(grad(u), grad(u1))*dx \
    - div(u1)*p*dx - p1*div(u)*dx 

   a = lhs(F)
   L = rhs(F)

   solve(a==L, w, bcs)
   (uvel, p) = w.split(True)
  
   print "Step 1 done."

   # Step 2:- Solving the stokes system for v with zero boundary conditions

   bc = DirichletBC(W.sub(0), (0,0), bdry)
   pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")
   
   bcs = [bc, pref]

   # Define Variational Problem
   (v, q) = TrialFunctions(W)
   (v1, q1) = TestFunctions(W)
   w = Function(W)

   gradE = as_vector((Ex,Ey))

   F = mu * inner(grad(v), grad(v1))*dx \
    - div(v1)*q*dx - q1*div(v)*dx \
    +(inner(gradE,uvel))*(inner(gradE,v1))*dx\
    +(Et*Ex*v1[0] + Et*Ey*v1[1])*dx
  
   a = lhs(F)
   L = rhs(F)

   solve(a==L, w, bcs)

   (vvel,q) = w.split()

   print "Step 2 done."

   # Step 3:- Now determining the value of lambda.
   lam = Function(V)
   lam =  q*eta - mu *(grad(vvel)*eta)
   
   print "Step 2 done."
   # Step 4:- Determining the new value of g

   zeta = assemble(inner(eta,eta)*ds)
   nu = assemble(inner(lam,eta)*ds)
   

   gmod = (1.0/K) * (lam-(zeta/nu)*eta)
   gmod = project(gmod, V)

   print "Step 4 done."

   
   L2error = assemble(inner(g-gmod,g-gmod)*ds)
   
   print "L2error =", L2error
   g.vector()[:] = gmod.vector()
   File("u.pvd") << uvel
   File("gmod.pvd") << gmod
   File("g.pvd") << g
   

# Save the solution to file
out_file << vel


 
