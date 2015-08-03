# This is a program to solve the Stokes type system for 
# estimating the motion of clouds.

from dolfin import *

# Create mesh and define function space.

mesh = UnitSquare(10,10)
V = VectorFunctionSpace(mesh,"CG", 1)
Q = FunctionSpace(mesh,"CG", 1)
W = V * Q

# Determining the exact velocity with which the image will be 
# moved by solving a stokes problem with some given boundary 
# conditions.

t = 0
T  = 100
dt = 0.1
#while t < T:
 # Boundaries

def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
     return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

 # Boundary condition for velocity

class Boundary(Expression):
    def eval(self, value, x):
        value[0] = x[1]
        value[1] = -x[0]
    def value_shape(self):
        return (2,)
noslip = Boundary()

bc0 = DirichletBC(W.sub(0), noslip, top_bottom)
bc1 = DirichletBC(W.sub(0), noslip, left)
bc2 = DirichletBC(W.sub(0), noslip, right)

 # Collect Boundary Conditions
bcs = [bc0, bc1, bc2]

alpha = 1.0


# Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
test = Function(W)

F = Constant((0.0,0.0))
n  = FacetNormal(mesh)

a = pow(alpha, 2.0)* inner(grad(u), grad(v))*dx \
    + div(v)*p*dx + q*div(u)*dx \
    + 1e-10*p*q*dx
L = inner(F, v)*dx

solve(a==L, test, bcs)
(ue, pe) = test.split()
plot(ue)
plot(pe)
interactive()

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

 # Boundaries

def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS



 # Define Variational Problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w = Function(W)


gradE = as_vector((Ex,Ey))


F = pow(alpha, 2.0) * inner(grad(u), grad(v))*dx \
          - div(v)*p*dx - q*div(u)*dx \
          +(inner(gradE,u))*(inner(gradE,v))*dx\
          +(Et*Ex*v[0] + Et*Ey*v[1])*dx+1e-10*p*q*dx

a = lhs(F)
L = rhs(F)

solve(a==L, w,bcs)
(vel, p) = w.split()
plot(vel)
plot(p)
interactive()
t = t+dt
 
