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

n  = FacetNormal(mesh)
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


a = pow(alpha, 2.0)* inner(grad(u), grad(v))*dx \
    - div(v)*p*dx + q*div(u)*dx \
    - (p*dot(v,n))*ds + 1e-10*p*q*dx
L = inner(F, v)*dx

solve(a==L, test)
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

class MyExpression2(Expression):
    def eval(self, value, x):
        value[0] = -Ex(x)*ue[0](x)-Ey(x)*ue[1](x)
Et = MyExpression2()

 # Boundaries

def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS



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

class MyExpression4(Expression):
    def eval(self, value, x):
        value[0] = Ex(x)
        value[1] = Ey(x)
    def value_shape(self):
        return (2,)
derE = MyExpression4()

a = pow(alpha, 2.0)* inner(grad(u), grad(v))*dx \
    - div(v)*p*dx + q*div(u)*dx \
    +(inner(derE,u))*(inner(derE,v))*dx\
    -(p*dot(v,n))*ds +1e-10*p*q*dx
L = inner(f, v)*dx

solve(a==L, w)
(vel, p) = w.split()
plot(vel)
plot(p)
interactive()
t = t+dt
 
