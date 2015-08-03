"""
Solving Navier Stokes constraint flow using test case as Stokes flow

"""

from dolfin import *
from math import *

# Create mesh and define function space.

mesh = Mesh("mesh.xml")

# Output file
out_file = File("velocity.pvd")


# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

we = Function(W)
File("exact.xml") >> we.vector()
ue = Function(V)
pe = Function(Q)
(ue,pe) = we.split()

nstep= 1

# Define Variational Problem

(v,q) = TestFunctions(W)
w = Function(W)
(u,p) = TrialFunctions(W)
u0 = interpolate(Constant((0.0,0.0)),V)
# Boundary Conditions.

noslip  = DirichletBC(W.sub(0), (0,0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS")

lid  = DirichletBC(W.sub(0), (1,0), "x[1] > 1.0 - DOLFIN_EPS")

pref = DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

bcs = [noslip, lid, pref]

conv=[]

for j in range(nstep):
      u0 = interpolate(Constant((0.0,0.0)),V)
      nu  = 0.001
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

      # Solving Linear Variational Problem

      eps = 1.0
      tol  = 1.0E-7
      iter = 0
      maxiter = 1000
      F = nu*inner(grad(u), grad(v))*dx \
          +(inner(gradE,u))*(inner(gradE,v))*dx\
          +(Et*Ex*v[0] + Et*Ey*v[1])*dx\
          -div(v)*p*dx - q*div(u)*dx\
          + inner(grad(u)*u0, v)*dx 
      a = lhs(F)
      L = rhs(F)
      while eps > tol and iter < maxiter:
            iter += 1
            solve(a == L, w, bcs)
            (u1,p1) = w.split()
            eps = errornorm(u0, u1, 'L2')
            print "error =", eps
            print "Iterations = ", iter
            u0.assign(u1)
      out_file << u1
      #plot(u1, interactive = 'True')
      L2error = errornorm(u1, ue, 'L2')
      print "L2error =", L2error
      """
      dw = TrialFunction(W)       
      dF = derivative(F, w, dw)
      nsproblem = NonlinearVariationalProblem(F, w, bcs, dF)
      solver = NonlinearVariationalSolver(nsproblem)
      
      prm = solver.parameters
      prm['newton_solver']['absolute_tolerance'] = 1E-6
      prm['newton_solver']['relative_tolerance'] = 1E-5
      prm['newton_solver']['maximum_iterations'] = 100
      prm['newton_solver']['relaxation_parameter'] = 1.0
      
      solver.solve()
      (u,p) = w.split()
      L2error = errornorm(u, ue, 'L2')
      print "L2error =", L2error
        
      """



