from dolfin import *
import time


ex_file = File("1Exactflow.pvd")
out_file = File("1velocity.pvd")
ex_vorfile = File("vorticity.pvd")
ex2_vorfile = File("exvorticity.pvd")
img_file = File("Image_cyl.pvd")

center = Point(0.2, 0.2)
radius = 0.05

class Cylinder(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        return(on_boundary and (r < 2*radius + sqrt(DOLFIN_EPS)))
    #def snap(self, x):
     #   r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
      #  if r <= radius:
       #     x[0] = center[0] + (radius / r)*(x[0] - center[0])
       #     x[1] = center[1] + (radius / r)*(x[1] - center[1])

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and ((x[1] < DOLFIN_EPS) or (x[1] > (0.41 - DOLFIN_EPS))))

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and (x[0] < DOLFIN_EPS))

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and (x[0] > (2.2 - DOLFIN_EPS)))

# Parameters
nu = Constant(0.001)
dt = 0.1
idt = Constant(1./dt)
t_end = 10.0
theta=0.5   # Crank-Nicholson timestepping

# Mesh
mesh = Mesh("bench_L1.xml")
boundary_parts = MeshFunction("uint", mesh, mesh.topology().dim()-1)
boundary_parts.set_all(0)
#Gamma = Cylinder()
#Gamma.mark(boundary_parts, 1)

#refine the mesh and snap boundary
#mesh = refine(mesh)
#mesh.snap_boundary(Gamma, False)

# Define function spaces (Taylor-Hood)
V1 = FunctionSpace(mesh, "CG", 2)
V = V1*V1
Q = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, Q])
Z = FunctionSpace(mesh, "DG", 0)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, Walls())

vc = Expression(("0.0","0.0"))
bc_cylinder = DirichletBC(W.sub(0), vc, Cylinder())

# Inflow boundary condition for velocity and temperature
v_in = Expression(("1.5 * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )","0.0"))
bc1 = DirichletBC(W.sub(0), v_in, Inflow())

# Collect boundary conditions
bcs = [bc_cylinder, bc0, bc1]

# Define Variational Problem

(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
u0 = interpolate(Constant((0.0,0.0)),V)
w = Function(W)
ue = Function(V)
p1 = Function(Q)


g = Constant((0.0,0.0))
eps = 1.0
tol  = 1.0E-7
iter = 0
maxiter = 100
F = nu*inner(grad(u), grad(v))*dx \
    -div(v)*p*dx - q*div(u)*dx\
+ inner(grad(u)*u0, v)*dx \
- inner(g,v)*dx
   
a = lhs(F)
L = rhs(F)
while eps > tol and iter < maxiter:
     iter += 1
     solve(a == L, w, bcs)
     (ue,p1) = w.split()
     eps = errornorm(u0, ue, 'L2')
     print "L2error =", eps
     print "Iterations = ", iter
     u0.assign(ue)



ex_file << ue

uvor = Function(Q)
uvor = project((Dx(ue[1],0)-Dx(ue[0],1)),Q)
ex2_vorfile << uvor

conv=[]
# Image data and its derivatives
class Image(Expression):
    def eval(self, value, x):
        value[0] = exp(-100*(pow((x[0]-0.4),2.0) + \
                        pow((x[1]-0.1),2.0)))+\
                   exp(-100*(pow((x[0]-0.4),2.0) + \
                        pow((x[1]-0.3),2.0)))
E = Image()
E1 = project(E,Q)
img_file << E1

class MyExpression0(Expression):
    def eval(self, value, x):
        value[0] = -2*100*(x[0]-0.4)*E(x)
Ex = MyExpression0()

class MyExpression1(Expression):
    def eval(self, value, x):
        value[0] = -2*100*(x[1]-0.1)*exp(-100*(pow((x[0]-0.4),2.0) + \
                        pow((x[1]-0.1),2.0))) +\
                   -2*100*(x[1]-0.3)*exp(-100*(pow((x[0]-0.4),2.0) + \
                        pow((x[1]-0.3),2.0))) 
                    
Ey = MyExpression1()

Et = -(Ex*ue[0] + Ey*ue[1])
 
gradE = as_vector((Ex,Ey))


L2norm = norm(ue, 'L2')
j = 0

nstep = 50
while j < nstep:

      nu  = 0.001 + 0.005*j**3
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

      #uvor = Function(Q)
      #uvor = project((Dx(vel[1],0)-Dx(vel[0],1)),Q)
      #ex_vorfile << uvor
      
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

