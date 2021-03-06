from dolfin import *
import time

#set_log_level(PROGRESS)
parameters['form_compiler']['representation'] = 'quadrature'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True

center = Point(0.2, 0.2)
radius = 0.05

class Cylinder(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        return(on_boundary and (r < 2*radius + sqrt(DOLFIN_EPS)))
    def snap(self, x):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        if r <= radius:
            x[0] = center[0] + (radius / r)*(x[0] - center[0])
            x[1] = center[1] + (radius / r)*(x[1] - center[1])

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
Gamma = Cylinder()
Gamma.mark(boundary_parts, 1)

#refine the mesh and snap boundary
mesh = refine(mesh)
mesh.snap_boundary(Gamma, False)

# Define function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, P])

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, Walls())

vc = Expression(("-cos(atan2(x[0]-0.2,x[1]-0.2))","sin(atan2(x[0]-0.2,x[1]-0.2))"))
bc_cylinder = DirichletBC(W.sub(0), vc, Cylinder())

# Inflow boundary condition for velocity and temperature
v_in = Expression(("1.5 * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )","0.0"))
bc1 = DirichletBC(W.sub(0), v_in, Inflow())

# Collect boundary conditions
bcs = [bc_cylinder, bc0, bc1]

# Define unknown and test function(s)
(u, q) = TestFunctions(W)

n = FacetNormal(mesh)
I = Identity(V.cell().d)    # Identity tensor

# current time step
w = Function(W)
(v, p) = (as_vector((w[0], w[1])), w[2])
D = 0.5*(grad(v)+grad(v).T)
T = -p*I + 2.0*nu*D

# previous time step
w0 = Function(W)
(v0, p0) = (as_vector((w0[0], w0[1])), w0[2])
D0 = 0.5*(grad(v0)+grad(v0).T)
T0 = -p*I + 2.0*nu*D0

# Define variational forms without time derivative in previous time
F0_eq1 = (inner(T0, grad(u)))*dx + inner(grad(v0)*v0, u)*dx
F0_eq2 = 0*q*dx
F0 = F0_eq1 + F0_eq2

# variational form without time derivative in current time
F1_eq1 = (inner(T, grad(u)) + inner(grad(v)*v, u))*dx
F1_eq2 = q*div(v)*dx
F1 = F1_eq1 + F1_eq2

# combine variational forms with time derivative
#
#  dw/dt + F(t) = 0 is approximated as
#  (w-w0)/dt + (1-theta)*F(t0) + theta*F(t) = 0
#
F = idt*inner((v-v0),u)*dx + (1.0-theta)*F0 + theta*F1

# residual of strong Navier-Stokes
r = idt*(v-v0) + theta*grad(v)*v + (1.0-theta)*grad(v0)*v0 \
   - theta*div(T) - (1.0-theta)*div(T0)

# stabilization parameter
h = CellSize(mesh)
velocity = v0
vnorm = sqrt(dot(velocity, velocity))
tau = ( (2.0*idt)**2 + (2.0*vnorm/h)**2 + (4.0*nu/h**2)**2 )**(-0.5)

# add SUPG stabilization
#F += tau*inner(grad(u)*v, r)*dx

# add PSPG stabilization
#F += tau*inner(grad(q), r)*dx

# define Jacobian
J = derivative(F, w)

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# create variational problem and solver
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 20

# Time-stepping
t = dt
while t < t_end:

    print "t =", t

    # Compute
    begin("Solving ....")
    solver.solve()
    end()

    # Extract solutions:
    (v, p) = w.split()

    # Plot
    #plot(v)

    # Save to file
    ufile << v
    pfile << p

    # Move to next time step
    w0.assign(w)
    t += dt
