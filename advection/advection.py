from dolfin import *

def boundary_value(n):
    if n < 10:
        return float(n)/10.0
    else:
        return 1.0

# Load mesh and subdomains
mesh = Mesh("../mesh.xml.gz")
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz");
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function
#velocity = Function(V, "../velocity.xml.gz");
velocity = Constant( (-1.0, 0.0) )

# Initialise source function and previous solution function
f  = Constant(0.0)
u0 = Function(Q)

# Parameters
T = 5.0
dt = 0.1
t = dt


# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Mid-point solution
u_mid = 0.5*(u0 + u)

# Residual
r = u-u0 + dt*(dot(velocity, grad(u_mid)) - c*div(grad(u_mid)) - f)

# Galerkin variational problem
F = v*(u-u0)*dx + dt*(grad(v)*dot(velocity, grad(u_mid))*dx 

# Create bilinear and linear forms
#a = lhs(F)
#L = rhs(F)

# Set up boundary condition
g = Constant( boundary_value(0) )
bc = DirichletBC(Q, g, sub_domains, 1)

# Assemble matrix
A = assemble(a)
bc.apply(A)

# Create linear solver and factorize matrix
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
out_file = File("temperature.pvd")

# Set intial condition
u = u0

# Time-stepping
while t < T:

    # Assemble vector and apply boundary conditions
    b = assemble(L)
    bc.apply(b)

    # Solve the linear system (re-use the already factorized matrix A)
    solver.solve(u.vector(), b)

    # Copy solution from previous interval
    u0 = u

    # Plot solution
    plot(u)

    # Save the solution to file
    out_file << (u, t)

    # Move to next interval and adjust boundary condition
    t += dt
    g.assign( boundary_value( int(t/dt) ) )

# Hold plot
#interactive()

