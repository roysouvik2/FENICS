from dolfin import *


np = 128

mesh = UnitSquare(np,np)
File("mesh.xml") << mesh


W = FunctionSpace(mesh, "DG", 1)

V = VectorFunctionSpace(mesh, "CG", 1)
U = W * V

Z = FunctionSpace(mesh,"CG", 1)

R = FunctionSpace(mesh,"DG",0)
 
u = TrialFunction(V)
v = TestFunction(V)



Rc = 0.1
Ro = 0.2
Uc = 1

A = -(Uc*Rc)/(pow(Ro,2)-pow(Rc,2))
A1 = Uc/Rc

class vortex(Expression):
     def eval(self, value, x):
		  value[0] = -2*sin(x)*sin(y)



vorin = vortex()


				  
def Boundary(x, on_boundary):
	return on_boundary

g = Constant((0.2,0.0))
f = Constant(0.0)

vbc = DirichletBC(V, g, Boundary)


# Parameter values 

dt = 0.01
T = 1
alpha = 1
beta = 1

# Functions
w0 = Function(W)
w1 = Function(W)
wob = Function(W)
u0 = Function(V)
u1 = Function(V)
umod = Function(V)

# Finding out vorticity and velocity at the first time step using 
# Backward Euler for vorticity.

# First we compute initial velocity using the velocity equation

inivel = inner(grad(u), grad(v))*dx -vorin*Dx(v[1],0)*dx + vorin*Dx(v[0],1)*dx+\
         pow(alpha,2.0)*div(u)*div(v)*dx  +\
         pow(beta,2.0)*(Dx(u[1],0)-Dx(u[0],1)-vorin)*(Dx(v[1],0)-Dx(v[0],1))*dx
 
a = lhs(inivel)
L = rhs(inivel)


solve(a == L, u0 , vbc)
uob = Function(Z)
uob = project(div(u0),Z)
diverr = errornorm(f, uob, 'L2')


uvor = Function(Z)
uvor = project((Dx(u0[1],0)-Dx(u0[0],1)-vorin),Z)
curlerr = errornorm(f, uvor,'L2')

print "divergence error =", diverr
print "Total error = ", diverr+curlerr

#plot(u0, interactive = 'True')

outfile = File("patVelocitypen.pvd")
outfile2 = File("patVorticitypen.pvd")
w0 = project(vorin, W)

outfile << u0
outfile2 << w0

# Now solving w1 with Backward Euler.

w2 = TrialFunction(W)
wt = TestFunction(W)


n   = FacetNormal(mesh)
un  = dot(u0,n)
unp = 0.5*(un+abs(un))
unm = 0.5*(un-abs(un))
H   = unp('+')*w2('+') + unm('+')*w2('-')


form = (1/dt)*(w2-w0)*wt*dx - w2*inner(u0, grad(wt))*dx + H*jump(wt)*dS +\
       unp*w2*wt*ds + unm*f*wt*ds
a = lhs(form)
L = rhs(form)

solve(a==L, w1)


outfile2 << w1

# Solving for second step velocity

inivel = inner(grad(u), grad(v))*dx -w1*Dx(v[1],0)*dx + w1*Dx(v[0],1)*dx+\
         pow(alpha,2.0)*div(u)*div(v)*dx +\
         pow(beta,2.0)*(Dx(u[1],0)-Dx(u[0],1)-w1)*(Dx(v[1],0)-Dx(v[0],1))*dx
 
a = lhs(inivel)
L = rhs(inivel)

solve(a == L, u1 , vbc)

outfile << u1

# Now calculating the vorticity and velocity for subsequent time steps.
umod = 2*u1-u0
un  = dot(umod,n)
unp = 0.5*(un+abs(un))
unm = 0.5*(un-abs(un))
H   = unp('+')*w2('+') + unm('+')*w2('-')

form  = (3.0/(2*dt))*(w2-(4.0/3.0)*w1+(1.0/3.0)*w0)*wt*dx -\
        w2*inner(umod, grad(wt))*dx + H*jump(wt)*dS + unp*w2*wt*ds + unm*f*wt*ds 

a = lhs(form)
L = rhs(form)

a1 = lhs(inivel)
L1 = rhs(inivel)

t = 2*dt

while t < T+dt:

     solve(a == L, wob)
     w0.assign(w1)
     w1.assign(wob)
     u0.assign(u1)
     solve(a1 == L1, u1 , vbc)
     
     # Printing error in divergence condition
     uob = Function(Z)
     uob = project(div(u1),Z)
    
     uvor = Function(Z)
     uvor = project((Dx(u1[1],0)-Dx(u1[0],1)-w1),Z)
     
     diverr = errornorm(f, uob, 'L2')
     curlerr = errornorm(f, uvor,'L2')
     z = TestFunction(R)
     

     errvec = assemble((Dx(u1[1],0)-Dx(u1[0],1)-w1)**2*z*dx)
     err = 0.0
     
     coor = mesh.coordinates()
     mesh.num_vertices() == len(errvec)
     #print "Number of Vertices = ", mesh.num_vertices()
     for i in range(mesh.num_vertices()):
            #print 'errvector[%d]= %g' % (i,errvec[i])
            err+=errvec[i]


     print "Time = ", t
     print "divergence error =", diverr
     print "curl error =", curlerr
     print "curl error by assembling =", sqrt(err)
     print "Total error = ", diverr+curlerr

     # Storing solutions
     outfile << u1
     outfile2 << w1
     t += dt

     
     


    
    



   










				
		  
		       



