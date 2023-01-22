# Contact:
# Birkan Tunc
# tuncbirkan@gmail.com

from dolfin import *
from mshr import *
import random
import os
import shutil
import numpy as np
import time
from PIL import Image

# Global parameters
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 12
parameters['linear_algebra_backend'] = "PETSc"
print('Linear algebra backend:',parameters['linear_algebra_backend'])

# Delete existing folder and generate new files
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.splitext(__file__)[0]
myfile=("%s.results" % (fileName))
if os.path.exists(myfile):
   shutil.rmtree("%s/%s.results" % (dir_path,fileName), ignore_errors=True)
lfile = File("%s.results/l.pvd" % (fileName))
cfile = File("%s.results/c.pvd" % (fileName))
vfile = File("%s.results/v.pvd" % (fileName))
rfile = File("%s.results/r.pvd" % (fileName))
ufile = File("%s.results/u.pvd" % (fileName))
cvolfile = File("%s.results/cvol.pvd" % (fileName))

#################################### Read Images ####################################
# Read MRI and segment images
T2wS1 = Image.open('%s/levelset.png'%(dir_path)).load(); T2wS1_a = list()
meshS = Image.open('%s/mesh.png'%(dir_path)).load(); meshS_a = list()

# Convert intensity values to array
width, height = Image.open('%s/mesh.png'%(dir_path)).size
for i in range(height):
	for k in range(width):
           T2wS1_a.append(T2wS1[k, height-1-i])
           meshS_a.append(meshS[k, height-1-i])
           
#################################### Generate Submesh ####################################
# Create mesh
# image resolution is 1mm x 1mm x 1mm
mesh = RectangleMesh(Point(0, 0), Point(width, height), width-1, height-1)

# Define function spaces
CI = FunctionSpace(mesh, "Lagrange", 1)
dofmap = dof_to_vertex_map(CI)
m = Function(CI);
li1 = Function(CI);

# Assign image data to function
for i in range(CI.dim()):
   li1.vector()[i]=T2wS1_a[dofmap[i]]/255.
   m.vector()[i]=meshS_a[dofmap[i]]/255.

# Define subdomains
class outerbound(SubDomain):
    def inside(self, x, on_boundary):
         return (m(x)>0.5)
sub_domains=MeshFunction('size_t', mesh, mesh.topology().dim() - 0)
sub_domains.set_all(0)
obound=outerbound()
obound.mark(sub_domains,1)

# Generate sub mesh
smesh=SubMesh(mesh,sub_domains,1)

#################################### Initialize level set ####################################
def reinit(level, num_steps=5, itype=True):
    # Estimate the average mesh
    vol = assemble(Constant(1.0)*dx(smesh))
    diam = assemble(CellDiameter(smesh)*dx)/vol
    
    # Constants
    C1      = 0.5
    dtau    = Constant( C1*(diam**1.1) )
    epsilon = Constant( C1*(diam**0.9) )

    # Function space
    FS = FunctionSpace(smesh,"CG",1)
    VFS = VectorFunctionSpace(smesh,"CG",1)

    # functions setup
    phi = Function(FS);
    phi0 = Function(FS);
    w = TestFunction(FS)
    
    # initial values
    phi.assign(level)
    phi0.assign(level)
    
    # Unit normal vector (fixed-value based on initial LS)
    grad_n = project(grad(level),VFS)
    n = grad_n/(sqrt(dot(grad_n,grad_n)))
        
    for ii in range(num_steps):
       if itype: F = (phi-phi0)/dtau*w*dx + dtau*(-phi*(1.-phi)*dot(n,grad(w))*dx + epsilon*dot(grad(phi),grad(w))*dx) #initialize
       else: F = (phi-phi0)/dtau*w*dx + (-phi*(1.-phi)*dot(n,grad(w))*dx + epsilon*dot(n,grad(phi))* dot(n,grad(w))*dx) #reinitialize
       dF = derivative(F,phi)
       bcs=[]
       problem=NonlinearVariationalProblem(F,phi,bcs,dF)
       solver=NonlinearVariationalSolver(problem)

       # Solver parameters
       prm = solver.parameters
       prm['newton_solver']['maximum_iterations'] = 12
       prm['newton_solver']['error_on_nonconvergence'] = False
       solver.solve()
       phi0.assign(phi)

    return phi

#################################### Problem Definition ####################################
# Generic parameters
degree = 1
TOL = 1E-12
opt_count=0

# Create segment functions
CS = FunctionSpace(smesh, "Lagrange", degree)
cs1 = Function(CS);
ls1 = Function(CS);

# Initialize level set
ls1 = interpolate(li1,CS);

# Define mixed element
V = VectorElement("Lagrange", smesh.ufl_cell(), degree)  # Displacement
C = FiniteElement("Lagrange", smesh.ufl_cell(), degree)  # Concentration
Element = MixedElement([V, C, V])

def getres(write=False, dstep=55.*2., t_end=5.5, alpha=1.):
   
   # Material parameters
   kappa = .072
   rho = .075
   mu = 850. / 1E6
   K = 100. * mu
   nu = (3.*K-2.*mu)/2./(3.*K+mu)
   lamda = 2.*mu*nu/(1.-2.*nu)
   c1 = mu/2.0
   D1 = 2.0/K
   alphal = alpha*3.*K*1E6

   # Print parameters
   print("kappa: %.6f, rho: %.6f, alpha: %.6f, nu: %.6f\n"%(kappa, rho, alpha, nu))

   # Time step
   t=0.; dt = t_end/dstep; nsteps = int(t_end/dt);

   # Define function space
   W = FunctionSpace(smesh,Element)
   w = Function(W); w0 = Function(W); wn = Function(W);
   (v,c,r) = split(w); (v0,c0,r0) = split(w0);
   (v_,c_,r_) = TestFunctions(W)
   VL = VectorFunctionSpace(smesh, "Lagrange", degree)
   I=Identity(v.geometric_dimension())
   L = FunctionSpace(smesh, "Lagrange", degree)
   l = TrialFunction(L); l0 = Function(L); l_ = TestFunction(L);
   u = Function(VL); ci = Function(L)
   cvol = Function(L);
   
   # Initialize velocity
   assign(w0.sub(0),interpolate(Constant((0.,0.)),VL))
   assign(w.sub(0),interpolate(Constant((0.,0.)),VL))
   
   # Initialize level set
   assign(l0,reinit(ls1))
   
   # Initialize concentration
   cinit = Expression('sqrt((x[0]-60.)*(x[0]-60.) + (x[1]-175.)*(x[1]-175.))-r', degree=1, r=1.2)
   ci.assign(interpolate(cinit,L))
   for i in range(L.dim()):
       if ci.vector().get_local()[i] <= 0.: ci.vector()[i] = 1.
       else : ci.vector()[i] = 0.
   assign(w0.sub(1),ci)
   assign(w.sub(1),ci)

   # Initialize reference map
   reference = Expression(('x[0]','x[1]'), degree=degree)
   assign(w0.sub(2),interpolate(reference,VL))
   assign(w.sub(2),interpolate(reference,VL))

   alphae = Expression('alpha*i/10.', alpha=alpha, i=1., degree=degree)
   
   # Define boundaries
   def outer(x, on_boundary):
       return on_boundary

   # Stress
   def T(r,c):
      lamdac = pow(1.0+alphae*c,1./3.)
      F = inv(grad(r))
      Fg = lamdac*I
      Fe = F*inv(Fg)
      J = det(F)
      Jg = pow(lamdac,3)
      Je = det(Fe) / lamdac
      Ce = Fe.T*Fe
      Ce33 = 1./pow(lamdac,2)
      detCe = Ce33 * det(Ce)
      Ceinv = Ce33 / detCe * as_tensor([[Ce[1,1], -Ce[0,1]],[-Ce[1,0], Ce[0,0]]])
      I1e = tr(Ce) + Ce33
      Je23 = conditional(le(Je,TOL),1.,pow(Je,-2./3.))
      PK2e = 2.0/D1*Je*(Je-1.0)*Ceinv + Je23*2.0*c1*(I-I1e/3.0*Ceinv)
      PK2 = Jg*inv(Fg)*PK2e*inv(Fg).T
      return 1./J*F*PK2*F.T
      
   # Boundary conditions
   bcs=list()
   bcs.append(DirichletBC(W.sub(0), Constant((0.,0.)), outer))
   bcs.append(DirichletBC(W.sub(2), reference, outer))
   
   def R(c): 
       return conditional(le(c,0.),0.,1.)
   
   # Problem definition
   Fsol = inner(T(r,c), grad(v_))*dx \
      + (c-c0)/dt*c_*dx + dot(grad(c),v)*c_*dx + c*div(v)*c_*dx + kappa*dot(grad(c),grad(c_))*dx - rho*c*(1.-c)*R(c)*c_*dx \
      + inner((r-r0)/dt,r_)*dx + inner(grad(r)*v,r_)*dx
   Jsol = derivative(Fsol,w)
   problem=NonlinearVariationalProblem(Fsol,w,bcs,Jsol)
   solver=NonlinearVariationalSolver(problem)

   # Problem definition for level set
   theta = 0.5;
   FL = (l-l0)/dt*l_*dx + theta*(inner(v,grad(l))*l_*dx) + (1.-theta)*(inner(v0,grad(l0))*l_*dx)
   al = lhs(FL); Ll = rhs(FL)
   
   # Solver parameters
   prm = solver.parameters
   prm['nonlinear_solver'] = 'newton'
   prm['newton_solver']['linear_solver'] = 'lu'   
   prm['newton_solver']['absolute_tolerance'] = 1E-10
   prm['newton_solver']['relative_tolerance'] = 1E-9
   prm['newton_solver']['maximum_iterations'] = 16
         
   # Write initial values to files
   t=0.
   (v,c,r)=w.split(True);
   if write:
      v.rename("v", "v"); vfile << (v,t)
      c.rename("c", "c"); cfile << (c,t)
      r.rename("r", "r"); rfile << (r,t)
      u.rename("u", "u"); ufile << (u,t)
      l0.rename("l", "l"); lfile << (l0,t)
         
   # Solve problem
   iflag = 0
   objective = 0; objectivel=0; objectivec=0;
   while TOL < t_end-t:

      # Update time
      t+=dt

      # Print time step
      iflag += 1
      if abs(alpha-10.)<TOL:
         if iflag <=5: alphae.i = iflag*2
      else:
         alphae.i = 10.
      if write: print("\n *** Time: %.2f / %.2f *** Time step: %d / %d *** \n"%((t),(t_end),(iflag),(nsteps)))
      
      # Solve
      solver.solve()
      (v,c,r) = w.split(True);
      l = Function(L)
      solve(al == Ll, l)

      # Calculate tumor volume
      volfile = open("volume.txt", "a")
      for i in range(L.dim()):
         if c.vector().get_local()[i] >= 0.1: cvol.vector()[i] = 1.
         else: cvol.vector()[i] = 0.
      vol = assemble(cvol*dx);
      volfile.write("%.8f\n"%vol)
      volfile.close

      # Assign previous
      u.assign(project(u+v*dt,VL))
      w0.assign(w)
      assign(l,reinit(l,1,False))
      l0.assign(l)
      
      # Write results to files
      if write:
         v.rename("v", "v"); vfile << (v,t)
         c.rename("c", "c"); cfile << (c,t)
         r.rename("r", "r"); rfile << (r,t)
         u.rename("u", "u"); ufile << (u,t)
         l0.rename("l", "l"); lfile << (l0,t)
         cvol.rename("cvol", "cvol"); cvolfile << (cvol,t)
   
   return

getres(dstep=240., t_end=120., write=True, alpha=1.)
