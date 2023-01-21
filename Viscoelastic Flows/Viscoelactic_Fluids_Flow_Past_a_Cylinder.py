# Contact:
# Birkan Tunc
# tuncbirkan@gmail.com

from dolfin import *
import time
from mshr import *
import ufl
import os
import shutil
from math import floor, ceil
import numpy as np

# Global parameters
parameters["form_compiler"]["cpp_optimize"] = True
parameters['linear_algebra_backend'] = "PETSc"
print('Linear algebra backend:',parameters['linear_algebra_backend'])

# Delete existing folders/files and generate new ones
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.splitext(__file__)[0]
myfile=("%s.results" % (fileName))
if os.path.exists(myfile):
   shutil.rmtree("%s/%s.results" % (dir_path,fileName), ignore_errors=True)
pfile = File("%s.results/p.pvd" % (fileName))
vfile = File("%s.results/v.pvd" % (fileName))
taufile = File("%s.results/tau.pvd" % (fileName))
mfile = File("%s.results/m.pvd" % (fileName))

# Tolerance
TOL=1E-8

# Geometric parameters
a = 3.188
H = 2.*a

# Material parameters
lamda = 0.794
beta = .41
mu_N = 1.
mu_P = mu_N/(1.-beta)-mu_N
mu_0 = mu_N + mu_P
De = .025
Vprsc = De*a/lamda
Newtonian = False
if Newtonian:
   De = 0.
   lamda = 0.
   Vprsc = 2./3.

# Generate mesh
channel = Rectangle(Point(-12*a, -H), Point(16*a, H))
cylinder = Circle(Point(0, 0), a, 70)
domain = channel - cylinder
mesh = generate_mesh(domain, 115)

# Refine mesh for zone_1
zone_1 = MeshFunction('bool', mesh, mesh.topology().dim() - 0)
zone_1.set_all(False)
for i in cells(mesh):
    if i.midpoint().x() > -3*a and i.midpoint().x() < 3*a:
        zone_1[i] = True
mesh = refine(mesh, zone_1)

# Refine mesh for zone_2
zone_2 = MeshFunction('bool', mesh, mesh.topology().dim() - 0)
zone_2.set_all(False)
for i in cells(mesh):
    if sqrt(i.midpoint().x()*i.midpoint().x()+i.midpoint().y()*i.midpoint().y())<1.4*a:
        zone_2[i] = True
mesh = refine(mesh, zone_2)
n = FacetNormal(mesh)

# Print mesh statistic and save mesh
print('Number of nodes: %d, Number of elements: %d'%(mesh.num_vertices(),mesh.num_cells()))
mfile << mesh

# Mixed element
P = FiniteElement("CG", triangle, 1)   # Pressure
V = VectorElement("CG", triangle, 2)   # Velocity
Tau = TensorElement("CG", triangle, 2) # Extra stress
E = MixedElement([P, V, Tau])

# Elements and space
W = FunctionSpace(mesh,E)
w = Function(W);
(p,v,tau) = split(w);
(p_,v_,tau_) = TestFunctions(W)
I = Identity(2)

# Define boundaries
def top(x, on_boundary):
   return on_boundary and x[1]>H-TOL
def bottom(x, on_boundary):
   return on_boundary and x[1]<-H+TOL
def inlet(x, on_boundary):
   return on_boundary and x[0]<-12*a+TOL
def outlet(x, on_boundary):
   return on_boundary and x[0]>16*a-TOL
def cylinder(x, on_boundary):
   return on_boundary and x[0]>-a-TOL and x[0]<a+TOL and x[1]>-a-TOL and x[1]<a+TOL

# Mark cylinder boundary
class wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]>-a-TOL and x[0]<a+TOL and x[1]>-a-TOL and x[1]<a+TOL
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
cylinderwall=wall()
cylinderwall.mark(boundary_markers, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define boundary conditions
bcs = list()
velocity_profile = Expression(('3./2.*Vprsc*(1. - x[1]*x[1]/(H*H))', '0.'), \
                Vprsc=Vprsc, H=H, degree=2)
bcs.append( DirichletBC(W.sub(1),Constant((0,0)),top) )
bcs.append( DirichletBC(W.sub(1),Constant((0,0)),bottom) )
bcs.append( DirichletBC(W.sub(1),Constant((0,0)),cylinder) )
bcs.append( DirichletBC(W.sub(1),velocity_profile,inlet) )
bcs.append( DirichletBC(W.sub(1),velocity_profile,outlet) )
bcs.append( DirichletBC(W.sub(0),Constant(0),inlet) )

# Functions
def L(v):
   return grad (v)
def D(v):
   return ( grad (v)+ grad (v).T) /2.0
def T(p,v,tau):
   D = (grad(v)+grad(v).T)/2.0
   return -p*I + 2.*mu_N*D + tau

# SUPG Coefficient
def eta(v):
   he=CellDiameter(mesh)
   nb=sqrt(dot(v,v))
   nb=conditional(le(nb,TOL), 1., nb)
   return he/nb

# Problem definition
w12 = div(v)*p_*dx
w13 = inner((-p*I+2.*mu_N*D(v)+tau),grad(v_))*dx
w16 = inner((tau+lamda*(dot(v,nabla_grad(tau))-dot(L(v),tau) \
   -dot(tau,L(v).T))-2.*mu_P*D(v)),tau_+dot(eta(v)*v,nabla_grad(tau_)))*dx
weak_form = w12 + w13 + w16
derivative_of_weak_form = derivative(weak_form,w)
my_problem = NonlinearVariationalProblem(weak_form,w,bcs,derivative_of_weak_form)
my_solver = NonlinearVariationalSolver(my_problem)

# Solver parameters
prm = my_solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['absolute_tolerance'] = 1.E-10
prm['newton_solver']['relative_tolerance'] = 1.E-9
prm['newton_solver']['maximum_iterations'] = 16

# Solve problem
increment = 0
while increment < 20:

   # Increment
   if Newtonian:
      increment = 20
   else:
      increment += 1

   # Update velocity based on Deborah number
   if not Newtonian:
      if increment == 1: De = 0.025
      elif increment == 2: De = 0.05
      elif increment == 3: De = 0.1
      else: De += 0.1
      Vprsc = De*a/lamda
      velocity_profile.Vprsc = Vprsc
   
   # Solve
   my_solver.solve()
   (p,v,tau) = w.split(True)
   
   # Save solution
   p.rename("p", "p"); pfile << (p , increment)
   v.rename("v", "v"); vfile << (v , increment)
   tau.rename("tau", "tau"); taufile << (tau , increment)

   # Calculate drag coefficient
   Fx = -assemble(dot(T(p,v,tau),n)[0]*ds(1))
   F = (4.*np.pi*(mu_0)*Vprsc)
   drag =  Fx / F
   print('Increment: %d, De: %.3f, Fx: %.4f, F: %.4f, Drag Coefficient: %.4f'%(increment, De, Fx, F, drag))
   resfile = open("Results_Flow_Past_a_Cylinder.txt", "a")
   resfile.write("De: %.8f, Drag Coefficient: %.8f\n"%(De,drag))
   resfile.close
