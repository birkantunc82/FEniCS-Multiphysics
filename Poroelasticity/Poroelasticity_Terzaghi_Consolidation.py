# Contact:
# Birkan Tunc
# tuncbirkan@gmail.com

from dolfin import *
from mshr import *
import os
import shutil
import numpy as np
import math
from numpy import zeros

# Global parameters
set_log_level(30)
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters['linear_algebra_backend'] = "PETSc"
print('Linear algebra backend:',parameters['linear_algebra_backend'])

# Delete existing folders/files and generate new files
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.splitext(__file__)[0]
myfile=("%s.results" % (fileName))
if os.path.exists(myfile):
   shutil.rmtree("%s/%s.results" % (dir_path,fileName), ignore_errors=True)
ufile = File("%s.results/u.pvd" % (fileName))
pfile = File("%s.results/p.pvd" % (fileName))

# Generate mesh
h = 1.; L = 20.*h
mesh = RectangleMesh(Point(-h/2.,0),Point(h/2.,L),1,20)
n = FacetNormal(mesh)

# Material parameters
c = 1.68E6
Ku = 16200.
Kd = 8000.
G = 6000.
alpha = 0.78
kappa = 1.9E-7
mu = 1E-9
Sigma0 = 1.
p0 = 3.*(Ku-Kd)*Sigma0/(alpha*(3.*Ku+4.*G))
w_0 = 3.*Sigma0*L / (3.*Ku+4.*G)
w_inf = 3.*Sigma0*L / (3.*Kd+4.*G)
w_dif = w_inf-w_0
TOL = 1E-10

# Time step
tau = [0.0001, 0.001, 0.01, 0.04, 0.1, 0.4, 1., 1.5]
t = 0.
index = 7
t_end = tau[index]*L**2/c
dt = t_end/100.
nsteps=int(t_end/dt);

# Function space
degree=2; pdegree=1;
U = VectorElement("Lagrange", mesh.ufl_cell(), degree)   # Displacement
P = FiniteElement("Lagrange", mesh.ufl_cell(), pdegree)  # Pressure
Element = MixedElement([U, P])
W = FunctionSpace(mesh,Element)
wi1 = Function(W); wi = Function(W)
(ui1, pi1) = split(wi1); (ui, pi) = split(wi)
(u_, p_) = TestFunctions(W)
I=Identity(ui1.geometric_dimension())

# Boundaries
def top(x, on_boundary):
   return on_boundary and x[1] > L-TOL
def bottom(x, on_boundary):
   return on_boundary and x[1] < TOL
def left(x, on_boundary):
   return on_boundary and x[0] < -h/2.+TOL
def right(x, on_boundary):
   return on_boundary and x[0] > h/2.-TOL

# Boundary conditions
bcs=list()
bcs.append(DirichletBC(W.sub(0), Constant((0.,0.)), bottom))
bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.), left))
bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.), right))
bcs.append(DirichletBC(W.sub(1), Constant(0.), top))
   
# Mark top surface
class wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > L-TOL
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
topwall=wall()
topwall.mark(boundary_markers, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Functions
def eps(u):
    return sym(grad(u))

def T(u, p):
    return 2.*G*eps(u) + (Kd-2./3.*G)*div(u)*I - alpha*p*I

# Problem definition
F = inner(T(ui1, pi1), eps(u_))*dx \
   + (kappa/mu)*inner(grad(pi1),grad(p_))*dx + alpha*(div(ui1)-div(ui))/dt*p_*dx + (alpha**2./(Ku-Kd))*(pi1-pi)/dt*p_*dx \
   + Sigma0*dot(n,u_)*ds(1)
J = derivative(F,wi1)
problem=NonlinearVariationalProblem(F,wi1,bcs,J)
solver=NonlinearVariationalSolver(problem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['absolute_tolerance'] = 1.E-10
prm['newton_solver']['relative_tolerance'] = 1.E-9
prm['newton_solver']['maximum_iterations'] = 16

# Reading points
pointlist = [L-TOL, L-0.5, L-1., L-1.5, L-2., L-2.5, L-3., L-4.,\
             L-5., L-6., L-7., L-8., L-9., L-10., L-11., L-12., \
             L-13., L-14., L-15., L-16., L-17., L-18., L-19., TOL]

# Solve problem
iflag = 0
while abs(t-t_end) > TOL:

   # Print time step
   t += dt
   iflag += 1
   print("\n *** Time: %.2f / %.2f *** Time step: %d / %d *** \n"%((t),(t_end),(iflag),(nsteps)))
   
   # Solve
   solver.solve()
   (ui1,pi1) = wi1.split(True)

   # Assign previous
   wi.assign(wi1)
   
   # Write results
   ui1.rename('u','u'); ufile << (ui1,t)
   pi1.rename('p','p'); pfile << (pi1,t)

   # Write displacement data to file
   resfile = open("terzaghi_settlement_results.txt", "a")
   resfile.write("Tau: %.6f, Settlement: %.6f\n"%(t*c/L**2, (-ui1.sub(1)(Point(0,L-TOL))-w_0)/w_dif))
   resfile.close

# Write pressure data to file
resfile = open("terzaghi_pressure_results.txt", "a")
for i in range(len(pointlist)):
   pointID = Point(0,pointlist[i])
   resfile.write("z: %.6f, p/p0: %.6f\n"%(-(pointlist[i]-L)/L, pi1(pointID)/p0))
resfile.close
