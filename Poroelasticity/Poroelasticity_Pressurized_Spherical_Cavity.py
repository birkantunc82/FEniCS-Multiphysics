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

# Material parameters
a = 30
c = 1.68E6
Ku = 16200.
Kd = 8000.
G = 6000.
alpha = 0.78
kappa = 1.9E-7
mu = 1E-9
Sigma0 = 1.
TOL = 1E-8

# Time step
t_hat = [0.0001, 0.001, 0.01, 0.1, 1]
t = 0.
index = 4
t_end = t_hat[index]*a**2/c
dt = t_end/100.
nsteps=int(t_end/dt);

# Generate mesh
Ro = 300.; Ri = 30.
mesh = RectangleMesh(Point(Ri,0),Point(Ro,math.pi/2.0),160,30)
rTheta = mesh.coordinates()
xc = zeros((len(rTheta),2))
for i in range(0,len(xc)):
    r = rTheta[i,0]
    theta = rTheta[i,1]
    xc[i,0] = r*math.cos(theta)
    xc[i,1] = r*math.sin(theta)
mesh.coordinates()[:] = xc
n = FacetNormal(mesh)

# Function space
degree=2; pdegree=1;
U = VectorElement("Lagrange", mesh.ufl_cell(), degree)   # Displacement
P = FiniteElement("Lagrange", mesh.ufl_cell(), pdegree)  # Pressure
Element = MixedElement([U, P])
W = FunctionSpace(mesh,Element)
wi1 = Function(W); wi = Function(W)
(ui1, pi1) = split(wi1); (ui, pi) = split(wi)
(u_, p_) = TestFunctions(W)
I=Identity(ui1.geometric_dimension()+1)

# Define boundaries
TOLM = .5
def Bottom(x, on_boundary):
   return x[1] < TOLM and on_boundary
def Left(x, on_boundary):
   return x[0] < TOLM and on_boundary
def Outer(x, on_boundary):
   return sqrt(x[0]**2+x[1]**2) > (Ro-TOLM) and on_boundary  
def Inner(x, on_boundary):
   return sqrt(x[0]**2+x[1]**2) < (Ri+TOLM) and on_boundary  

# Boundary conditions
bcs=list()
bcs.append(DirichletBC(W.sub(0).sub(1), Constant(0.), Bottom))
bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.), Left))
bcs.append(DirichletBC(W.sub(1), Constant(0.), Outer))
bcs.append(DirichletBC(W.sub(1), Constant(Sigma0), Inner))

# Mark inner surface
class wall(SubDomain):
    def inside(self, x, on_boundary):
        return sqrt(x[0]**2+x[1]**2) < (Ri+TOLM) and on_boundary
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
wall().mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

# Define strain in cylindrical coordinates
x = SpatialCoordinate(mesh)
def eps(u):
   return as_tensor([[u[0].dx(0), 0, 0.5*(u[0].dx(1)+u[1].dx(0))],
                      [0, u[0]/x[0], 0],
                      [0.5*(u[0].dx(1)+u[1].dx(0)), 0, u[1].dx(1)]])

# Define Laplace equation in cylindrical coordinates
def pr(p):
   return as_vector([p.dx(0), p.dx(1)])

# Define stress
def T(u,p):
   return 2.*G*eps(u) + (Kd-2./3.*G)*tr(eps(u))*I - alpha*p*I

# Define problem
F = inner(T(ui1,pi1), eps(u_))*x[0]*dx \
    + (kappa/mu)*dot(pr(pi1),pr(p_))*x[0]*dx + alpha*(tr(eps(ui1))-tr(eps(ui)))/dt*p_*x[0]*dx + (alpha**2./(Ku-Kd))*(pi1-pi)/dt*p_*x[0]*dx \
    - inner(Sigma0*n, u_)*x[0]*ds(1)
J = derivative(F,wi1)
problem=NonlinearVariationalProblem(F,wi1,bcs,J)
solver=NonlinearVariationalSolver(problem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['absolute_tolerance'] = 1E-10
prm['newton_solver']['relative_tolerance'] = 1E-9
prm['newton_solver']['maximum_iterations'] = 16

# Reading points
pointlist = [30.+TOL, 33., 36., 39., 42., 45., 48., 51., 54., 57., 60.]

# Solve problem
iflag = 0
while abs(t-t_end) > TOL:

   # Print time step
   t += dt
   iflag += 1
   print("\n Time step: %d / %d \n"%(iflag,nsteps))
   
   # Solve
   solver.solve()
   (ui1, pi1) = wi1.split(True)

   # Assign previous
   wi.assign(wi1)
   
   # Write results
   #ui1.rename('u','u'); ufile << (ui1,t)
   #pi1.rename('p','p'); pfile << (pi1,t)

# Write data to file
resfile = open("Spherical_cavity_results.txt", "a")
for i in range(len(pointlist)):
   pointID = Point(pointlist[i] * math.cos(math.radians(45)),pointlist[i] * math.sin(math.radians(45)))
   resfile.write("R_hat: %.6f, p/Sigma0: %.6f\n"%(pointlist[i]/Ri, pi1(pointID)/Sigma0))
resfile.close
