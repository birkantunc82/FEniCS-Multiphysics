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
p1 = Point(0,L-TOL)
p_1 = Point(0,L-0.5)
p2 = Point(0,L-1.)
p_2 = Point(0,L-1.5)
p3 = Point(0,L-2.)
p_3 = Point(0,L-2.5)
p4 = Point(0,L-3.)
p5 = Point(0,L-4.)
p6 = Point(0,L-5.)
p7 = Point(0,L-6.)
p8 = Point(0,L-7.)
p9 = Point(0,L-8.)
p10 = Point(0,L-9.)
p11 = Point(0,L-10.)
p12 = Point(0,L-11.)
p13 = Point(0,L-12.)
p14 = Point(0,L-13.)
p15 = Point(0,L-14.)
p16 = Point(0,L-15.)
p17 = Point(0,L-16.)
p18 = Point(0,L-17.)
p19 = Point(0,L-18.)
p20 = Point(0,L-19.)
p21 = Point(0,TOL)

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
   resfile.write("%.6f, %.6f\n"%(t*c/L**2, (-ui1.sub(1)(p1)-w_0)/w_dif))
   resfile.close

# Write pressure data to file
resfile = open("terzaghi_pressure_results.txt", "a")
resfile.write("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n"%(tau[index], pi1(p1)/p0, pi1(p_1)/p0, pi1(p2)/p0, pi1(p_2)/p0, pi1(p3)/p0, pi1(p_3)/p0, pi1(p4)/p0, pi1(p5)/p0, pi1(p6)/p0, pi1(p7)/p0, pi1(p8)/p0, pi1(p9)/p0, pi1(p10)/p0, pi1(p11)/p0, pi1(p12)/p0, pi1(p13)/p0, pi1(p14)/p0, pi1(p15)/p0, pi1(p16)/p0, pi1(p17)/p0, pi1(p18)/p0, pi1(p19)/p0, pi1(p20)/p0, pi1(p21)/p0))
resfile.close
