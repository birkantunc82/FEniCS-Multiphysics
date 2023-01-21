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
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 12
parameters['linear_algebra_backend'] = "PETSc"
print('Linear algebra backend:',parameters['linear_algebra_backend'])

# Delete existing folders/files and generate new files
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.splitext(__file__)[0]
myfile=("%s.results" % (fileName))
if os.path.exists(myfile):
   shutil.rmtree("%s/%s.results" % (dir_path,fileName), ignore_errors=True)
pfile = File("%s.results/p.pvd" % (fileName))
vfile = File("%s.results/v.pvd" % (fileName))
taufile = File("%s.results/tau.pvd" % (fileName))
mfile = File("%s.results/m.pvd" % (fileName))
ufile = File("%s.results/u.pvd" % (fileName))

# Time step
t = 0.0
dt = .1
TOL = 1E-8
degree = 2
pdegree = 1

# Material parameters
alpha = 1./9.
Wi = 1.05            
De = Wi / 3.
Newtonian = True
if Newtonian:
   alpha = 1.
   De = 0.

# Generate mesh
LH = 1.; LW1=2.; LI=3.; LW2=LI+LW1
mesh = RectangleMesh(Point(0,0), Point(LW2,LH), 128,16)
n = FacetNormal(mesh)
print('Number of nodes: %d, Number of elements: %d'%(mesh.num_vertices(),mesh.num_cells()))
mfile << mesh

# Mixed element
P = FiniteElement("CG", 'triangle', 1)    # Pressure
V = VectorElement("CG", 'triangle', 2)    # Velocity
Tau = TensorElement("CG", 'triangle', 2)  # Extra stress
E = MixedElement([P, V, Tau])

# Elements and space
W = FunctionSpace(mesh,E); w = Function(W);
(p,v,tau) = split(w); (p_,v_,tau_) = TestFunctions(W)
I = Identity(2)
V1 = VectorFunctionSpace(mesh, "CG", 1)
v_tilde = Function(V1);
V2 = VectorFunctionSpace(mesh, "CG", 2)
u = Function(V1); ufinal = Function(V2); uinit = Function(V2);

# Save initial coordinates of the mesh
coord = Expression(('x[0]','x[1]'), degree=degree)
uinit.assign(interpolate(coord, V2))

# Generate DoF map for mesh
mmap = vertex_to_dof_map(V1)
mmap.resize((int(len(mmap)/2), mesh.geometry().dim()))

# Define boundaries
def top(x, on_boundary):
   return on_boundary and x[1]>LH-TOL and x[0]<LW1+TOL
def bottom(x, on_boundary):
   return on_boundary and x[1]<TOL
def inlet(x, on_boundary):
   return on_boundary and x[0]<TOL
def outlet(x, on_boundary):
   return on_boundary and x[0]>LW2-TOL

# Define boundary conditions
bcs = list()
velocity = Expression(('1.5*(1. - x[1]*x[1])*t', '0'), t=t, degree=degree)
bcs.append( DirichletBC(W.sub(1),Constant((0,0)),top) )
bcs.append( DirichletBC(W.sub(1).sub(1),Constant(0),bottom))
bcs.append( DirichletBC(W.sub(1),velocity,inlet) )
bcs.append( DirichletBC(W.sub(1).sub(1),Constant(0),outlet))

# Functions
def Constitutive(v, tau):
   L = grad(v)
   D = (grad(v)+grad(v).T)/2.0
   return tau + De * (dot(v,nabla_grad(tau)) - dot(L,tau) - dot(tau,L.T) ) - 2.*(1.-alpha)*D

def T(p,v,tau):
   D = (grad(v)+grad(v).T)/2.0
   return -p*I + 2.*alpha*D + tau

# SUPG Coefficient
def eta(v):
   he=CellDiameter(mesh)
   nb=sqrt(dot(v,v))
   nb=conditional(le(nb,TOL), 1., nb)
   return he/nb

# Problem definition
Fsol = div(v)*p_*dx \
      + inner(T(p,v,tau),grad(v_))*dx \
      + inner(Constitutive(v,tau),tau_)*dx \
      + inner(Constitutive(v,tau),dot(eta(v)*v,nabla_grad(tau_)))*dx
Jsol = derivative(Fsol,w)
problem=NonlinearVariationalProblem(Fsol,w,bcs,Jsol)
solver=NonlinearVariationalSolver(problem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['absolute_tolerance'] = 1.E-10
prm['newton_solver']['relative_tolerance'] = 1.E-9
prm['newton_solver']['maximum_iterations'] = 16

# Free surface nodes
coords = mesh.coordinates()
fsnodes=list()
for i in range(len(coords)):
   if coords[i][0] > LW1-TOL and coords[i][1] > LH-TOL:
      fsnodes.append((i,coords[i][0],coords[i][1]))
fsnodes = np.array(fsnodes, dtype=[('index', int),('x', float),('y', float)])
fsnodes = np.sort(fsnodes, order='x')
u=np.zeros(len(fsnodes))
u_old=np.zeros(len(fsnodes))
delta_u=np.zeros(len(fsnodes))
vhat=np.zeros((2,len(fsnodes)))

# Rigth boundary nodes
rbnodes=list()
for i in range(len(coords)):
   if coords[i][0] > LW2-TOL:
      rbnodes.append((i,coords[i][0],coords[i][1]))
rbnodes = np.array(rbnodes, dtype=[('index', int),('x', float),('y', float)])
rbnodes = np.sort(rbnodes, order='y')

run = True
if run:
   # Solve problem
   iflag = 0
   t = 0
   ufinal.rename("u", "u"); ufile << (ufinal, t)
   L2TOL=1
   while iflag < 11 or L2TOL > 1E-4:

      # Print time step
      t += dt
      iflag += 1
      if iflag<=10: velocity.t=t
      print("\n Iteration: %d \n"%(iflag))
      
      # Solve
      solver.solve()
      (p,v,tau) = w.split(True);
      
      # Interpolate the velocity to a 1st order space
      v_tilde.assign(interpolate(v,V1))

      # Get the velocity components for the free surface nodes
      for i in range(len(fsnodes)):
         vhat[0][i] = v_tilde.vector().get_local()[mmap[fsnodes[i][0]][0]]
         vhat[1][i] = v_tilde.vector().get_local()[mmap[fsnodes[i][0]][1]]
	
      # Calculate the displacement of the free surface nodes using the streamline method
      for i in range(len(fsnodes)-1):
            u[i+1] = u[i] + (vhat[1][i+1] / vhat[0][i+1] + vhat[1][i] / vhat[0][i] ) \
                     /2. *(fsnodes[i+1][1]-fsnodes[i][1])

      # Calculate the incremental displacement and assign the current displacement to use in the next step
      delta_u = np.subtract(u,u_old)
      for i in range(len(u)):
         u_old[i]=u[i]
      
      # Calculate the L2 norm of the incremental displacements to use as a stopping criteria
      L2TOL = 0
      for i in range(len(u)):
         L2TOL += delta_u[i]*delta_u[i]
      L2TOL = sqrt(L2TOL)
      print("L2TOL: %.8f"%(L2TOL))
               
      # Move the free surface nodes
      for i in range(len(fsnodes)):
         coords[fsnodes[i][0]][1] += delta_u[i]

      # Move the right boundary nodes   
      enddisp = delta_u[len(fsnodes)-1]
      for i in range(len(rbnodes)-2):
         coords[rbnodes[i+1][0]][1] += enddisp * rbnodes[i+1][2] / rbnodes[len(rbnodes)-1][2]

      # Mesh smoothing
      mesh.smooth(10)

      # Save final coordinates of the mesh and calculate displacement of each node
      coord = Expression(('x[0]','x[1]'), degree=degree)
      ufinal.assign(interpolate(coord, V2))
      ufinal.assign(project(ufinal-uinit, V2))

      # Save solution
      p.rename("p", "p"); pfile << (p , t)
      v.rename("v", "v"); vfile << (v , t)
      tau.rename("tau", "tau"); taufile << (tau , t)
      ufinal.rename("u", "u"); ufile << (ufinal , t)

      # Print swelling ratio 
      maxdisp = np.max(ufinal.vector().get_local())
      swelling = maxdisp + LH
      print('Swelling ratio: %.4f'%(swelling))
