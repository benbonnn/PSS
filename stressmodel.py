# -*- coding: utf-8 -*-
"""
Created on 2022 - 05 - 24
Benjamin Bondsman
"""

# Import necessary packages
import numpy as np
import calfem.core as cfc

import calfem.geometry as cfg  # Geometry routines
import calfem.mesh as cfm      # Discretisation
import calfem.vis as cfv       # Visualisation
import calfem.utils as cfu     # Utilities
import pandas as pd            # Pandas to construct data frame
from tabulate import tabulate  # Google tabulate
import json                    # Jason to produce report
import pyvtk as vtk            # Export to Paraview

# -----------------------------------------------------------------------------
class input_data(object):
    """ Class for definine input data """
    
    print(" Processing input data...")
    
    def __init__(self):
        
        # Constants
        
        self.material = "Steel"
        self.version = 1
        self.h = 0.1
        self.w = 0.3
        self.a = 0.05
        self.b = 0.025
        self.E = 2.08e10
        self.nu = 0.2
        self.t = 0.15
        self.Elementsize = 0.05
        self.load_mag = 100e3
        
        self.a_param = self.a
        self.t_param = self.t

    # -------------------------------------------------------------------------

    def geometry(self):
        
        """ Define geometrical boundaries"""        
        
        w = float(self.w)
        h = float(self.h)
        a = float(self.a)
        b = float(self.b)
        
        
        # Initialise labeling for fix and loading boundaries
        q = 20
        clamp = 30
        
        """ Create geometry """
        g = cfg.Geometry()
        
        
        """ Define geometrical nodal points"""
        g.point([0, 0])             # Node 0
        g.point([(w-a)*0.5, 0])     # Node 1
        g.point([(w-a)*0.5, b])     # Node 2
        g.point([(w+a)*0.5, b])     # Node 3
        g.point([(w+a)*0.5, 0])     # Node 4
        g.point([w, 0])             # Node 5
        g.point([w, h])             # Node 6
        g.point([(w+a)*0.5, h])     # Node 7
        g.point([(w+a)*0.5, h-b])   # Node 8
        g.point([(w-a)*0.5, h-b])   # Node 9
        g.point([(w-a)*0.5, h])     # Node 10
        g.point([0, h])             # Node 11
        
        
        
        """ Create lines to connect the nodes """
        g.spline([0, 1])                    # Line 1
        g.spline([1, 2])                    # Line 2
        g.spline([2, 3])                    # Line 3
        g.spline([3, 4])                    # Line 4
        g.spline([4, 5])                    # Line 5
        g.spline([5, 6],marker = q)         # Line 6
        g.spline([6, 7])                    # Line 7
        g.spline([7, 8])                    # Line 8
        g.spline([8, 9])                    # Line 9
        g.spline([9,10])                    # Line 10
        g.spline([10, 11])                  # Line 11
        g.spline([11, 0],marker = clamp)    # Line 12
        
        
        
        """ Create surface """
        
        g.surface([0,1,2,3,4,5,6,7,8,9,10,11])

        return g
    # -------------------------------------------------------------------------
    
    # ------- Save input_data to a file
    def save(self,filename):
        
        """Initialise input data dictionary"""
        
        input_data = {}
        
        
        # ------- Assign data
        input_data = {}
        input_data["version"] = self.version
        input_data["Material"] = self.material
        input_data["t"] = self.t
        # input_data["ep"] = self.ep
        
        input_data["w"] = self.w
        input_data["a"] = self.a
        input_data["b"] = self.b
        input_data["h"] = self.h
        input_data["t"] = self.t
        
        input_data["nu"] = self.nu
        input_data["E"] = self.E
        
        input_data["Elementsize"] = self.Elementsize
        input_data["load"] = self.load_mag
        # input_data["D"] = self.D.tolist()
        
        
        # ------- Open/Create file
        with open(filename,'w', encoding = 'utf-8') as ofile:
        # ofile = open(filename,"w")
        
            json.dump(input_data, ofile, sort_keys = True, indent = 4)
        
        # ofile.close()
        
        return input_data
    # -------------------------------------------------------------------------
        
    # ------- Read data from file
    def load(self,filename):
        
        # ------- Read the file
        with open(filename, 'r', encoding = 'utf-8') as ifile:
        
            # ifile = open(filename,"r")
            input_data = json.load(ifile)
            # ifile.close()
            
            # self.version = input_data["version"]
            # self.material = input_data["Material"]
            self.t = input_data["t"]
            # self.ep = input_data["ep"]    
            
            self.w = input_data["w"]
            self.a = input_data["a"]
            self.b = input_data["b"]
            self.h = input_data["h"]
            
            self.nu = input_data["nu"]        
            self.E = input_data["E"]
            
            self.Elementsize = input_data["Elementsize"]
            self.load_mag = input_data["load"]
            
        return input_data
    
# -----------------------------------------------------------------------------

# Output data class to store results
class output_data(object):
    def __init__(self):
        
        self.calcDone = False   # Assurance
        # ------- Initialise quantities to be stored
        self.ex = None
        self.ey = None
        
        self.edof = None
        self.coords = None
        self.geometry = None
        self.dofs_per_node = None
        self.Elementsize = None
        self.el_type = None
        self.a_solve = None
        self.ed = None
        self.r = None
        self.es = None
        self.et = None
        
        self.vonMises = None
        self.nelm = None
        
# -----------------------------------------------------------------------------

# Solver class to solve the equation system
class solver(object):
    
    print("Solving the system...")
    
    #  ------- Initialise in and output data
    def __init__(self,input_data,output_data):
        
        
        self.input_data = input_data
        self.output_data = output_data
    # -------------------------------------------------------------------------
    
    # ------- Execute the solver
    def execute(self):
        
        print('Solving the system ...')
       
        # ------- Extract data
        

        # ep = self.input_data.ep
        Elementsize = self.input_data.Elementsize
        load_mag = self.input_data.load_mag
        
        # Geometrical description
        geometry = self.input_data.geometry()    # g from the input class
        
        # Element type and degrees-of-freedom per nodal point
        el_type = 3
        dofs_per_node = 2
        
        # Generate mesh
        mesh = cfm.GmshMeshGenerator(geometry)
        
        
        # Constitutive material behaviour (Steel)
        ep = [1, self.input_data.t]
        D = cfc.hooke(1,self.input_data.E,self.input_data.nu)
        
        
        # Assign into mesh
        mesh.elSizeFactor = Elementsize
        mesh.elType = el_type
        mesh.dofsPerNode = dofs_per_node
        mesh.returnBoundaryElements = True
        
        # Generate mesh
        coords, edof, dof, bdofs, elementmarkers, boundaryElements = mesh.create()
        self.output_data.topo = mesh.topo
        
        
        
        
        nodes = np.arange(1,len(coords[:,0]),1)
        
        # Boundaty conditions
        bc = np.array([],'i')
        bcVal = np.array([],'i')
        
        
        
       
        # Extract coordinates
        ex,ey = cfc.coordxtr(edof,coords,dof)
       
        # Initialise necessities
        ndof = np.size(dof)
        nelm = np.max(edof.shape[0])

        
        K = np.zeros((ndof,ndof), dtype = float)
        f = np.zeros((ndof,1), dtype = float)
        
        
        # Apply Dirichlet boundary condition
        bc, bcVal = cfu.applybc(bdofs,bc,bcVal,30,0.0,0)    # Precribe right boundary
        
        cfu.applyforce(bdofs,f,20,load_mag,1)  # Apply force at left boundary
        

        for elx, ely, eltopo in zip(ex,ey,edof):
            ke = cfc.planqe(elx,ely,ep,D)
            cfc.assem(eltopo,K,ke)

        # Solve the linear equation system Ka = f

        a_solve, r = cfc.solveq(K,f,bc,bcVal)
        
       
        # -------  Element displacements
        ed = cfc.extract_eldisp(edof,a_solve)
        
        
        es = np.array(np.zeros((nelm,3), dtype = float))
        et = np.array(np.zeros((nelm,3), dtype = float))

        
        # i = 0
        vonMises = []
        
        # Compute stresses and strains
        # for i, elx, ely, eld in enumerate(zip(ex,ey,ed)):
        for i, (elx, ely, eld) in enumerate(zip(ex, ey, ed)):
            es_el, et_el = cfc.planqs(elx,ely,ep,D,eld)
            
            es[i,:] = es_el     # [sigx sigy tauxy]
            et[i,:] = et_el
            
            vonMises.append( np.sqrt( pow(es_el[0],2)  + pow(es_el[1],2) - es_el[0]*es_el[1] + 3*pow(es_el[2],2) ))
            
            # i+=1
        vonMises = np.array(vonMises, dtype = float)
        
        
        print("Solving done...")
        # ---- save
        self.output_data.ep = ep
        self.output_data.nodes = nodes
        self.output_data.D = D
        self.output_data.edof = edof
        self.output_data.dof = dof
        self.output_data.coords = coords
        self.output_data.nelm = nelm
        self.output_data.ndof = ndof
        self.output_data.ex = ex
        self.output_data.ey = ey
        
        self.output_data.geometry = geometry
        self.output_data.dofs_per_node = dofs_per_node
        self.output_data.Elementsize = Elementsize
        self.output_data.el_type = el_type
        
        self.output_data.a_solve = a_solve
        self.output_data.ed = ed
        self.output_data.r = r
        self.output_data.es = es
        self.output_data.et = et
        self.output_data.vonMises = vonMises
        self.output_data.calcDone = True
        
        # Export data to matlab
        # np.savetxt('coords.txt', coords, fmt = '%.6e')
        # np.savetxt('edof.txt', edof, fmt = '%.6e')
        # np.savetxt('D.txt', D, fmt = '%.6e')
        # np.savetxt('ex.txt', ex, fmt = '%.6e')
        # np.savetxt('ey.txt', ey, fmt = '%.6e')
        # np.savetxt('a_solve.txt', a_solve, fmt = '%.6e')
        # np.savetxt('r.txt', r, fmt = '%.6e')
        
        # np.savetxt('bc.txt', bc, fmt = '%.6e')
        # np.savetxt('bcVal.txt', bcVal, fmt = '%.6e')
        
        # np.savetxt('bdofstxt', np.array(bdofs), fmt = '%.6e')
        # ---------------------------------------------------------------------
        
    def executeParamStudy(self):
        ''' Execute parametric study'''
        
        a_old = self.input_data.a
        t_old = self.input_data.t
        
    
        if self.input_data.param_a:
            
            a_range = np.linspace(self.input_data.a_start, self.input_data.a_end, self.input_data.param_steps)
            
            # Execute parametric study
            i = 1
            
            for a in a_range:
                print("Executing a = %g..." %a)
                
                filename = 'stress_study_'
                
                self.input_data.a = float(a)
                # Execute
                Solver = solver(self.input_data,self.output_data)
                Solver.execute()
            
                # Export to vtk
                
                filename += str(i)
                self.exportVtk(filename)
                i +=1
    
        elif self.input_data.param_t:
            
            # Construct array
            
            t_range = np.linspace(self.input_data.t_start, self.input_data.t_end, self.input_data.param_steps)
            
            for t in range(t_range):
                print("Executing a = %g..." %t)
                
                filename = 'stress_study'
                
                self.input_data.t = float(t)
                
                # Execute
                Solver = solver(self.input_data,self.output_data)
                Solver.execute()
                
                # Export to Vtk
                
                filename += str(i)
                self.exportVtk(filename)
                
                i +=1
                
                
        # Reset
        self.input_data.a = a_old
        self.input_data.t = t_old                
            # -----------------------------------------------------------------
            

    def exportVtk(self,filename):
        
 
        print("Exporting results %s." % filename, 'to Paraview')
        
        points = self.output_data.coords.tolist()
        polys = (self.output_data.topo-1).tolist()
        
        
        res_data = vtk.CellData(vtk.Scalars(self.output_data.vonMises, name = 'Mises'),
                                vtk.Scalars(self.output_data.es[:,0]/1e6, name = 'Stresses 11'),
                                vtk.Scalars(self.output_data.es[:,1]/1e6, name = 'Stresses 22'),
                                vtk.Scalars(self.output_data.es[:,2]/1e6, name = 'Stresses 12')
                            )
        
        # Construct the structure
        struc = vtk.PolyData(points = points, polygons = polys)
        
        # Construct the results
        
        vtk_data = vtk.VtkData(struc, res_data)
        
        # Export
        vtk_data.tofile(filename, "ascii")
        
# -----------------------------------------------------------------------------

# Report class to produce a report(print out in- and outputs)
class report(object):
    
    print("Generating report...")
    
    def __init__(self,input_data,output_data):
        
        print("Generating report...")
        
        # ------- Initialise in and output data
        self.input_data = input_data
        self.output_data = output_data
    # -------------------------------------------------------------------------
    
        # ------- Clear the report
    def clear(self):
        self.report = ""
    
    # -------------------------------------------------------------------------        

    # ------- Add text to a new line
    def add_text(self,text = ""):
        self.report += str(text) + "\n"
    
    # -------------------------------------------------------------------------
    def table_construct(self,variable,header):
        
        self.tab = tabulate(variable, header)
        
        return print(self.tab)
    # -------------------------------------------------------------------------
    
    # ------- Results to be printed
    def __str__(self):
        
        # ------- Clear the report
        self.clear()
        
        self.add_text()
        self.add_text("--------- Version ---------")
        
        self.add_text()
        self.add_text("* v. 1.0")
        
        self.add_text()
        self.add_text("--------- Material ---------")
        
        self.add_text()
        self.add_text("* Steel: " + "E = " + str(self.input_data.E/1e6) + "[MPa], " + "nu = " + str(self.input_data.nu))

        
        self.add_text()
        self.add_text("--------- FE discretisation ---------")
        
        self.add_text()
        self.add_text("* Number of elements: " + str(self.output_data.nelm))        
        
        self.add_text()
        self.add_text("* Number of degrees of freedom: "+ str(self.output_data.ndof))
        
        self.add_text()
        self.add_text("* Size of the system matrix: "+ str(self.output_data.ndof) + ' x ' + str(self.output_data.ndof))
        
        
        # ------- Text to a new line
        self.add_text()
        
        self.add_text("--------- Results ---------")
        
        
        # --- Construct table with element results
        element_data = self.element_data = np.zeros((self.output_data.nelm, 12), dtype = float)

        self.elements = np.arange(len(self.output_data.edof[:,0]))+1
        
        for i in range(self.output_data.nelm):
            
            element_data[i,0] = self.elements[i]
            
            
            element_data[i,1] = self.output_data.topo[i,0]
            element_data[i,2] = self.output_data.topo[i,1]
            element_data[i,3] = self.output_data.topo[i,2]
            element_data[i,4] = self.output_data.topo[i,3]
            
            
            # Stresses
            element_data[i,5] = self.output_data.es[i,0]/1e6
            element_data[i,6] = self.output_data.es[i,1]/1e6
            element_data[i,7] = self.output_data.es[i,2]/1e6
            element_data[i,8] = self.output_data.vonMises[i]/1e6
            
            # Strains
            element_data[i,9] = self.output_data.et[i,0]    
            element_data[i,10] = self.output_data.et[i,1]
            element_data[i,11] = self.output_data.et[i,2]
        
        
        # Construct pandas data frame
        df = pd.DataFrame(element_data, columns =["Element no.",
                                    "Node 1", "Node 2", "Node 3", "Node 4",
                                    "σ₁₁ [MPa]","σ₂₂ [MPa]","σ₁₂ [MPa]", "VonMises [MPa]",
                                     "ε₁₁ [-]","ε₂₂ [-]","ε₁₂ [-]"])
        
        # Set Element number as index
        df.set_index('Element no.', inplace=True)

        # Header
        self.add_text()
        self.add_text("* Element stresses and strains:")
        self.add_text()
        
        # Table        
        self.report += tabulate(df, headers = 'keys', tablefmt="psql",colalign=("center","center","center",
                                                                                "center","center","center",
                                                                                "center","center","center",
                                                                                "center","center","center"
                                                                                ),
                                                                                floatfmt=(".0f", ".0f", ".0f",
                                                                                          ".0f", ".3f", ".3f",
                                                                                          ".3f", ".3f", ".3f",
                                                                                          ".3f", ".3f", ".3f"))
        
        
        # --- Node tables
        
        dofs = np.array(self.output_data.dof, dtype = float).astype(int)
        a_solve = np.array(self.output_data.a_solve, dtype = float)
        r = np.array(self.output_data.r, dtype = float)
        
        # Initialise matrix
        nodal_data = self.nodal_data = np.zeros(
            (len(self.output_data.coords[:,0]),9), dtype = float
            )
        
        # Assign first column
        nodal_data[:,0] = np.arange(1,len(self.output_data.coords[:,0])+1,1)
        
        nodal_data[:,1] = self.output_data.coords[:,0]
        nodal_data[:,2] = self.output_data.coords[:,1]
        
        
        nodal_data[:,3] = dofs[:,0]
        nodal_data[:,4] = dofs[:,1]
        
        
        nodal_data[:,5] = np.squeeze(a_solve[dofs[:,0]-1])*1000
        nodal_data[:,6] = np.squeeze(a_solve[dofs[:,1]-1])*1000
        
        nodal_data[:,7] = np.squeeze(r[dofs[:,0]-1])
        nodal_data[:,8] = np.squeeze(r[dofs[:,1]-1])
        
        

        
        df2 = pd.DataFrame(nodal_data, columns = ["Node", "x_coord", "y_coord",
                                                  "x_dof", "y_dof","a_x [mm]",
                                                  "a_y [mm]","r_x [N]", "r_y [N]"])
        df2.set_index('Node', inplace=True)
        
        
        
        # Construct
        self.add_text()
        self.add_text()
        self.add_text("* Nodal solutions:")
        self.add_text()  
        
        # Table        
        self.report += tabulate(df2,
                                headers = 'keys',
                                tablefmt="psql",
                                colalign=("center","center","center",
                                          "center","center","center",
                                          "center","center","center"),
                                floatfmt=(".0f", ".5f", ".5f",
                                          ".0f", ".0f", ".5f",
                                          ".5f", ".5f", ".5f"))   
        
    
        
        return self.report

# -----------------------------------------------------------------------------
class Visualisation(object):
    
    print("Visualising results...")
    
    def __init__(self, input_data, output_data, calcDone):
        """ Python constructor """
        
        self.input_data = input_data
        self.output_data = output_data
        self.calcDone = calcDone
    # --- Store references to the opened figures

        self.geomFig = None    
        self.meshFig = None
        self.elValueFig = None   
        self.nodeValueFig = None        
    
    
    
    def vis_geom(self):
        
        if self.calcDone == True:
            
            self.geomFig = cfv.figure(self.geomFig)
            
            cfv.clf()
            cfv.draw_geometry(self.output_data.geometry,
                              title="Geometry")
            cfv.showAndWait()
    
    def vis_mesh(self):
        
        if self.calcDone == True:
            
            
            self.meshFig = cfv.figure(self.meshFig)
            cfv.clf()

            cfv.drawMesh(coords = self.output_data.coords,
                          edof = self.output_data.edof,
                          dofs_per_node = self.output_data.dofs_per_node,
                          el_type = self.output_data.el_type, 
                          title="FE-Domain")
            
            cfv.showAndWait()
        
    def vis_res(self, plt_type):
        
        
        if self.calcDone == True:
            
            
            """
            Element Stresses
            
            plt_type = 1  ---> Axial stresses S_11
            plt_type = 2  ---> Transversal stresses S_22
            plt_type = 12 ---> Shear stresses S_12
            
            plt_type = 3  ---> von Mises stresses
            """
    
            a_solve = self.output_data.a_solve
            
            # Arrange nodal displacements for plotting
            disp = np.array([[a_solve[:,0:2:-1], a_solve[:,1:2:-1]]])
            
            
            if plt_type == 1:
            
                cfv.figure()
                cfv.drawElementValues(self.output_data.es[:,0]/1e6,
                                      coords = self.output_data.coords,
                                      edof = self.output_data.edof,
                                      dofs_per_node = self.output_data.dofs_per_node,
                                      el_type = self.output_data.el_type,
                                      displacements=disp, 
                                      clim=None, 
                                      axes=None, 
                                      axes_adjust=True, 
                                      draw_elements=False, 
                                      draw_undisplaced_mesh=False, 
                                      magnfac=10.0, 
                                      title="Axial stresses")
                 
                cfv.colorBar().SetLabel("Stress [MPa]")
                cfv.info(" Plotting done! ")
                cfv.showAndWait()
                        
            elif plt_type == 2:
            
                cfv.figure()
                
                cfv.drawElementValues(self.output_data.es[:,1]/1e6,
                                      coords = self.output_data.coords,
                                      edof = self.output_data.edof,
                                      dofs_per_node = self.output_data.dofs_per_node,
                                      el_type = self.output_data.el_type,
                                      displacements=disp, 
                                      clim=None, 
                                      axes=None, 
                                      axes_adjust=True, 
                                      draw_elements=False, 
                                      draw_undisplaced_mesh=False, 
                                      magnfac=1.0, 
                                      title="Transversal stresses")
                                       
                cfv.colorBar().SetLabel("Stress [MPa]")
                cfv.showAndWait()
            elif plt_type == 12:
                cfv.figure()
                cfv.drawElementValues(self.output_data.es[:,2]/1e6,
                                      coords = self.output_data.coords,
                                      edof = self.output_data.edof,
                                      dofs_per_node = self.output_data.dofs_per_node,
                                      el_type = self.output_data.el_type,
                                      displacements=disp, 
                                      title = "Mises stress",
                                      magnfac=1)
                 
                cfv.colorBar().SetLabel("Shear stresses [MPa]")
                cfv.showAndWait()
            if plt_type == 3:
            
                cfv.figure()
                cfv.drawElementValues(ev = self.output_data.vonMises/1e6,
                                      coords = self.output_data.coords,
                                      edof = self.output_data.edof,
                                      dofs_per_node = self.output_data.dofs_per_node,
                                      el_type = self.output_data.el_type,
                                      displacements=disp, 
                                      clim=None, 
                                      axes=None, 
                                      axes_adjust=True, 
                                      draw_elements=False, 
                                      draw_undisplaced_mesh=False, 
                                      magnfac=10.0, 
                                      title="von Mises stresses")
                 
                cfv.colorBar().SetLabel("stresse [MPa]")
                cfv.showAndWait()
        else:
            print('Please execute ')
            
    def vis_nodal_sol(self):
        
        if self.calcDone == True:
        
        
            coords = self.output_data.coords
            edof = self.output_data.edof
            dofs_per_node = self.output_data.dofs_per_node
            el_type = self.output_data.el_type
            a_solve = self.output_data.a_solve
            
    
            cfv.figure()
            cfv.drawDisplacements(a_solve,
                                  coords,
                                  edof,
                                  dofs_per_node,
                                  el_type,
                                  node_vals=None,
                                  clim=None,
                                  axes=None,
                                  axes_adjust=True,
                                  draw_undisplaced_mesh=True,
                                  magnfac=10.0,
                                  title="Nodal displacements")
            cfv.showAndWait()
            
        else:
            print('Please execute')
    
    def wait(self):

        cfv.showAndWait()
    
    
    def closeAll(self):
        
        self.geomFig = None
        self.meshFig = None
        self.elValueFig = None
        self.nodeValueFig = None
        
        
        cfv.clf()
        cfv.close_all()
# -------------------------------- End ----------------------------------------
