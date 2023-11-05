# -*- coding: utf-8 -*-
"""
Created on 2022 - 05 - 24
Benjamin Bondsman
"""


import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from PyQt5.uic import loadUi
import stressmodel as sm

# -----------------------------------------------------------------------------    
class solverThread(QThread):
    """ Execusion in the background """

    def __init__(self,solver,paramStudy):
        """ Constructor """
        QThread.__init__(self)
        
        self.solver = solver
        self.paramStudy = paramStudy
    # ------------------------------------------------
    
    def __del__(self):
        self.wait()
    # -----------------------------------------------
    
    def run(self):
        
        if (self.paramStudy):
            self.solver.executeParamStudy()
            
        else:
            self.solver.execute()
# ----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """ Main window class"""
    
    def __init__(self):
        """Python constructor"""
        super(QMainWindow, self).__init__()        
        
        
        self.calcDone = "false" # Assurance
        
        # --- Save reference to the application
        
        self.app = app
        self.filename =""
        
        
        # --- Load the window
        
        self.ui = loadUi('mainWindow.ui', self)
        
        
        # --- Visualisation of the main window
        self.setWindowTitle("PSS - Plane stress Simulatior")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.setGeometry(500, 300, 1000, 560)
        self.show()
        
        self.ui.show()
        self.ui.raise_()
        
        # Call and update
        self.initModel()
        self.updateControls()
        self.CalcDone = True
        
        
        
        # --- Connect the controls to evenets
        self.ui.actionNew.triggered.connect(self.onActionNew)
        self.ui.actionOpen.triggered.connect(self.onActionOpen)
        self.ui.actionSave.triggered.connect(self.onActionSave)
        self.ui.actionSave_as.triggered.connect(self.onActionSave_as)
        self.ui.actionExit.triggered.connect(self.onActionExit)
        self.ui.execusionButton.clicked.connect(self.onActionExecute)
        self.ui.actionExecute.triggered.connect(self.onActionExecute)
        
        #
        
        # Visualisation
        
        self.vis = sm.Visualisation(self.input_data,self.input_data,self.CalcDone)
        self.ui.showElementValuesButton.clicked.connect(self.showElementValues)
        self.ui.showNodalValuesButton.clicked.connect(self.showNodalValues)
        self.ui.showGeometryButton.clicked.connect(self.showGeometry)
        self.ui.showMeshButton.clicked.connect(self.showMesh)
        
        
        
        
        
        self.ui.paramButton.clicked.connect(self.onExecuteParamStudy)
    # -------------------------------------------------------------------------
    
    def updateControls(self):
        """ Input data from window"""

        
        self.ui.wEdit.setText(str(self.input_data.w))
        self.ui.hEdit.setText(str(self.input_data.h))
        self.ui.aEdit.setText(str(self.input_data.a))
        self.ui.bEdit.setText(str(self.input_data.b))
        self.ui.tEdit.setText(str(self.input_data.t))
        self.ui.loadEdit.setText(str(self.input_data.load_mag))
        self.ui.elSizeEdit.setText(str(self.input_data.Elementsize))
        self.ui.a_param_start_edit.setText(str(self.input_data.a))
        self.ui.t_param_start_edit.setText(str(self.input_data.t))
        
        
        self.ui.a_param_end_edit.setText(str(0.08))
        self.ui.t_param_end_edit.setText(str(0.2))

        self.ui.param_steps.setValue(5)
        
        
        self.ui.EEdit.setText(str(self.input_data.E))
        self.ui.nuEdit.setText(str(self.input_data.nu))
    # -------------------------------------------------------------------------

    def updateModel(self):
        """ Assign input data to the model """
        
        self.input_data.w = float(self.ui.wEdit.text())
        self.input_data.h = float(self.ui.hEdit.text())
        self.input_data.a = float(self.ui.aEdit.text())
        self.input_data.b = float(self.ui.bEdit.text())
        self.input_data.t = float(self.ui.tEdit.text())
        self.input_data.load_mag = float(self.ui.loadEdit.text())
        self.input_data.E = float(self.ui.EEdit.text())
        self.input_data.nu = float(self.ui.nuEdit.text())
        
        
        self.input_data.a_param = float(self.ui.a_param_start_edit.text())
        self.input_data.t_param = float(self.ui.t_param_start_edit.text())
        
        self.input_data.a_param_end_edit = float(self.ui.a_param_end_edit.text())
        self.input_data.t_param_end_edit = float(self.ui.t_param_end_edit.text())
        
        self.input_data.param_steps = self.ui.param_steps.setValue(5)
        
        self.input_data.Elementsize = float(self.ui.elSizeEdit.text())
    # -------------------------------------------------------------------------


    def onActionNew(self):
        """ Create a new model"""
        print("onActionNew")
        
        self.filename = ""
        self.ui.plainTextEdit.setPlainText("")
        self.vis.closeAll()
        self.updateControls()
    # -------------------------------------------------------------------------
        
    def onActionOpen(self):
        """ Open input file """        
        
        self.filename, _ = QFileDialog.getOpenFileName(self.ui, 
            "Open new model", "", "Modell formats (*.json *.jpg *.bmp)")
        
        if self.filename!="":
        
            self.input_data.load(self.filename)
                
        self.updateControls()
    # -------------------------------------------------------------------------

    def onActionSave(self):
        """" Save the modell """
        
        self.updateModel()
        # self.savedata = sm.input_data()


        if self.filename == "":
            
            self.filename, _  = QFileDialog.getSaveFileName(self.ui, 
                "Save modell", "temp_filename", "Modell format (*.json)")
        
        if self.filename!="":

            self.input_data.save(self.filename)
    # -------------------------------------------------------------------------
    
    def onActionSave_as(self):
        
        self.updateModel()
        # self.savedata = sm.input_data()
        
        self.filename, _  = QFileDialog.getSaveFileName(self.ui, 
                "Save modell", "temp_filename", "Modell format (*.json)")

        self.input_data.save(self.filename)
    # -------------------------------------------------------------------------
        
    def onActionExit(self):
        
        """ Multiple choice of exit """
        
        choice = QMessageBox.question(self,'Exit',
                                      "Are you sure you want to exit?",
                                      QMessageBox.Yes | QMessageBox.No
                                      )
        
        if choice == QMessageBox.Yes:
            print("Exiting ...")
            # sys.exit()
            self.app.exit()
            self.close()
            
        else:
            pass
    # -------------------------------------------------------------------------
    
    def onActionExecute(self):
        """ Execute the calculation"""
    
        self.ui.setEnabled(False)
        
        # Update the input values
        self.updateModel()
        
        # --- Reset
        
        sm.Visualisation.closeAll(self)
        
        
        # Solver
        self.solver = sm.solver(self.input_data,self.output_data)
        
        # Avoid freezing windows      
        self.solverThread = solverThread(self.solver,paramStudy=False)    
        self.solverThread.finished.connect(self.onSolverFinished)  
        self.solverThread.start()
        
        
        # print(self.input_data.Elementsize)
        self.vis.closeAll()
        
    # -------------------------------------------------------------------------
    
    def onSolverFinished(self):
        """ Post processing """
        
        self.ui.setEnabled(True)
        
        
        # --- Generate report
        self.calcDone = True
        self.vis = sm.Visualisation(self.input_data,self.output_data,self.calcDone)
    
        self.ui.plainTextEdit.setPlainText(str(sm.report(self.input_data,self.output_data)))
        
    # -------------------------------------------------------------------------
    
    def showGeometry(self):
        print("Generating geometry")
        self.vis.vis_geom()
    # -------------------------------------------------------------------------
    
    def showMesh(self):
        self.vis.vis_mesh()
    # -------------------------------------------------------------------------

    def showNodalValues(self):
        self.vis.vis_nodal_sol()
    # -------------------------------------------------------------------------
    
    def showElementValues(self):    # Stresses
        self.vis.vis_res(3)
    # -------------------------------------------------------------------------
    
    def initModel(self):
        """Initierar värden på modellen"""

        self.input_data = sm.input_data()
        self.output_data= sm.output_data()
    # -------------------------------------------------------------------------
    
    def onExecuteParamStudy(self):
        self.ui.setEnabled(False)
        self.input_data.param_a = self.ui.a_param_vary_a_radio.isChecked()
        self.input_data.param_t = self.ui.t_param_vary_a_radio.isChecked()
        
        
        #
        self.input_data.param_filename  = 'paramStudy'
        self.input_data.param_steps = int(self.ui.param_steps.value())
        
    
        
        # Find out which one of them is checked
        
        if self.input_data.param_a:
            
            self.input_data.a_start = float(self.ui.a_param_start_edit.text())
            self.input_data.a_end =   float(self.ui.a_param_end_edit.text())
        
        elif self.input_data.param_t:
            self.input_data.t_start = float(self.ui.t_param_start_edit.text())
            self.input_data.t_end =   float(self.ui.t_param_end_edit.text())
        
        

        elif self.input_data.param_steps == 0:
            QMessageBox.warning(self, "Warning", "Number of steps must be greater than zero.")
            
            
        else:
            QMessageBox.warning(self, "Warning", "parameter to be varied is not selected.")
        
        
        # Call the solver
        
        self.solver = sm.solver(self.input_data,self.output_data)
        
        self.SolverThread = solverThread(self.solver, paramStudy = True)   
        
        self.SolverThread.finished.connect(self.onSolverFinished)        
        self.SolverThread.start()        
        
        
        
if __name__ == '__main__':

    # --- Generate application instance 
    app = QApplication(sys.argv)

    # --- Create & visualise the main window
    widget = MainWindow()
    widget.show()

    # --- Start the event   
    sys.exit(app.exec_())
# -------------------------------- End ----------------------------------------
