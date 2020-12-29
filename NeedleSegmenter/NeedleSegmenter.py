import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import math
import numpy as np
from vtk.util import numpy_support
from scipy import ndimage
from skimage.filters import meijering, sato
import cv2
import tempfile
import matplotlib.pyplot as plt


class NeedleSegmenter(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "2D Needle Segmenter "
    self.parent.categories = ["Filtering"]
    self.parent.dependencies = []
    self.parent.contributors = ["Ahmed Mahran (BWH), Junichi Tokuda (BWH)"]
    self.parent.helpText = """This is a 2D needle segmenter module used to localize needle tip in the MRI image. Input requirment: 
    Magnitude image and Phase Image. Uses Prelude phase unwrapping algorithm. """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """"""

class NeedleSegmenterWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    ####################
    # For debugging
    #
    # Reload and Test area
    reloadCollapsibleButton = ctk.ctkCollapsibleButton()
    reloadCollapsibleButton.text = "Reload && Test"
    self.layout.addWidget(reloadCollapsibleButton)
    reloadFormLayout = qt.QFormLayout(reloadCollapsibleButton)
    
    # reload button
    # (use this during development, but remove it when delivering
    #  your module to users)
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "Reload"
    reloadFormLayout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.onReload)

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)
    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input magnitude volume
    #
    self.magnitudevolume = slicer.qMRMLNodeComboBox()
    self.magnitudevolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.magnitudevolume.selectNodeUponCreation = True
    self.magnitudevolume.addEnabled = True
    self.magnitudevolume.removeEnabled = True
    self.magnitudevolume.noneEnabled = True
    self.magnitudevolume.showHidden = False
    self.magnitudevolume.showChildNodeTypes = False
    self.magnitudevolume.setMRMLScene( slicer.mrmlScene )
    self.magnitudevolume.setToolTip("Select the magnitude image")
    parametersFormLayout.addRow("Magnitude Image: ", self.magnitudevolume)

    #
    # input phase volume
    #
    self.phasevolume = slicer.qMRMLNodeComboBox()
    self.phasevolume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.phasevolume.selectNodeUponCreation = True
    self.phasevolume.addEnabled = True
    self.phasevolume.removeEnabled = True
    self.phasevolume.noneEnabled = True
    self.phasevolume.showHidden = False
    self.phasevolume.showChildNodeTypes = False
    self.phasevolume.setMRMLScene( slicer.mrmlScene )
    self.phasevolume.setToolTip("Select the phase image")
    parametersFormLayout.addRow("Phase Image: ", self.phasevolume)
   
    #
    # True phase points (vtkMRMLMarkupsFiducialNode)
    #
    self.truePhasePointSelector = slicer.qMRMLNodeComboBox()
    self.truePhasePointSelector.nodeTypes = ( ("vtkMRMLMarkupsFiducialNode"), "" )
    self.truePhasePointSelector.addEnabled = True
    self.truePhasePointSelector.removeEnabled = False
    self.truePhasePointSelector.noneEnabled = True
    self.truePhasePointSelector.showHidden = False
    self.truePhasePointSelector.renameEnabled = True
    self.truePhasePointSelector.showChildNodeTypes = False
    self.truePhasePointSelector.setMRMLScene( slicer.mrmlScene )
    self.truePhasePointSelector.setToolTip( "Pick up a Markups node listing a true phase point." )
    parametersFormLayout.addRow("True Phase Point: ", self.truePhasePointSelector)

    #
    # Select which scene view to track
    #
    self.sceneViewButton_red = qt.QRadioButton('Red')
    self.sceneViewButton_yellow = qt.QRadioButton('Yellow')
    self.sceneViewButton_green = qt.QRadioButton('Green')
    self.sceneViewButton_green.checked = 1

    
    layout = qt.QHBoxLayout(parametersCollapsibleButton)
    layout.addWidget(self.sceneViewButton_red)
    layout.addWidget(self.sceneViewButton_yellow)
    layout.addWidget(self.sceneViewButton_green)
    parametersFormLayout.addRow("Scene view:",layout)

    # Auto slice selecter
    self.autosliceselecterButton = qt.QPushButton("Segment Needle")
    self.autosliceselecterButton.toolTip = "Observe slice from scene viewer"
    self.autosliceselecterButton.enabled = False
    parametersFormLayout.addRow(self.autosliceselecterButton)

    realtimebutton = qt.QHBoxLayout()    
     
    #   
    # Start Real-Time Tracking 
    #
    self.trackingButton = qt.QPushButton("Start Simulated Tracking")
    self.trackingButton.toolTip = "Observe slice from scene viewer"
    self.trackingButton.enabled = False
    self.trackingButton.clicked.connect(self.SimStartTimer)
    realtimebutton.addWidget(self.trackingButton)

    self.SimTimer = qt.QTimer()
    self.SimTimer.timeout.connect(self.onRealTimeTracking)

    # Stop Real-Time Tracking
    self.stopsequence = qt.QPushButton('Stop Simulated Tracking')
    self.stopsequence.clicked.connect(self.SimStopTimer)
    realtimebutton.addWidget(self.stopsequence)
     
    parametersFormLayout.addRow("", realtimebutton)


    # Add vertical spacer
    self.layout.addStretch(1)

    ####################################
    ##                                ##
    ## Scanner Remote Control Protocol##
    ##                                ##
    ####################################

    SRCcollapsibleButton = ctk.ctkCollapsibleButton()
    SRCcollapsibleButton.text =  "Scanner Remote Control Protocol"
    self.layout.addWidget(SRCcollapsibleButton)
    SRCFormLayout = qt.QFormLayout(SRCcollapsibleButton)
    
    realtimebutton = qt.QHBoxLayout()    

    
    # FPS 
    self.fpsBox = qt.QDoubleSpinBox()
    self.fpsBox.setSingleStep(0.1)
    self.fpsBox.setMaximum(40)
    self.fpsBox.setMinimum(0.1)
    self.fpsBox.setSuffix(" FPS")
    self.fpsBox.value = 0.5
    SRCFormLayout.addRow("Update Rate:", self.fpsBox)


    # Start SRC Real-Time Tracking 
    self.SRCtrackingButton = qt.QPushButton("Start Live Tracking")
    self.SRCtrackingButton.toolTip = "Observe slice from scene viewer"
    self.SRCtrackingButton.enabled = False
    self.SRCtrackingButton.clicked.connect(self.StartTimer)
    realtimebutton.addWidget(self.SRCtrackingButton)

    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.SRCRealTimeTracking)

    # Stop Real-Time Tracking
    self.stopsequence = qt.QPushButton('Stop Live Tracking')
    self.stopsequence.clicked.connect(self.StopTimer)
    realtimebutton.addWidget(self.stopsequence)
     
    SRCFormLayout.addRow("", realtimebutton)


    # Add vertical spacer
    self.layout.addStretch(1)

    ######################
    # Advanced Parameters#
    ######################

    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    advancedCollapsibleButton.collapsed=1
    self.layout.addWidget(advancedCollapsibleButton)
    
    # Layout within the collapsible button
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
    
    #
    # 2D slice value
    #
    self.imageSliceSliderWidget = ctk.ctkSliderWidget()
    self.imageSliceSliderWidget.singleStep = 1
    self.imageSliceSliderWidget.minimum = 0
    self.imageSliceSliderWidget.maximum = 70
    self.imageSliceSliderWidget.value = 1
    self.imageSliceSliderWidget.setToolTip("Select 2D slice")
    advancedFormLayout.addRow("2D Slice ", self.imageSliceSliderWidget)
       
    #
    # Mask Threshold
    #
    self.maskThresholdWidget = ctk.ctkSliderWidget()
    self.maskThresholdWidget.singleStep = 1
    self.maskThresholdWidget.minimum = 0
    self.maskThresholdWidget.maximum = 100
    self.maskThresholdWidget.value = 20
    self.maskThresholdWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    advancedFormLayout.addRow("Mask Threshold ", self.maskThresholdWidget)

    #
    # Ridge operator filter
    #
    self.ridgeOperatorWidget = ctk.ctkSliderWidget()
    self.ridgeOperatorWidget.singleStep = 1
    self.ridgeOperatorWidget.minimum = 0
    self.ridgeOperatorWidget.maximum = 100
    self.ridgeOperatorWidget.value = 5
    self.ridgeOperatorWidget.setToolTip("set up meijering filter threshold")
    advancedFormLayout.addRow("Ridge Operator Threshold", self.ridgeOperatorWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Manual apply Button
    #
    self.applyButton = qt.QPushButton("Manual")
    self.applyButton.toolTip = "Select slice manually"
    self.applyButton.enabled = False
    advancedFormLayout.addRow(self.applyButton)

    # Refresh Apply button state
    self.onSelect()

    #
    # check box to trigger preview final processed image
    #
    self.enableprocessedimagecheckbox = qt.QCheckBox()
    self.enableprocessedimagecheckbox.checked = 0
    self.enableprocessedimagecheckbox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    advancedFormLayout.addRow("Enable Processed Image", self.enableprocessedimagecheckbox)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.trackingButton.connect('clicked(bool)', self.onRealTimeTracking)
    self.SRCtrackingButton.connect('clicked(bool)', self.SRCRealTimeTracking)
    self.autosliceselecterButton.connect('clicked(bool)', self.autosliceselecter)
    self.magnitudevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.phasevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.lastMatrix = vtk.vtkMatrix4x4()
    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.SRCRealTimeTracking) 
    self.SimTimer = qt.QTimer()
    self.SimTimer.timeout.connect(self.onRealTimeTracking)

  def SimStartTimer(self):
      self.SimTimer.start(int(1000/int(50)))
      self.counter = 0

  def SimStopTimer(self):
      self.SimTimer.stop()
      print ("Stopped Simulated Tracking...")

  def StartTimer(self):
    self.timer.start(int(1000/float(self.fpsBox.value)))
    self.counter = 0

  def StopTimer (self):
    self.timer.stop()
    print ("Stopped Live Tracking ...")

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()
    self.trackingButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()
    self.autosliceselecterButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()
    self.SRCtrackingButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()

  def getViewSelecter(self):
    viewSelecter = None
    if (self.sceneViewButton_red.checked == True):
      viewSelecter = ("Red")
    elif (self.sceneViewButton_yellow.checked ==True):
      viewSelecter = ("Yellow")
    elif (self.sceneViewButton_green.checked ==True):
      viewSelecter = ("Green")
    return viewSelecter
    
  def autosliceselecter (self):
    logic = NeedleSegmenterLogic()
    enableProcessedFlag = self.enableprocessedimagecheckbox.checked
    if (enableProcessedFlag==True):
      enablenumber = 1
      
    viewSelecter = self.getViewSelecter()
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.needlefinder(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), self.truePhasePointSelector.currentNode(), maskThreshold, ridgeOperator, viewSelecter)

  def onRealTimeTracking(self):
    self.counter = 0
    logic = NeedleSegmenterLogic()
    
    viewSelecter = self.getViewSelecter()
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.realtime(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), self.truePhasePointSelector.currentNode(), maskThreshold, ridgeOperator, viewSelecter, self.counter, self.lastMatrix)

  def SRCRealTimeTracking(self):
  #set observer node so that i can the image as it updates
    self.counter = 0
    logic = NeedleSegmenterLogic()
    viewSelecter = self.getViewSelecter()      
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.SRCrealtime(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), self.truePhasePointSelector.currentNode(), maskThreshold, ridgeOperator, viewSelecter, self.counter)

  def onApplyButton(self):
    logic = NeedleSegmenterLogic()
    imageSlice = self.imageSliceSliderWidget.value
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.run(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), self.truePhasePointSelector.currentNode(), imageSlice, maskThreshold, ridgeOperator)

  def onReload(self,moduleName="NeedleSegmenter"):
    """Generic reload method for any scripted module.
    ModuleWizard will subsitute correct default moduleName.
    """
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)

    
class NeedleSegmenterLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.cliParamNode = None

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def detectNeedle(self, magnitudevolume, phasevolume, truePhasePoint, maskThreshold, ridgeOperator, slice_index):

    #magnitude volume
    magn_imageData = magnitudevolume.GetImageData()
    magn_rows, magn_cols, magn_zed = magn_imageData.GetDimensions()
    magn_scalars = magn_imageData.GetPointData().GetScalars()
    magn_imageOrigin = magnitudevolume.GetOrigin()
    magn_imageSpacing = magnitudevolume.GetSpacing()
    magn_matrix = vtk.vtkMatrix4x4()
    magnitudevolume.GetIJKToRASMatrix(magn_matrix)
    # magnitudevolume.CreateDefaultDisplayNodes()

    # phase volume
    phase_imageData = phasevolume.GetImageData()
    phase_rows, phase_cols, phase_zed = phase_imageData.GetDimensions()
    phase_scalars = phase_imageData.GetPointData().GetScalars()
    
    #Convert vtk to numpy
    magn_array = numpy_support.vtk_to_numpy(magn_scalars)
    numpy_magn = magn_array.reshape(magn_zed, magn_rows, magn_cols)
    phase_array = numpy_support.vtk_to_numpy(phase_scalars)
    numpy_phase = phase_array.reshape(phase_zed, phase_rows, phase_cols)
    
    # slice = int(slice_number)  
    # slice = (slice_index)
    # maskThreshold = int(maskThreshold)

    #2D Slice Selector
    ### 3 3D values are : numpy_magn , numpy_phase, mask
    numpy_magn = numpy_magn[slice_index,:,:]
    numpy_phase = numpy_phase[slice_index,:,:]
    #mask = mask[slice,:,:]
    numpy_magn_sliced = numpy_magn.astype(np.uint8)

    #mask thresholding 
    img = cv2.pyrDown(numpy_magn_sliced)
    _, threshed = cv2.threshold(numpy_magn_sliced, maskThreshold, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find maximum contour and draw   
    cmax = max(contours, key = cv2.contourArea) 
    epsilon = 0.002 * cv2.arcLength(cmax, True)
    approx = cv2.approxPolyDP(cmax, epsilon, True)
    cv2.drawContours(numpy_magn_sliced, [approx], -1, (0, 255, 0), 3)

    width, height = numpy_magn_sliced.shape

    #fill maximum contour and draw   
    mask = np.zeros( [width, height, 3],dtype=np.uint8 )
    cv2.fillPoly(mask, pts =[cmax], color=(255,255,255))
    mask = mask[:,:,0]

    #phase_cropped
    phase_cropped = cv2.bitwise_and(numpy_phase, numpy_phase, mask=mask)
    phase_cropped =  np.expand_dims(phase_cropped, axis=0)

    node = slicer.vtkMRMLScalarVolumeNode()
    node.SetName('phase_cropped')
    slicer.mrmlScene.AddNode(node)

    slicer.util.updateVolumeFromArray(node, phase_cropped)
    node.SetOrigin(magn_imageOrigin)
    node.SetSpacing(magn_imageSpacing)
    node.SetIJKToRASDirectionMatrix(magn_matrix)

    unwrapped_phase = slicer.vtkMRMLScalarVolumeNode()
    unwrapped_phase.SetName('unwrapped_phase')
    slicer.mrmlScene.AddNode(unwrapped_phase)

    #
    # Run phase unwrapping module
    #
    parameter_name = slicer.mrmlScene.GetNodeByID('vtkMRMLCommandLineModuleNode1')
    
    if parameter_name is None:
        slicer.cli.createNode(slicer.modules.phaseunwrapping)
    else:
        pass
    cli_input = slicer.util.getFirstNodeByName('phase_cropped')
    cli_output = slicer.util.getNode('unwrapped_phase')

    cli_params = {'inputVolume': cli_input, 'outputVolume': cli_output, 'truePhase': truePhasePoint}
    self.cliParamNode = slicer.cli.runSync(slicer.modules.phaseunwrapping, node=self.cliParamNode, parameters=cli_params)

    pu_imageData = unwrapped_phase.GetImageData()
    pu_rows, pu_cols, pu_zed = pu_imageData.GetDimensions()
    pu_scalars = pu_imageData.GetPointData().GetScalars()
    pu_NumpyArray = numpy_support.vtk_to_numpy(pu_scalars)
    phaseunwrapped = pu_NumpyArray.reshape(pu_zed, pu_rows, pu_cols)

    # for debug
    self.phaseunwrapped_numpy = pu_NumpyArray.reshape(pu_cols,pu_rows)

    #Delete unwrapped_phase after I get the information from it 
    # delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('Phase Unwrapping')
    # slicer.mrmlScene.RemoveNode(delete_unwrapped)

    I = phaseunwrapped.squeeze()
    A = np.fft.fft2(I)
    A1 = np.fft.fftshift(A)

    # Image size
    [M, N] = A.shape

    # filter size parameter
    R = 10

    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)

    [X, Y] = np.meshgrid(X, Y)
    Cx = 0.5 * N
    Cy = 0.5 * M
    Lo = np.exp(-(((X - Cx) ** 2) + ((Y - Cy) ** 2)) / ((2 * R) ** 2))
    Hi = 1 - Lo

    J = A1 * Lo
    J1 = np.fft.ifftshift(J)
    B1 = np.fft.ifft2(J1)

    K = A1 * Hi
    K1 = np.fft.ifftshift(K)
    B2 = np.fft.ifft2(K1)
    B2 = np.real(B2)

    #Remove border  for false positive
    border_size = 20
    top, bottom, left, right = [border_size] * 4
    mask_borderless = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
    
    kernel = np.ones((5, 5), np.uint8)
    mask_borderless = cv2.erode(mask_borderless, kernel, iterations=5)
    mask_borderless = ndimage.binary_fill_holes(mask_borderless).astype(np.uint8)
    x, y = mask_borderless.shape
    mask_borderless = mask_borderless[0 + border_size:y - border_size, 0 + border_size:x - border_size]

    B2 = cv2.bitwise_and(B2, B2, mask=mask_borderless)

    # for debug
    self.mask_borderless = mask_borderless

    # ridgeOperator = int(ridgeOperator)
    meiji = sato(B2, sigmas=(ridgeOperator, ridgeOperator), black_ridges=True)

    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(meiji)
    
    result2 = np.reshape(meiji, meiji.shape[0]*meiji.shape[1])
    
    ids = np.argpartition(result2, -51)[-51:]
    sort = ids[np.argsort(result2[ids])[::-1]]
    
    (y1,x1) = np.unravel_index(sort[0], meiji.shape) # best match

    self.meiji = meiji

    point = (x1,y1)
    coords = [x1,y1,slice_index]
    circle1 = plt.Circle(point,2,color='red')

    self.x1 = x1
    self.y1 = y1

    # Find or create MRML transform node
    transformNode = None
    try:
      transformNode = slicer.util.getNode('TipTransform')
    except slicer.util.MRMLNodeNotFoundException as exc:
      transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
      transformNode.SetName("TipTransform")
      
    transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)
    
    # Fiducial Creation
    fidNode1 = None
    try: 
      fidNode1 = slicer.util.getNode('needle_tip')
    except slicer.util.MRMLNodeNotFoundException as exc:
      fidNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "needle_tip")

    fidNode1.RemoveAllMarkups()
      
    #  fidNode1.CreateDefaultDisplayNodes()
    #  fidNode1.SetMaximumNumberOfControlPoints(1) 

    fidNode1.AddFiducialFromArray(coords)
    fidNode1.SetAndObserveTransformNodeID(transformNode.GetID())

    ###TODO: dont delete the volume after use. create a checkpoint to update on only one volume
    delete_wrapped = slicer.mrmlScene.GetFirstNodeByName('phase_cropped')
    slicer.mrmlScene.RemoveNode(delete_wrapped)
    delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('unwrapped_phase')
    slicer.mrmlScene.RemoveNode(delete_unwrapped)


    #print ("Needle tip location",y1,x1)
    #self.counter = 0
    return True


  def findSliceIndex(self, viewSelecter):
    
    ## Find Slice location
    sliceNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
    fov_0,fov_1,fov_2 = sliceNode.GetFieldOfView()
    layoutManager = slicer.app.layoutManager()
    slice_index = None
    offset = 0.0
    for sliceViewName in [''+ str(viewSelecter)]:
      sliceWidget = layoutManager.sliceWidget(sliceViewName)
      sliceWidgetLogic = sliceWidget.sliceLogic()
      offset = sliceWidgetLogic.GetSliceOffset()
      slice_index = sliceWidgetLogic.GetSliceIndexFromOffset(offset)
      slice_index = (slice_index - 1)
      # offsets.append(offset)
      
    fov = [fov_0,fov_1,fov_2]
      
    return (slice_index, sliceNode, fov, offset)
  
  
  def SRCrealtime(self, magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator,viewSelecter, counter):
    
    (slice_index, sliceNode, fov, offset) = self.findSliceIndex(viewSelecter)
    
    self.detectNeedle(magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator, slice_index)

    ## Setting the Slice view 
    slice_logic = slicer.app.layoutManager().sliceWidget(''+ str(viewSelecter)).sliceLogic()
    slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(magnitudevolume.GetID())


    # view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
    sliceNode.SetFieldOfView(fov[0],fov[1],fov[2])
    sliceNode.SetSliceOffset(offset)

    
  def realtime(self, magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator,viewSelecter, counter, lastMatrix):

    ## Counter is disabled for current use, only updates when slice view changes
    inputransform = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter)).GetXYToRAS()

    (slice_index, sliceNode, fov, offset) = self.findSliceIndex(viewSelecter)
      
    if (not self.CompareMatrices(lastMatrix, inputransform) or counter >= 20) :

      self.detectNeedle(magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator, slice_index)
     
      ## Setting the Slice view 
      slice_logic = slicer.app.layoutManager().sliceWidget(''+ str(viewSelecter)).sliceLogic()
      slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(magnitudevolume.GetID())

      # view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
      sliceNode.SetFieldOfView(fov[0],fov[1],fov[2])
      sliceNode.SetSliceOffset(offset)
      
      # self.lastMatrix = view_selecter.GetXYToRAS()
      self.counter = 0
      lastMatrix.DeepCopy(inputransform)
      return True
   
    else: 
      counter = counter + 1

      
  def CompareMatrices(self, m, n):
    for i in range(0,4):
      for j in range(0,4):
        if m.GetElement(i,j) != n.GetElement(i,j):
          print ("HII Processing new slice ...")
          return False
    return True


  def needlefinder(self, magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator, viewSelecter):

    (slice_index, sliceNode, fov, offset) = self.findSliceIndex(viewSelecter)

    self.detectNeedle(magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator, slice_index)

    ## Setting the Slice view 
    slice_logic = slicer.app.layoutManager().sliceWidget(''+ str(viewSelecter)).sliceLogic()
    slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(magnitudevolume.GetID())

    # view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
    sliceNode.SetFieldOfView(fov[0],fov[1],fov[2])
    sliceNode.SetSliceOffset(offset)
    # if (viewSelecter == "Red"): 
    #   view_selecter.SetSliceOffset(z_ras)
    # elif (viewSelecter == "Yellow"):
    #   view_selecter.SetSliceOffset(x_ras)
    # elif (viewSelecter == "Green"):
    #   view_selecter.SetSliceOffset(y_ras)


  def run(self, magnitudevolume , phasevolume, truePhasePoint, imageSlice, maskThreshold, ridgeOperator):

    slice_index = int(imageSlice)

    self.detectNeedle(magnitudevolume , phasevolume, truePhasePoint, maskThreshold, ridgeOperator, slice_index)

    fig, axs = plt.subplots(1,3)
    fig.suptitle('Needle Tracking')

    axs[0].imshow(self.phaseunwrapped_numpy, cmap='gray')
    axs[1].imshow(self.meiji, cmap='hsv')
    axs[1].axis('off')
    axs[2].imshow(self.mask_borderless,zorder=1, cmap='gray')
    axs[2].scatter(self.x1,self.y1,zorder=2, s=1)
    axs[2].axis('off')
    plt.savefig('mygraph.png')

    return True

class NeedleSegmenterTest(ScriptedLoadableModuleTest):

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_NeedleSegmenter1()

  def test_NeedleSegmenter1(self):

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = NeedleSegmenterLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
