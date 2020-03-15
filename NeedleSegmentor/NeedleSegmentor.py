##adding real time needle tracking
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import math
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import meijering, frangi, sato
import cv2
import tempfile


class NeedleSegmentor(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "2D Needle Segmenter "
    self.parent.categories = ["Filtering"]
    self.parent.dependencies = []
    self.parent.contributors = ["Ahmed Mahran (BWH)"]
    self.parent.helpText = """This is a 2D needle segmenter module used to localize needle tip in the MRI image. Input requirment: 
    Magnitude image and Phase Image. Uses Prelude phase unwrapping algorithm. """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """"""

class NeedleSegmentorWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

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
    # 2D slice value
    #
    self.imageSliceSliderWidget = ctk.ctkSliderWidget()
    self.imageSliceSliderWidget.singleStep = 1
    self.imageSliceSliderWidget.minimum = 0
    self.imageSliceSliderWidget.maximum = 70
    self.imageSliceSliderWidget.value = 1
    self.imageSliceSliderWidget.setToolTip("Select 2D slice")
    parametersFormLayout.addRow("2D Slice ", self.imageSliceSliderWidget)

    #
    # Mask Threshold
    #
    self.maskThresholdWidget = ctk.ctkSliderWidget()
    self.maskThresholdWidget.singleStep = 1
    self.maskThresholdWidget.minimum = 0
    self.maskThresholdWidget.maximum = 100
    self.maskThresholdWidget.value = 20
    self.maskThresholdWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Mask Threshold ", self.maskThresholdWidget)

    #
    # Ridge operator filter
    #
    self.ridgeOperatorWidget = ctk.ctkSliderWidget()
    self.ridgeOperatorWidget.singleStep = 1
    self.ridgeOperatorWidget.minimum = 0
    self.ridgeOperatorWidget.maximum = 100
    self.ridgeOperatorWidget.value = 5
    self.ridgeOperatorWidget.setToolTip("set up meijering filter threshold")
    parametersFormLayout.addRow("Ridge Operator Threshold", self.ridgeOperatorWidget)

    #
    # FPS 
    #
    self.fpsBox = qt.QSpinBox()
    self.fpsBox.setSingleStep(1)
    self.fpsBox.setMaximum(144)
    self.fpsBox.setMinimum(1)
    self.fpsBox.setSuffix(" FPS")
    self.fpsBox.value = 30
    parametersFormLayout.addRow("Update Rate:", self.fpsBox)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Select which scene view to track
    #
    self.sceneViewButton_red = qt.QRadioButton('Red')
    self.sceneViewButton_red.setFixedWidth(120)

    self.sceneViewButton_yellow = qt.QRadioButton('Yellow')
    self.sceneViewButton_yellow.setFixedWidth(120)

    self.sceneViewButton_green = qt.QRadioButton('Green')
    self.sceneViewButton_green.checked = 1
    self.sceneViewButton_green.setFixedWidth(120)

    
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

    #   
    # Start Real-Time Tracking 
    #
    self.trackingButton = qt.QPushButton("Start Real-Time Tracking")
    self.trackingButton.toolTip = "Observe slice from scene viewer"
    self.trackingButton.enabled = False
    self.trackingButton.clicked.connect(self.StartTimer)
    parametersFormLayout.addRow(self.trackingButton)

    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.onRealTimeTracking)

    # Stop Real-Time Tracking
    self.stopsequence = qt.QPushButton('Stop Realtime Tracking')
    self.stopsequence.clicked.connect(self.StopTimer)
    parametersFormLayout.addRow(self.stopsequence)


    # Add vertical spacer
    self.layout.addStretch(1)


    #
    # Advanced Parameters
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    self.layout.addWidget(advancedCollapsibleButton)

    # Layout within the dummy collapsible button
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
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Manual")
    self.applyButton.toolTip = "Select slice manually"
    self.applyButton.enabled = False
    advancedFormLayout.addRow(self.applyButton)

    # Refresh Apply button state
    self.onSelect()

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.trackingButton.connect('clicked(bool)', self.onRealTimeTracking)
    self.autosliceselecterButton.connect('clicked(bool)', self.autosliceselecter)
    self.magnitudevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.phasevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.lastMatrix = vtk.vtkMatrix4x4()
    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.onRealTimeTracking)


  def StartTimer(self):
    self.timer.start(int(1000/int(self.fpsBox.value)))
    self.counter = 0

  def StopTimer (self):
    self.timer.stop()
    
  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()
    self.trackingButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()
    self.autosliceselecterButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()

  def autosliceselecter (self):
    logic = NeedleSegmentorLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    if (self.sceneViewButton_red.checked == True):
      viewSelecter = ("Red")
      self.z_axis = (0)
    elif (self.sceneViewButton_yellow.checked ==True):
      viewSelecter = ("Yellow")
      self.z_axis = 1
    elif (self.sceneViewButton_green.checked ==True):
      viewSelecter = ("Green")
      self.z_axis = (2)

    imageSlice = self.imageSliceSliderWidget.value
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.needlefinder(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), imageSlice, maskThreshold, ridgeOperator, self.z_axis,
    viewSelecter)

  def onRealTimeTracking(self):
    self.counter = 0
    logic = NeedleSegmentorLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    if (self.sceneViewButton_red.checked == True):
      viewSelecter = ("Red")
      self.z_axis = (0)
    elif (self.sceneViewButton_yellow.checked ==True):
      viewSelecter = ("Yellow")
      self.z_axis = 1
    elif (self.sceneViewButton_green.checked ==True):
      viewSelecter = ("Green")
      self.z_axis = (2)
    
    imageSlice = self.imageSliceSliderWidget.value
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.realtime(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), imageSlice, maskThreshold, ridgeOperator, self.z_axis,
    viewSelecter, self.counter, self.lastMatrix)


  def onApplyButton(self):
    logic = NeedleSegmentorLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageSlice = self.imageSliceSliderWidget.value
    maskThreshold = self.maskThresholdWidget.value
    ridgeOperator = self.ridgeOperatorWidget.value
    logic.run(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), imageSlice, maskThreshold, ridgeOperator, enableScreenshotsFlag)

class NeedleSegmentorLogic(ScriptedLoadableModuleLogic):

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


  def realtime(self, magnitudevolume , phasevolume, imageSlice, maskThreshold, ridgeOperator,z_axis,viewSelecter, counter, lastMatrix):

    ## Counter is disabled for current use, only updates when slice view changes
    inputransform = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter)).GetXYToRAS()

    if (not self.CompareMatrices(lastMatrix, inputransform) or counter >= 20) :
     
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


      ## Find Slice location
      view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
      fov_0,fov_1,fov_2 = view_selecter.GetFieldOfView()
      layoutManager = slicer.app.layoutManager()
      for sliceViewName in [''+ str(viewSelecter)]:
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        sliceWidgetLogic = sliceWidget.sliceLogic()
        offset = sliceWidgetLogic.GetSliceOffset()
        slice_index = sliceWidgetLogic.GetSliceIndexFromOffset(offset)
        slice_index = (slice_index - 1)
        # offsets.append(offset)

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
      cli_input = slicer.util.getFirstNodeByName('phase_cropped')
      cli_output = slicer.util.getNode('unwrapped_phase')
      cli_params = {'inputVolume': cli_input, 'outputVolume': cli_output}
      slicer.cli.runSync(slicer.modules.phaseunwrapping, None, cli_params)


      pu_imageData = unwrapped_phase.GetImageData()
      pu_rows, pu_cols, pu_zed = pu_imageData.GetDimensions()
      pu_scalars = pu_imageData.GetPointData().GetScalars()
      pu_NumpyArray = numpy_support.vtk_to_numpy(pu_scalars)
      phaseunwrapped = pu_NumpyArray.reshape(pu_zed, pu_rows, pu_cols)


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
      mask_borderless = cv2.erode(mask_borderless, kernel, iterations=2)
      mask_borderless = ndimage.binary_fill_holes(mask_borderless).astype(np.uint8)
      x, y = mask_borderless.shape
      mask_borderless = mask_borderless[0 + border_size:y - border_size, 0 + border_size:x - border_size]

      B2 = cv2.bitwise_and(B2, B2, mask=mask_borderless)

      # ridgeOperator = int(ridgeOperator)
      meiji = sato(B2, sigmas=(ridgeOperator, ridgeOperator), black_ridges=True)

      #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(meiji)
      
      result2 = np.reshape(meiji, meiji.shape[0]*meiji.shape[1])
      
      ids = np.argpartition(result2, -51)[-51:]
      sort = ids[np.argsort(result2[ids])[::-1]]
      
      (y1,x1) = np.unravel_index(sort[0], meiji.shape) # best match

      point = (x1,y1)
      coords = [x1,y1,slice_index]
      circle1 = plt.Circle(point,2,color='red')

      # Create MRML transform node
      
      transforms = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLLinearTransformNode','Transform')
      nbTransforms = transforms.GetNumberOfItems()
      if (nbTransforms >= 1): 
        for i in range(nbTransforms):
          transformNode = slicer.util.getNode('Transform')
          transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

      else:
        # transformNode = slicer.mrmlScene.CreateNodeByClass ('vtkMRMLAnnotationFiducialNode')
        transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        transformNode.SetName("Transform")
        transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

      # Fiducial Creation
      nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsFiducialNode')
      nbNodes = nodes.GetNumberOfItems()
      if (nbNodes >= 1): 
        for i in range(nbNodes):
          fidNode1 = slicer.util.getNode('needle_tip')
          ## to view mutiple fiducial comment the line below
          fidNode1.RemoveAllMarkups()
      else:
        fidNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "needle_tip")
      #  fidNode1.CreateDefaultDisplayNodes()
      #  fidNode1.SetMaximumNumberOfControlPoints(1) 

      fidNode1.AddFiducialFromArray(coords)
      fidNode1.SetAndObserveTransformNodeID(transformNode.GetID())

      ###TODO: dont delete the volume after use. create a checkpoint to update on only one volume
      delete_wrapped = slicer.mrmlScene.GetFirstNodeByName('phase_cropped')
      slicer.mrmlScene.RemoveNode(delete_wrapped)
      delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('unwrapped_phase')
      slicer.mrmlScene.RemoveNode(delete_unwrapped)


      ## Setting the Slice view 
      slice_logic = slicer.app.layoutManager().sliceWidget(''+ str(viewSelecter)).sliceLogic()
      slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(magnitudevolume.GetID())

      # view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
      view_selecter.SetFieldOfView(fov_0,fov_1,fov_2)
      view_selecter.SetSliceOffset(offset)
      
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
          print ("Processing new slice ...")
          return False
    return True


  def needlefinder(self, magnitudevolume , phasevolume, imageSlice, maskThreshold, ridgeOperator,z_axis,viewSelecter, enableScreenshots=0):

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


    ## Find Slice location
    #TODO: offset only gives the RAS of the center of the image, this will not for reformated images with
    ## oblique slice views. 
    view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
    fov_0,fov_1,fov_2 = view_selecter.GetFieldOfView()
    layoutManager = slicer.app.layoutManager()
    offsets = []
    for sliceViewName in [''+ str(viewSelecter)]:
      sliceWidget = layoutManager.sliceWidget(sliceViewName)
      sliceWidgetLogic = sliceWidget.sliceLogic()
      offset = sliceWidgetLogic.GetSliceOffset()
      slice_index = sliceWidgetLogic.GetSliceIndexFromOffset(offset)
      slice_index = (slice_index - 1)
      offsets.append(offset)

    ##LEGACY 
    print ("Slice Number:",slice_index)
    # z_ras,x_ras,y_ras = offsets
    # z_index, x_index, y_index = slice_index

    # Inputs
    # markupsIndex = 0

    # # Get point coordinate in RAS
    # point_Ras = [x_ras, y_ras, z_ras, 1]
    # #markupsNode.GetNthFiducialWorldCoordinates(markupsIndex, point_Ras)
    # # If volume node is transformed, apply that transform to get volume's RAS coordinates
    # transformRasToVolumeRas = vtk.vtkGeneralTransform()
    # slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, magnitudevolume.GetParentTransformNode(), transformRasToVolumeRas)
    # point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_Ras[0:3])

    # # Get voxel coordinates from physical coordinates
    # volumeRasToIjk = vtk.vtkMatrix4x4()
    # magnitudevolume.GetRASToIJKMatrix(volumeRasToIjk)
    # point_Ijk = [0, 0, 0, 1]
    # volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
    # point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]

    # # Print output
    
    # x_ijk,y_ijk,slice_number = point_Ijk

    #Convert vtk to numpy
    magn_array = numpy_support.vtk_to_numpy(magn_scalars)
    numpy_magn = magn_array.reshape(magn_zed, magn_rows, magn_cols)
    phase_array = numpy_support.vtk_to_numpy(phase_scalars)
    numpy_phase = phase_array.reshape(phase_zed, phase_rows, phase_cols)

    # slice = int(slice_number)  
    # slice = (slice_index)
    maskThreshold = int(maskThreshold)

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
    cli_input = slicer.util.getFirstNodeByName('phase_cropped')
    cli_output = slicer.util.getNode('unwrapped_phase')
    cli_params = {'inputVolume': cli_input, 'outputVolume': cli_output}
    slicer.cli.runSync(slicer.modules.phaseunwrapping, None, cli_params)


    pu_imageData = unwrapped_phase.GetImageData()
    pu_rows, pu_cols, pu_zed = pu_imageData.GetDimensions()
    pu_scalars = pu_imageData.GetPointData().GetScalars()
    pu_NumpyArray = numpy_support.vtk_to_numpy(pu_scalars)
    phaseunwrapped = pu_NumpyArray.reshape(pu_zed, pu_rows, pu_cols)


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
    mask_borderless = cv2.erode(mask_borderless, kernel, iterations=2)
    mask_borderless = ndimage.binary_fill_holes(mask_borderless).astype(np.uint8)
    x, y = mask_borderless.shape
    mask_borderless = mask_borderless[0 + border_size:y - border_size, 0 + border_size:x - border_size]

    B2 = cv2.bitwise_and(B2, B2, mask=mask_borderless)

    ridgeOperator = int(ridgeOperator)
    meiji = sato(B2, sigmas=(ridgeOperator, ridgeOperator), black_ridges=True)

    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(meiji)
    
    result2 = np.reshape(meiji, meiji.shape[0]*meiji.shape[1])
    
    ids = np.argpartition(result2, -51)[-51:]
    sort = ids[np.argsort(result2[ids])[::-1]]
    
    (y1,x1) = np.unravel_index(sort[0], meiji.shape) # best match

    point = (x1,y1)
    coords = [x1,y1,slice_index]
    circle1 = plt.Circle(point,2,color='red')

    # Create MRML transform node
    
    transforms = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLLinearTransformNode','Transform')
    nbTransforms = transforms.GetNumberOfItems()
    if (nbTransforms >= 1): 
      for i in range(nbTransforms):
        transformNode = slicer.util.getNode('Transform')
        transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

    else:
      # transformNode = slicer.mrmlScene.CreateNodeByClass ('vtkMRMLAnnotationFiducialNode')
      transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
      transformNode.SetName("Transform")
      transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

    # Fiducial Creation
    nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsFiducialNode')
    nbNodes = nodes.GetNumberOfItems()
    if (nbNodes >= 1): 
      for i in range(nbNodes):
        fidNode1 = slicer.util.getNode('needle_tip')
        ## to view mutiple fiducial comment the line below
        fidNode1.RemoveAllMarkups()
    else:
     fidNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "needle_tip")
    #  fidNode1.CreateDefaultDisplayNodes()
    #  fidNode1.SetMaximumNumberOfControlPoints(1) 

    fidNode1.AddFiducialFromArray(coords)
    fidNode1.SetAndObserveTransformNodeID(transformNode.GetID())

    



    ###TODO: dont delete the volume after use. create a checkpoint to update on only one volume
    delete_wrapped = slicer.mrmlScene.GetFirstNodeByName('phase_cropped')
    slicer.mrmlScene.RemoveNode(delete_wrapped)
    delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('unwrapped_phase')
    slicer.mrmlScene.RemoveNode(delete_unwrapped)


    ## Setting the Slice view 
    slice_logic = slicer.app.layoutManager().sliceWidget(''+ str(viewSelecter)).sliceLogic()
    slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(magnitudevolume.GetID())

    # view_selecter = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+ str(viewSelecter))
    view_selecter.SetFieldOfView(fov_0,fov_1,fov_2)
    view_selecter.SetSliceOffset(offset)
    # if (viewSelecter == "Red"): 
    #   view_selecter.SetSliceOffset(z_ras)
    # elif (viewSelecter == "Yellow"):
    #   view_selecter.SetSliceOffset(x_ras)
    # elif (viewSelecter == "Green"):
    #   view_selecter.SetSliceOffset(y_ras)

#      
#    
#    fig, axs = plt.subplots(1,2)
#    fig.suptitle('Needle Tracking')
#    axs[0].imshow(numpy_magn, cmap='gray')
#    axs[0].set_title('Magnitude + Tracked')
#    axs[0].add_artist(circle1)
#    axs[0].axis('off')
#    axs[1].set_title('Processed Image')
#    axs[1].imshow(meiji, cmap='hsv')
#    axs[1].axis('off')
#    plt.savefig('mygraph.png')
#    
    # return True



  def run(self, magnitudevolume , phasevolume, imageSlice, maskThreshold, ridgeOperator,z_axis, enableScreenshots=0):

    #magnitude volume
    magn_imageData = magnitudevolume.GetImageData()
    magn_rows, magn_cols, magn_zed = magn_imageData.GetDimensions()
    magn_scalars = magn_imageData.GetPointData().GetScalars()
    magn_imageOrigin = magnitudevolume.GetOrigin()
    print (magn_imageOrigin)
    magn_imageSpacing = magnitudevolume.GetSpacing()
    print(magn_imageSpacing)
    magn_matrix = vtk.vtkMatrix4x4()
    magnitudevolume.GetIJKToRASMatrix(magn_matrix)
    # magnitudevolume.CreateDefaultDisplayNodes()


    # phase volume
    phase_imageData = phasevolume.GetImageData()
    phase_rows, phase_cols, phase_zed = phase_imageData.GetDimensions()
    phase_scalars = phase_imageData.GetPointData().GetScalars()
    # imageOrigin = phasevolume.GetOrigin()
    # imageSpacing = phasevolume.GetSpacing()
    # phase_matrix = vtk.vtkMatrix4x4()
    # phasevolume.GetIJKToRASDirectionMatrix(phase_matrix)

    
    if (z_axis == 1):
      scene_viewer = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
      # element = scene_viewer.GetXYToRAS()
      element = element.GetSliceOffset()
      
      


    #Convert vtk to numpy
    magn_array = numpy_support.vtk_to_numpy(magn_scalars)
    numpy_magn = magn_array.reshape(magn_zed, magn_rows, magn_cols)
    phase_array = numpy_support.vtk_to_numpy(phase_scalars)
    numpy_phase = phase_array.reshape(phase_zed, phase_rows, phase_cols)

    slice = int(imageSlice)  
    maskThreshold = int(maskThreshold)

    #2D Slice Selector
    ### 3 3D values are : numpy_magn , numpy_phase, mask
    numpy_magn = numpy_magn[slice,:,:]
    numpy_phase = numpy_phase[slice,:,:]
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
    cli_input = slicer.util.getFirstNodeByName('phase_cropped')
    cli_output = slicer.util.getNode('unwrapped_phase')
    cli_params = {'inputVolume': cli_input, 'outputVolume': cli_output}
    slicer.cli.runSync(slicer.modules.phaseunwrapping, None, cli_params)


    pu_imageData = unwrapped_phase.GetImageData()
    pu_rows, pu_cols, pu_zed = pu_imageData.GetDimensions()
    pu_scalars = pu_imageData.GetPointData().GetScalars()
    pu_NumpyArray = numpy_support.vtk_to_numpy(pu_scalars)
    phaseunwrapped = pu_NumpyArray.reshape(pu_zed, pu_rows, pu_cols)


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
    mask_borderless = cv2.erode(mask_borderless, kernel, iterations=2)
    mask_borderless = ndimage.binary_fill_holes(mask_borderless).astype(np.uint8)
    x, y = mask_borderless.shape
    mask_borderless = mask_borderless[0 + border_size:y - border_size, 0 + border_size:x - border_size]

    B2 = cv2.bitwise_and(B2, B2, mask=mask_borderless)

    ridgeOperator = int(ridgeOperator)
    meiji = sato(B2, sigmas=(ridgeOperator, ridgeOperator), black_ridges=True)

    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(meiji)
    
    result2 = np.reshape(meiji, meiji.shape[0]*meiji.shape[1])
    
    ids = np.argpartition(result2, -51)[-51:]
    sort = ids[np.argsort(result2[ids])[::-1]]
    
    (y1,x1) = np.unravel_index(sort[0], meiji.shape) # best match

    point = (x1,y1)
    coords = [x1,y1,slice]
    circle1 = plt.Circle(point,2,color='red')

    # Create MRML transform node
    
    transforms = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLLinearTransformNode','Transform')
    nbTransforms = transforms.GetNumberOfItems()
    if (nbTransforms >= 1): 
      for i in range(nbTransforms):
        transformNode = slicer.util.getNode('Transform')
        transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

    else:
      # transformNode = slicer.mrmlScene.CreateNodeByClass ('vtkMRMLAnnotationFiducialNode')
      transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
      transformNode.SetName("Transform")
      transformNode.SetAndObserveMatrixTransformToParent(magn_matrix)

    # Fiducial Creation
    nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsFiducialNode')
    nbNodes = nodes.GetNumberOfItems()
    if (nbNodes >= 1): 
      for i in range(nbNodes):
        fidNode1 = slicer.util.getNode('needle_tip')
        ## to view mutiple fiducial comment the line below
        fidNode1.RemoveAllMarkups()
    else:
     fidNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "needle_tip")
    #  fidNode1.CreateDefaultDisplayNodes()
    #  fidNode1.SetMaximumNumberOfControlPoints(1) 


    fidNode1.AddFiducialFromArray(coords)
    fidNode1.SetAndObserveTransformNodeID(transformNode.GetID())

    # magn_imageOriginr, magn_imageOrigina, magn_imageOrigins = magnitudevolume.GetOrigin()
    # magn_imageSpacingr, magn_imageSpacinga, magn_imageSpacings = magnitudevolume.GetSpacing()

    # #dev/ delete once done
    # print("imageorigin: ", magn_imageOriginr, magn_imageOrigina, magn_imageOrigins)
    # print("imageSpacing: ", magn_imageSpacingr, magn_imageSpacinga, magn_imageSpacings)
    # print (x1, y1)

    #x,y = np.split(maxLoc, [-1], 0)
    #### RAS
    # R_loc = (magn_imageOriginr)-(x1*magn_imageSpacinga)
    # A_loc = (magn_imageOrigina)+(slice*magn_imageSpacinga)
    # S_loc = (magn_imageOrigins)-(y1*magn_imageSpacingr)
    # ras = (R_loc,A_loc,S_loc)


    # nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLAnnotationFiducialNode')
    # nbNodes = nodes.GetNumberOfItems()
    # if (nbNodes >= 1): 
    #   for i in range(nbNodes):
    #     # pass
    #     fiducial = slicer.util.getNode('needle_tip')
    #     # node = nodes.GetItemAsObject(i)
    #     # name = node.GetName()
    #     #        
    # else:
    #   fiducial = slicer.mrmlScene.CreateNodeByClass ('vtkMRMLAnnotationFiducialNode')
    #   fiducial.SetName('needle_tip')
    #   fiducial.Initialize(slicer.mrmlScene)
    #   fiducial.SetAttribute('TemporaryFiducial', '1')
    #   fiducial.SetLocked(True)
    #   displayNode = fiducial.GetDisplayNode()
    #   displayNode.SetGlyphScale(2)
    #   displayNode.SetColor(1,1,0)
    #   textNode = fiducial.GetAnnotationTextDisplayNode()
    #   textNode.SetTextScale(4)
    #   textNode.SetColor(1, 1, 0)

    # fiducial.SetFiducialCoordinates(ras)

    ###TODO: dont delete the volume after use. create a checkpoint to update on only one volume
    delete_wrapped = slicer.mrmlScene.GetFirstNodeByName('phase_cropped')
    slicer.mrmlScene.RemoveNode(delete_wrapped)
    delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('unwrapped_phase')
    slicer.mrmlScene.RemoveNode(delete_unwrapped)
    
    
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Needle Tracking')
    axs[0].imshow(meiji, cmap='gray')
    axs[0].set_title('Magnitude + Tracked')
    axs[0].add_artist(circle1)
    axs[0].axis('off')
    axs[1].set_title('Processed Phase Image')
    axs[1].imshow(meiji, cmap='hsv')
    axs[1].axis('off')
    plt.savefig('mygraph.png')

    return True

class NeedleSegmentorTest(ScriptedLoadableModuleTest):

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_NeedleSegmentor1()

  def test_NeedleSegmentor1(self):

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
    logic = NeedleSegmentorLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
