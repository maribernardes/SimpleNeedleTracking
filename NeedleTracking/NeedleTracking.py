import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np

from skimage import filters, morphology, feature, util
from skimage.restoration import unwrap_phase

import SimpleITK as sitk
import sitkUtils


class NeedleTracking(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "2D Needle Tracking "
    self.parent.categories = ["IGT"]
    self.parent.dependencies = []
    self.parent.contributors = ["Mariana Bernardes (BWH), Ahmed Mahran (BWH), Junichi Tokuda (BWH)"]
    self.parent.helpText = """This is a 2D needle tracking module used to segment the needle tip in MRI images. Input requirement: 
    Magnitude/Phase image or Real/Imaginary image. Uses scikit unwrapping algorithm. """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """"""

################################################################################################################################################
# Widget Class
################################################################################################################################################

class NeedleTrackingWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    ####################################
    ##                                ##
    ## Image selection                ##
    ##                                ##
    ####################################
    
    imagesCollapsibleButton = ctk.ctkCollapsibleButton()
    imagesCollapsibleButton.text = "Image Selection"
    self.layout.addWidget(imagesCollapsibleButton)
    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(imagesCollapsibleButton)

    #
    # Input mode
    #
    self.inputModePhase = qt.QRadioButton('Magnitude/Phase')
    self.inputModeRealImag = qt.QRadioButton('Real/Imaginary')
    self.inputModePhase.checked = 1
    
    self.inputModeButtonGroup = qt.QButtonGroup()
    self.inputModeButtonGroup.addButton(self.inputModePhase)
    self.inputModeButtonGroup.addButton(self.inputModeRealImag)

    inputModeLayout = qt.QHBoxLayout(imagesCollapsibleButton)
    inputModeLayout.addWidget(self.inputModePhase)
    inputModeLayout.addWidget(self.inputModeRealImag)
    parametersFormLayout.addRow("Input Mode:",inputModeLayout)

    #
    # Input magnitude/real volume (first volume)
    #
    self.firstVolumeSelector = slicer.qMRMLNodeComboBox()
    self.firstVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.firstVolumeSelector.selectNodeUponCreation = True
    self.firstVolumeSelector.addEnabled = True
    self.firstVolumeSelector.removeEnabled = True
    self.firstVolumeSelector.noneEnabled = True
    self.firstVolumeSelector.showHidden = False
    self.firstVolumeSelector.showChildNodeTypes = False
    self.firstVolumeSelector.setMRMLScene( slicer.mrmlScene )
    self.firstVolumeSelector.setToolTip("Select the magnitude/real image")
    parametersFormLayout.addRow("Magnitude/Real Image: ", self.firstVolumeSelector)

    #
    # Input phase/imaginary volume (second volume)
    #
    self.secondVolumeSelector = slicer.qMRMLNodeComboBox()
    self.secondVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.secondVolumeSelector.selectNodeUponCreation = True
    self.secondVolumeSelector.addEnabled = True
    self.secondVolumeSelector.removeEnabled = True
    self.secondVolumeSelector.noneEnabled = True
    self.secondVolumeSelector.showHidden = False
    self.secondVolumeSelector.showChildNodeTypes = False
    self.secondVolumeSelector.setMRMLScene( slicer.mrmlScene )
    self.secondVolumeSelector.setToolTip("Select the phase/imaginary image")
    parametersFormLayout.addRow("Phase/Imaginary Image: ", self.secondVolumeSelector)
   
    #
    # Select which scene view to track
    #
    self.sceneViewButton_red = qt.QRadioButton('Red')
    self.sceneViewButton_yellow = qt.QRadioButton('Yellow')
    self.sceneViewButton_green = qt.QRadioButton('Green')
    self.sceneViewButton_green.checked = 1

    self.sceneViewButtonGroup = qt.QButtonGroup()
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_red)
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_yellow)
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_green)
    
    layout = qt.QHBoxLayout(imagesCollapsibleButton)
    layout.addWidget(self.sceneViewButton_red)
    layout.addWidget(self.sceneViewButton_yellow)
    layout.addWidget(self.sceneViewButton_green)
    parametersFormLayout.addRow("Scene view:",layout)

    #
    # Needle Detection
    #
    self.detectNeedleButton = qt.QPushButton("Detect Needle")
    self.detectNeedleButton.toolTip = "Detect needle tip in current image"
    self.detectNeedleButton.enabled = False
    parametersFormLayout.addRow(self.detectNeedleButton)

    ####################################
    ##                                ##
    ## Live Tracking                  ##
    ##                                ##
    ####################################

    RTcollapsibleButton = ctk.ctkCollapsibleButton()
    RTcollapsibleButton.text =  "Live Tracking"
    self.layout.addWidget(RTcollapsibleButton)
    TrackingFormLayout = qt.QFormLayout(RTcollapsibleButton)

    #
    # FPS 
    #   
    self.fpsBox = qt.QDoubleSpinBox()
    self.fpsBox.setSingleStep(0.1)
    self.fpsBox.setMaximum(40)
    self.fpsBox.setMinimum(0.1)
    self.fpsBox.setSuffix(" FPS")
    self.fpsBox.value = 0.5
    TrackingFormLayout.addRow("Update Rate:", self.fpsBox)

    #
    # Start/Stop Live Tracking 
    #    
    liveTrackingButton = qt.QHBoxLayout()    
    
    # Create timer for Live Tracking
    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.needleTracker)
    self.counter = 0 
    
    # Start Live Tracking 
    self.startTrackingButton = qt.QPushButton("Start Live Tracking")
    self.startTrackingButton.toolTip = "Track needle tip in current image at given update rate"
    self.startTrackingButton.enabled = False
    self.startTrackingButton.clicked.connect(self.startTimer)
    liveTrackingButton.addWidget(self.startTrackingButton)

    # Stop Live Tracking
    self.stopTrackingButton = qt.QPushButton('Stop Live Tracking')
    self.stopTrackingButton.clicked.connect(self.stopTimer)
    liveTrackingButton.addWidget(self.stopTrackingButton)
     
    TrackingFormLayout.addRow("", liveTrackingButton)

    ####################################
    ##                                ##
    ## Advanced parameters            ##
    ##                                ##
    ####################################

    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    advancedCollapsibleButton.collapsed=1
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)

    #
    # Debug mode check box (output images at intermediate steps)
    #
    self.debugFlagCheckBox = qt.QCheckBox()
    self.debugFlagCheckBox.checked = 0
    self.debugFlagCheckBox.setToolTip("If checked, output images at intermediate steps")
    advancedFormLayout.addRow("Debug", self.debugFlagCheckBox)
    
    # #
    # # 2D Slice: not in use
    # #
    # self.imageSliceSliderWidget = ctk.ctkSliderWidget()
    # self.imageSliceSliderWidget.singleStep = 1
    # self.imageSliceSliderWidget.minimum = 0
    # self.imageSliceSliderWidget.maximum = 70
    # self.imageSliceSliderWidget.value = 1
    # self.imageSliceSliderWidget.setToolTip("Select 2D Slice")
    # advancedFormLayout.addRow("2D Slice", self.imageSliceSliderWidget)
       
    #
    # Mask threshold
    #
    self.maskThresholdWidget = ctk.ctkSliderWidget()
    self.maskThresholdWidget.singleStep = 1
    self.maskThresholdWidget.minimum = 0
    self.maskThresholdWidget.maximum = 100
    self.maskThresholdWidget.value = 20
    self.maskThresholdWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    advancedFormLayout.addRow("Mask Threshold ", self.maskThresholdWidget)

    #
    # Hessian blob detector
    #
    self.hessianThresholdWidget = ctk.ctkSliderWidget()
    self.hessianThresholdWidget.singleStep = 1
    self.hessianThresholdWidget.minimum = 0
    self.hessianThresholdWidget.maximum = 100
    self.hessianThresholdWidget.value = 20
    self.hessianThresholdWidget.setToolTip("Set up hessian blob detector threshold")
    advancedFormLayout.addRow("Hessian Blob Detector Threshold", self.hessianThresholdWidget)

    self.layout.addStretch(1)
    
    # Refresh button states
    self.onSelect()

    # Connections
    self.startTrackingButton.connect('clicked(bool)', self.needleTracker)
    self.detectNeedleButton.connect('clicked(bool)', self.needleTracker)
    self.firstVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelect)
    self.secondVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelect)
    self.timer = qt.QTimer()
    self.timer.timeout.connect(self.needleTracker) 

  def startTimer(self):
    self.timer.start(int(1000/float(self.fpsBox.value)))
    print ("Started Live Tracking ...")
    self.counter = 0 

  def stopTimer (self):
    self.timer.stop()
    print ("Stopped Live Tracking")

  def cleanup(self):
    pass

  def onSelect(self):
    self.detectNeedleButton.enabled = self.firstVolumeSelector.currentNode() and self.secondVolumeSelector.currentNode()
    self.startTrackingButton.enabled = self.firstVolumeSelector.currentNode() and self.secondVolumeSelector.currentNode()

  def getViewSelecter(self):
    viewSelecter = None
    if (self.sceneViewButton_red.checked == True):
      viewSelecter = ("Red")
    elif (self.sceneViewButton_yellow.checked ==True):
      viewSelecter = ("Yellow")
    elif (self.sceneViewButton_green.checked ==True):
      viewSelecter = ("Green")
    return viewSelecter
    
  def needleTracker(self):
    logic = NeedleTrackingLogic()
    viewSelecter = self.getViewSelecter()
    maskThreshold = self.maskThresholdWidget.value
    hessianThreshold = self.hessianThresholdWidget.value
    debugFlag = self.debugFlagCheckBox.checked
    useRealImag = self.inputModeRealImag.checked
    self.counter += 1
    logic.getNeedleTip(self.firstVolumeSelector.currentNode(), self.secondVolumeSelector.currentNode(), maskThreshold, hessianThreshold, viewSelecter, debugFlag, useRealImag, self.counter)

################################################################################################################################################
# Logic Class
################################################################################################################################################

class NeedleTrackingLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.cliParamNode = None

  def segmentNeedle(self, firstVolume, secondVolume, maskThreshold, hessianThreshold, sliceIndex, debugFlag=False, useRealImag=False, counter=1):

    ####################################
    ##                                ##
    ## Step 0: Select img volumes     ##
    ##      (Mag/Phase or Re/Im)      ##
    ####################################
    
    sitk_magn = None
    sitk_phase = None
    numpy_magn = None
    numpy_phase = None

    # Pull the real/imaginary volumes from the MRML scene and convert them to magnitude/phase volumes
    if useRealImag:
      sitk_real = sitkUtils.PullVolumeFromSlicer(firstVolume)
      sitk_imag = sitkUtils.PullVolumeFromSlicer(secondVolume)
      numpy_real = sitk.GetArrayFromImage(sitk_real)
      numpy_imag = sitk.GetArrayFromImage(sitk_imag)
      numpy_comp = numpy_real + 1.0j * numpy_imag
      numpy_magn = np.absolute(numpy_comp)
      numpy_phase = np.angle(numpy_comp)
      
      # Construct mag/phase ITK image
      sitk_magn = sitk.GetImageFromArray(numpy_magn)
      sitk_phase = sitk.GetImageFromArray(numpy_phase)
      sitk_magn.SetOrigin(sitk_real.GetOrigin())
      sitk_magn.SetSpacing(sitk_real.GetSpacing())
      sitk_magn.SetDirection(sitk_real.GetDirection())
      sitk_phase.SetOrigin(sitk_real.GetOrigin())
      sitk_phase.SetSpacing(sitk_real.GetSpacing())
      sitk_phase.SetDirection(sitk_real.GetDirection())
      
      # Push debug images to Slicer
      if debugFlag:
        self.pushSitkToSlicer(sitk_magn, 'debug_magn')
        self.pushSitkToSlicer(sitk_phase, 'debug_phase')
        
    # Pull the magnitude/phase volumes from the MRML scene
    else:
      sitk_magn = sitkUtils.PullVolumeFromSlicer(firstVolume)
      sitk_phase = sitkUtils.PullVolumeFromSlicer(secondVolume)

      magImageData = firstVolume.GetImageData()
      phaseImageData = secondVolume.GetImageData()
            
      # Adjust the value range of the phase image to [-pi, pi] 
      if phaseImageData.GetScalarTypeAsString() == 'unsigned short':
        sitk_phase = sitk_phase*np.pi/2048.0 - np.pi
      else:
        sitk_phase = sitk_phase*np.pi/4096.0
      
      numpy_magn = sitk.GetArrayFromImage(sitk_magn)
      numpy_phase = sitk.GetArrayFromImage(sitk_phase)

      # if debugFlag:
      #   # In tests with AMIGO scanner images: 
      #   # Magnitude image = unsigned short
      #   # Phase image = short (signed)
      #   print('Magnitude image data type:')
      #   print(magImageData.GetScalarTypeAsString())
      #   print('Phase image data type:')
      #   print(phaseImageData.GetScalarTypeAsString())
    
    # Get slice
    imgMag = numpy_magn[sliceIndex,:,:]
    imgPhase = numpy_phase[sliceIndex,:,:]
    
    ####################################
    ##                                ##
    ## Step 1: Mask magnitude image   ##
    ##                                ##
    ####################################

    # Generate bool mask from magnitude image to remove background
    boolMask = imgMag < maskThreshold
    
    if debugFlag:
      # Construct mask ITK image
      imgMask = np.zeros((1,boolMask.shape[0],boolMask.shape[1]))
      imgMask[0,:,:] = np.array(boolMask).astype(int)
      sitk_mask = sitk.GetImageFromArray(imgMask)
      sitk_mask.SetOrigin(sitk_magn.GetOrigin())
      sitk_mask.SetSpacing(sitk_magn.GetSpacing())
      sitk_mask.SetDirection(sitk_magn.GetDirection())      
      self.pushSitkToSlicer(sitk_mask, 'debug_mask'+str(counter))
      
    ####################################
    ##                                ##
    ## Step 2: Unwrap/mask phase img  ##
    ##                                ##
    ####################################

    phaseMasked = np.ma.array(imgPhase, mask=boolMask)  # Mask phase image
    phaseUnwrapped = unwrap_phase(phaseMasked)                                              # Use unwrap from scikit-image (module: restoration)
    phaseUnwrapped = 255*((phaseUnwrapped - np.min(phaseUnwrapped))/np.ptp(phaseUnwrapped)) # Normalize to grayscale (0-255)

    if debugFlag:
      # Construct phase masked ITK image
      imgPhaseUnwrapped = np.zeros((1,phaseUnwrapped.shape[0],phaseUnwrapped.shape[1]))
      imgPhaseUnwrapped[0,:,:] = np.array(phaseUnwrapped).astype(np.short)
      sitk_phase_cropped = sitk.GetImageFromArray(imgPhaseUnwrapped)
      sitk_phase_cropped.SetOrigin(sitk_phase.GetOrigin())
      sitk_phase_cropped.SetSpacing(sitk_phase.GetSpacing())
      sitk_phase_cropped.SetDirection(sitk_phase.GetDirection())      
      self.pushSitkToSlicer(sitk_phase_cropped, 'debug_phase_masked'+str(counter))

    ####################################
    ##                                ##
    ## Step 3: High pass filter       ##
    ##         (Sharpening)           ##
    ####################################
    
    phaseSharpen = filters.butterworth(phaseUnwrapped, cutoff_frequency_ratio=0.05, order=4.0, high_pass=True)  # Sharpen with high pass filter
    
    if debugFlag:
      # Construct phase sharpen ITK image
      imgPhaseSharpen = np.zeros((1,phaseSharpen.shape[0],phaseSharpen.shape[1]))
      imgPhaseSharpen[0,:,:] = np.array(phaseSharpen)
      sitk_phase_sharpen = sitk.GetImageFromArray(imgPhaseSharpen)
      sitk_phase_sharpen.SetOrigin(sitk_phase.GetOrigin())
      sitk_phase_sharpen.SetSpacing(sitk_phase.GetSpacing())
      sitk_phase_sharpen.SetDirection(sitk_phase.GetDirection())      
      self.pushSitkToSlicer(sitk_phase_sharpen, 'debug_phase_sharpen'+str(counter))

    ####################################
    ##                                ##
    ## Step 4: Dilation in mask       ##
    ##                                ##
    ####################################

    boolMaskDilated = morphology.dilation(boolMask, morphology.disk(10))  # Bool mask with dilated borders
    phaseSharpen[boolMaskDilated] = np.mean(phaseSharpen)                 # Apply dilated mask

    if debugFlag:
      # Construct masked phase sharpen ITK image
      imgPhaseSharpenMasked = np.zeros((1,phaseSharpen.shape[0],phaseSharpen.shape[1]))
      imgPhaseSharpenMasked[0,:,:] = np.array(phaseSharpen)
      sitk_phase_sharpen_masked = sitk.GetImageFromArray(imgPhaseSharpenMasked)
      sitk_phase_sharpen_masked.SetOrigin(sitk_phase.GetOrigin())
      sitk_phase_sharpen_masked.SetSpacing(sitk_phase.GetSpacing())
      sitk_phase_sharpen_masked.SetDirection(sitk_phase.GetDirection())      
      self.pushSitkToSlicer(sitk_phase_sharpen_masked, 'debug_phase_sharpen_masked'+str(counter))

    ####################################
    ##                                ##
    ## Step 5: Hessian filter         ##
    ##         (Blob detection)       ##
    ####################################
    
    # Use hessian matrix determinant blob detector
    blobs_doh = feature.blob_doh(util.invert(phaseSharpen), threshold=hessianThreshold) # blobs_doh = (y,x,radius)
    blobsSorted = np.argsort(blobs_doh[:,2]) # Sorted by ascending radius of blobs           
      
    ####################################
    ##                                ##
    ## Step 6: Tip coordinates        ##
    ##                                ##
    ####################################
    
    # Find tip blob
    if blobsSorted.shape[0] > 0:        # If a blob was found
      y = blobs_doh[blobsSorted[-1],0]  # Index of biggest radius = blobsSorted[-1]
      x = blobs_doh[blobsSorted[-1],1]
      
      # Get RAS coordinates for tip
      coords_ijk = [x,y,sliceIndex,1.0]
      matrix = vtk.vtkMatrix4x4()
      firstVolume.GetIJKToRASMatrix(matrix)
      coords_ras = matrix.MultiplyPoint(coords_ijk)
      coords_ras = coords_ras[0:3]

      # Update needle tip fiducial
      fidNode = None
      try: 
        fidNode = slicer.util.getNode('needle_tip')
        fidNode.SetNthFiducialPositionFromArray(0, coords_ras)
      except slicer.util.MRMLNodeNotFoundException as exc:
        fidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "needle_tip")   
        fidNode.AddFiducialFromArray(coords_ras)   
      
      if debugFlag:
        print('Number of blobs = ',str(blobsSorted.size))   
        print('Tip (ijk) = ',str(coords_ijk))   
        print('Tip (ras) = ',str(coords_ras))   

    else:
      print('Needle tip not detected')
    return True

  def pushSitkToSlicer(self, sitkImage, name):
    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    node.SetName(name)
    sitkUtils.PushVolumeToSlicer(sitkImage, name, 0, True)
  
  def findSliceIndex(self, viewSelecter):   
    # Find current slice index
    layoutManager = slicer.app.layoutManager()
    sliceWidgetLogic = layoutManager.sliceWidget(str(viewSelecter)).sliceLogic()
    return sliceWidgetLogic.GetSliceIndexFromOffset(sliceWidgetLogic.GetSliceOffset()) - 1

  def getNeedleTip(self, firstVolume , secondVolume, maskThreshold, hessianThreshold, viewSelecter, debugFlag, useRealImag, counter):
    if debugFlag:
      print('Counter = ',str(counter))   
    
    # Detect needle in current slice from selected view
    sliceIndex = self.findSliceIndex(viewSelecter)
    self.segmentNeedle(firstVolume , secondVolume, maskThreshold, hessianThreshold, sliceIndex, debugFlag, useRealImag, counter)

    # Set the Slice view (is this needed?)
    slice_logic = slicer.app.layoutManager().sliceWidget(str(viewSelecter)).sliceLogic()
    slice_logic.GetSliceCompositeNode().SetBackgroundVolumeID(firstVolume.GetID())