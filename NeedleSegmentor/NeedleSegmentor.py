import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
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
    self.parent.title = "2D Needle Segmenter v0.2"
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

    # Instantiate and connect widgets ...

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
    self.imageSliceSliderWidget.maximum = 20
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
    self.maskThresholdWidget.value = 70
    self.maskThresholdWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("mask threshold ", self.maskThresholdWidget)

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
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.magnitudevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.phasevolume.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.magnitudevolume.currentNode() and self.phasevolume.currentNode()

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


  def run(self, magnitudevolume , phasevolume, imageSlice, maskThreshold, ridgeOperator, enableScreenshots=0):

    #magnitude volume
    magn_imageData = magnitudevolume.GetImageData()
    magn_rows, magn_cols, magn_zed = magn_imageData.GetDimensions()
    magn_scalars = magn_imageData.GetPointData().GetScalars()
    magn_imageOrigin = magnitudevolume.GetOrigin()
    magn_imageSpacing = magnitudevolume.GetSpacing()
    magn_matrix = vtk.vtkMatrix4x4()
    magnitudevolume.GetIJKToRASDirectionMatrix(magn_matrix)
    magnitudevolume.CreateDefaultDisplayNodes()


    # WRITE PHASE FILE IN NIFTI FORMAT
    phase_imageData = phasevolume.GetImageData()
    phase_rows, phase_cols, phase_zed = phase_imageData.GetDimensions()
    phase_scalars = phase_imageData.GetPointData().GetScalars()
    # imageOrigin = phasevolume.GetOrigin()
    # imageSpacing = phasevolume.GetSpacing()
    # phase_matrix = vtk.vtkMatrix4x4()
    # phasevolume.GetIJKToRASDirectionMatrix(phase_matrix)


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
    _, threshed = cv2.threshold(numpy_magn_sliced, 20, 255, cv2.THRESH_BINARY)
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

    #Remove border of for false positive
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
    
    (y1,x1) = np.unravel_index(sort[1], meiji.shape)

    point = (x1,y1)
    
    magn_imageOriginr, magn_imageOrigina, magn_imageOrigins = magnitudevolume.GetOrigin()
    magn_imageSpacingr, magn_imageSpacinga, magn_imageSpacings = magnitudevolume.GetSpacing()

    #dev/ delete once done
    print("imageorigin: ", magn_imageOriginr, magn_imageOrigina, magn_imageOrigins)
    print("imageSpacing: ", magn_imageSpacingr, magn_imageSpacinga, magn_imageSpacings)

    #x,y = np.split(maxLoc, [-1], 0)
    R_loc = (magn_imageOriginr)-(x1*magn_imageSpacingr)
    A_loc = (magn_imageOrigina)-(slice*magn_imageSpacings)
    S_loc = (magn_imageOrigins)-(y1*magn_imageSpacinga)
    

    result = slicer.vtkMRMLMarkupsFiducialNode()
    result.AddFiducial(R_loc,A_loc,S_loc,"Needle_Tip")
    result.SetName('needle_location')
    slicer.mrmlScene.AddNode(result)


    ###TODO: dont delete the volume after use. create a checkpoint to update on only one volume
    delete_wrapped = slicer.mrmlScene.GetFirstNodeByName('phase_cropped')
    slicer.mrmlScene.RemoveNode(delete_wrapped)
    delete_unwrapped = slicer.mrmlScene.GetFirstNodeByName('unwrapped_phase')
    slicer.mrmlScene.RemoveNode(delete_unwrapped)

    #TODO: convert the numpy coorinate to a RAS coorindate (R=x, S=y) and add a fiducial of the coordinate to the world coordinate (vtk)

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
