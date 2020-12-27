import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from skimage.filters import meijering, frangi, sato
import nibabel as nib
import cv2
import tempfile


class NeedleSegmentor(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Needle Segmenter"
    self.parent.categories = ["Filtering"]
    self.parent.dependencies = []
    self.parent.contributors = ["Ahmed Mahran (BWH)"]
    self.parent.helpText = """This is a needle segmenter module used to localize needle tip in the MRI image. It uses phase unwrapping method coupled
    algorithm to locate the needle tip. """
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
    self.magnitudevolume.setToolTip( "Select the magnitude image" )
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
    self.phasevolume.setToolTip( "Select the phase image" )
    parametersFormLayout.addRow("Phase Image: ", self.phasevolume)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

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
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.magnitudevolume.currentNode(), self.phasevolume.currentNode(), imageThreshold, enableScreenshotsFlag)

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


  def run(self, magnitudevolume , phasevolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    #magnitude volume
    magn_imageData = magnitudevolume.GetImageData()
    magn_rows, magn_cols, magn_zed = magn_imageData.GetDimensions()
    magn_scalars = magn_imageData.GetPointData().GetScalars()
    magn_imageOrigin = magnitudevolume.GetOrigin()
    magn_imageSpacing = magnitudevolume.GetSpacing()
    magn_matrix = vtk.vtkMatrix4x4()
    magnitudevolume.GetIJKToRASDirectionMatrix(magn_matrix)

    # WRITE PHASE FILE IN NIFTI FORMAT
    phase_imageData = phasevolume.GetImageData()
    phase_rows, phase_cols, phase_zed = phase_imageData.GetDimensions()
    phase_scalars = phase_imageData.GetPointData().GetScalars()
    # imageOrigin = phasevolume.GetOrigin()
    # imageSpacing = phasevolume.GetSpacing()
    # phase_matrix = vtk.vtkMatrix4x4()
    # phasevolume.GetIJKToRASDirectionMatrix(phase_matrix)

    #Convert vtk to numpy
    NumpyArray = numpy_support.vtk_to_numpy(magn_scalars)
    magnp = NumpyArray.reshape(magn_zed, magn_rows, magn_cols)
    phase_numpyarray = numpy_support.vtk_to_numpy(phase_scalars)
    phasep = phase_numpyarray.reshape(phase_zed, phase_rows, phase_cols)

    #mask
    mask = magnp > 55
    mask1 = np.array(mask)
    mask1 = mask1.astype(np.uint8)

    mask1 = mask1.squeeze()
    phasep = phasep.squeeze()

    #phase_croppedc
    phase_cropped = cv2.bitwise_and(phasep, phasep, mask=mask1)
    phase_cropped = np.expand_dims(phase_cropped, axis=0)

    node = slicer.vtkMRMLScalarVolumeNode()
    node.SetName('phase_cropped')
    slicer.mrmlScene.AddNode(node)
    slicer.util.updateVolumeFromArray(node, phase_cropped)
    node.SetOrigin(magn_imageOrigin)
    node.SetSpacing(magn_imageSpacing)
    node.SetIJKToRASDirectionMatrix(magn_matrix)
    node.CreateDefaultDisplayNodes()


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

    I = phaseunwrapped
    A = np.fft.fft2(I)
    A1 = np.fft.fftshift(A)

    # Image size
    [M, N, O] = A.shape

    # filter size parameter
    R = 10

    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)
    # Y = Y.astype(int)

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

    kernel1 = np.ones((3, 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask3 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel1)
    mask3 = cv2.erode(mask3, kernel, iterations=7)
    mask3 = mask3.astype(np.uint8)

    B2 = cv2.bitwise_and(B2.squeeze(), B2.squeeze(), mask=mask3)

    meiji = meijering(B2, sigmas=(1, 1), black_ridges=True)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(meiji)
    # B3 = cv2.circle(B2, maxLoc, 3, (255, 100, 0), 1)
    circle1 = plt.Circle(maxLoc, 2, color='red')
    #
    # plt.imshow(meiji, cmap='gray')
    # plt.add_artist(circle1)
    # plt.show()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Needle Tracking')
    axs[0].imshow(magnp.squeeze(), cmap='gray')
    axs[0].set_title('Magnitude + Tracked')
    axs[0].add_artist(circle1)
    axs[0].axis('off')
    axs[1].set_title('Processed Phase Image')
    axs[1].imshow(meiji, cmap='jet')
    axs[1].axis('off')
    plt.savefig("mygraph.png")
    # plt.show()

    # B3 = np.expand_dims(B3, axis=0)
    # final = slicer.vtkMRMLScalarVolumeNode()
    # final.SetName('final')
    # slicer.mrmlScene.AddNode(final)
    # slicer.util.updateVolumeFromArray(final, B3)
    # final.SetOrigin(magn_imageOrigin)
    # final.SetSpacing(magn_imageSpacing)
    # final.SetIJKToRASDirectionMatrix(magn_matrix)
    # final.CreateDefaultDisplayNodes()

    print(maxLoc)

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
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

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
