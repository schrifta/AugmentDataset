#class Label(object):
class Label:
    """label class."""
    def __init__(self, cls, x, y, w, h):
        self.cls = cls
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def GetXmlLables(labelFile, fx, fy):
    """Read labels from an XML file and return a label list"""
    
    import xml.etree.ElementTree as ET

    tree = ET.parse(labelFile)
    root = tree.getroot()
    labels = []
    boundingBox = []
    xmin=100000
    xmax=0
    ymin=100000
    ymax=0
    for lbl in root.iter('object'):
        name = lbl.find('name').text
        box = lbl.find('bndbox')
        xmin_ = fx*int(box.find('xmin').text)
        ymin_ = fy*int(box.find('ymin').text)
        xmax_ = fx*int(box.find('xmax').text)
        ymax_ = fy*int(box.find('ymax').text)
        labels.append([name,xmin_,ymin_,xmax_,ymax_])
        xmin=min(xmin,xmin_)
        xmax=max(xmax,xmax_)
        ymin=min(ymin,ymin_)
        ymax=max(ymax,ymax_)
        boundingBox = [xmin,ymin,xmax,ymax]
    return labels, boundingBox

def Calc90RotationROI(boundingBox, imgWidth, imgHeight):
    """ Calculate the ROI to be rotated, so it will include as much as possible 
    of the given bounding box"""
    
    # check if the bounding box is inside the centeral height x height region
    roi_xmin = int((imgWidth-imgHeight)/2)
    roi_xmax = int((imgWidth+imgHeight)/2)
    if boundingBox[0]<roi_xmin or boundingBox[2]>roi_xmax:
        # calculate a non-center roi
        # the center of the roi is the center fo the bounding box
        roi_xmin = max(0,int((boundingBox[0]+boundingBox[2]-imgHeight)/2))
        roi_xmax = roi_xmin+imgHeight
        if roi_xmax>imgWidth:
            roi_xmax = imgWidth
            roi_xmin = roi_xmax - imgHeight

    return [roi_xmin, 0, roi_xmax, imgHeight]

def RotateLabels( labels, imgWidth, imgHeight, roi, trans ):
    """ Rotate labels by the angleDeg around the center of roi"""

    import numpy as np
    tr_labels = []
    p0 = np.array([[(roi[0]+roi[2])/2],[(roi[1]+roi[3])/2]])
    T = np.array(trans)
    for lbl in labels:
        p1 = np.array([[lbl[1]],[lbl[2]]])
        p2 = np.array([[lbl[3]],[lbl[4]]])
        tr_p1 = (T@(p1-p0) + p0).tolist()
        tr_p2 = (T@(p2-p0) + p0).tolist()
        if tr_p1[0][0]>=0 and tr_p1[0][0]<=imgWidth and \
           tr_p1[1][0]>=0 and tr_p1[1][0]<=imgHeight and \
           tr_p2[0][0]>=0 and tr_p2[0][0]<=imgWidth and \
           tr_p2[1][0]>=0 and tr_p2[1][0]<=imgHeight:
            tr_labels.append([lbl[0],int(tr_p1[0][0]),int(tr_p1[1][0]), int(tr_p2[0][0]),int(tr_p2[1][0])])
        
    return tr_labels

def SaveLabelsToText( labels, imgWidth, imgHeight, labelsFile ):
    with open(labelsFile, "w") as f:
        for lbl in labels:
            clss = 1 if lbl[0]=='sperm' else (0 if lbl[0]=='nonsperm' else -1)
            x = (lbl[3]+lbl[1])/2/imgWidth
            w = abs(lbl[3]-lbl[1])/imgWidth
            y = (lbl[4]+lbl[2])/2/imgHeight
            h = abs(lbl[4]-lbl[2])/imgHeight
            f.write('%s %8.6f %8.6f %8.6f %8.6f\n' % (clss, x, y, w, h))

def ShowSavedFiles( imageFile, labelFile ):
    """Show images with labels"""
    
    import cv2
    image = cv2.imread(imageFile)
    height, width = image.shape[:2]
    with open(labelFile, "r") as f:
        for line in f:
            cls, x, y, w, h = line.split()
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            x1 = int((x-w/2)*width)
            y1 = int((y-h/2)*height)
            x2 = int((x+w/2)*width)
            y2 = int((y+h/2)*height)
            #labels.append(Label(cls, x, y, w, h))
            color = (0,255,0) if cls=='1' else (0,0,255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
        cv2.namedWindow(imageFile, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(imageFile, 1044, 780)
        cv2.imshow(imageFile, image)    
        rc = cv2.waitKeyEx(0)
        cv2.destroyAllWindows()

        if rc==27:
            return(-1)
        return(0)

def TestForBackCenter( image ):
    height, width = image.shape[:2]
    top = int(height/4)
    bottom = int(3*height/4)
    left = int((width-height)/2)
    right = int ((width+height)/2)
    s = image[top:bottom,left:right].mean()
    return True if s<5 else False
    
def AugmentDataset(datasetPath, augmentedDatasetPath):
    """Create augmentation for iamges in the given dataset."""
    """All given images and labeling file are in the given path.
    The labeling files are in XML format, created by LableImg.
    The augumented images are in jpg format.
    Augmentation contain rotations and reflections.
    Rotation centers are calculated to leave most of rotated labeling
    rectangles inside the rotated image."""

    import numpy as np
    import cv2
    import os
    #import copy
    import random
    from sys import exit
    from datetime import datetime
    #from ultralytics import YOLO
    #import torch
    #torch.cuda.is_available()
    #torch.cuda.set_device(0)
    #from xml.etree import ElementTree as ET
    
    resWidth = 1280
    resHeight = 960
    showIndicatedImages = False

    # Check the given dataset path and create an image file list
    imgList = []
    print("Processing {}".format(datasetPath))
    if os.path.isdir(datasetPath):
        # Get image file names
        for elem in os.listdir(datasetPath):
            if elem.endswith('.png'):
                xmlFile = os.path.join(datasetPath,elem.replace('.png', '.xml'))
                if os.path.exists(xmlFile):
                    imgList.append(elem)
    else:
        print(datasetPath + " not found!")
        exit(-1)

    # Create the output directory structure 
    try:
        os.mkdir(augmentedDatasetPath)
    except OSError as error:
        if os.path.isdir(augmentedDatasetPath):
            print(augmentedDatasetPath + " already exists!")
        else:
            print(augmentedDatasetPath + " creation problem!")
            exit(-1)

    subdirs = ["train", "test", "valid"]
    for subdir in subdirs:
        tpath = augmentedDatasetPath+"\\"+subdir
        try:
            os.mkdir(tpath)
        except OSError as error:
             if os.path.isdir(tpath):
                 print(tpath + " already exists!")
             else:
                 print(tpath + " creation problem!")
                 exit(-1)
        subdirs2 = ["images", "labels"]
        for subdir2 in subdirs2:
            tpath2 = augmentedDatasetPath+"\\"+subdir+"\\"+subdir2
            try:
                os.mkdir(tpath2)
            except OSError as error:
                 if os.path.isdir(tpath2):
                     print(tpath2 + " already exists!")
                 else:
                     print(tpath2 + " creation problem!")
                     exit(-1)

    # Create augmented images
    # Each image is rotated by 90, 180 and 270 degrees,
    # and horizontally and vertically reflected.
    # From each image has 6 image versions (including original image).
    
    # Create a random distribution list, to assign augumented iamges to the
    # train, val and test directories.
    jpgQuality = 75
    totalCount = len(imgList)*6
    indexes = list(range(totalCount))
    random.seed(42)
    random.shuffle(indexes)
    s1 = int(0.7*totalCount)
    s2 = int(0.9*totalCount)
    target = [0]*totalCount
    for i,j in enumerate(indexes,start=0):
        target[i] = 0 if j<s1 else 1 if j<s2 else 2
    
    # Create a log file for imaegs not added to the output dataset
    logFile = open(augmentedDatasetPath+'\\augmentation.log', "w")
    logFile.writelines([datetime.now().strftime('%d.%m.%y %H:%M:%S'), '   ', datasetPath, '\n'])
    logFile.close()
                  

    for i, elem in enumerate(imgList,start=0):
        
        # Read the input image, and resize it
        image = cv2.imread(os.path.join(datasetPath, elem))
        
        # Remove defected balck images
        blackCenter = TestForBackCenter( image )
        if blackCenter:
            continue
        
        height, width = image.shape[:2]
        fx = resWidth/width
        fy = resHeight/height
        image = cv2.resize(image, (resWidth, resHeight))
        height, width = image.shape[:2]
        
        # Read the input labels
        xmlFile = os.path.join(datasetPath, elem.replace('.png', '.xml'))
        labels, boundingBox = GetXmlLables(xmlFile, fx, fy)
        if len(labels)==0 or len(boundingBox)==0:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([elem,' is skipped. Empty input labels and bounding box\n'])
            logFile.close()
            continue

        roi90 = Calc90RotationROI(boundingBox, width, height)
        roi0 = [0,0,width,height]
            
        # original
        idx = 6*i
        trans = [[1,0],[0,1]] # identity
        rotatedLabels = RotateLabels(labels,width,height,roi0,trans)
        
        subdir = subdirs[target[idx]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '.txt'))
    
        if len(rotatedLabels):
            cv2.imwrite(targetImageFile, image, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()
        
        # 90 degrees
        trans = [[0,-1],[1,0]] # 90 deg
        rotatedLabels = RotateLabels(labels,width,height,roi90,trans)

        subdir = subdirs[target[idx+1]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '-90.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-90.txt'))

        if len(rotatedLabels):
            tmpImage = np.zeros((height, width,3), np.uint8)
            tmpImage[roi90[1]:roi90[3],roi90[0]:roi90[2]] = \
                cv2.rotate(image[roi90[1]:roi90[3],roi90[0]:roi90[2]],cv2.ROTATE_90_CLOCKWISE)
    
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()


        # 180 degrees
        trans = [[-1,0],[0,-1]] # 180 deg
        rotatedLabels = RotateLabels(labels,width,height,roi0,trans)

        subdir = subdirs[target[idx+2]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '-180.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-180.txt'))

        if len(rotatedLabels):
            tmpImage = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()

        
        # 270 degrees
        trans = [[0,1],[-1,0]] # 270 deg
        rotatedLabels = RotateLabels(labels,width,height,roi90,trans)
        
        subdir = subdirs[target[idx+3]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '-270.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-270.txt'))

        if len(rotatedLabels):
            tmpImage = np.zeros((height, width,3), np.uint8)
            tmpImage[roi90[1]:roi90[3],roi90[0]:roi90[2]] = \
                cv2.rotate(image[roi90[1]:roi90[3],roi90[0]:roi90[2]],cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()

        # horizontal reflection
        trans = [[-1,0],[0,1]] # x reflectin
        rotatedLabels = RotateLabels(labels,width,height,roi0,trans)

        subdir = subdirs[target[idx+4]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '-hor.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-hor.txt'))

        if len(rotatedLabels):
            tmpImage = cv2.flip(image, 1 )
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()
        
        # vertical reflection
        trans = [[1,0],[0,-1]] # y reflection
        rotatedLabels = RotateLabels(labels,width,height,roi0,trans)

        subdir = subdirs[target[idx+5]]
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
        targetImageFile = os.path.join(targetPath, elem.replace('.png', '-ver.jpg'))
        targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
        targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-ver.txt'))

        if len(rotatedLabels):
            tmpImage = cv2.flip(image, 0 )
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        else:
            logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
            logFile.writelines([targetLabelFile,' is skipped. No rotated labels\n'])
            logFile.close()

    logFile = open(augmentedDatasetPath+'\\augmentation.log', "a")
    logFile.writelines([datetime.now().strftime('%d.%m.%y %H:%M:%S'), '   finished'])
    logFile.close()

datasetPath = 'C:\\Projects\\Cytobit\\Spr#12-1\\Spr#12+iteration10'
augmentedDatasetPath = 'D:\\CytobitData\\DataSets\\My\Spr#12-1-my'
#datasetPath = 'C:\\Projects\\Cytobit\\Spr#12-1\\tmp'
#augmentedDatasetPath = 'D:\\CytobitData\\DataSets\\My\\tmp'
if __name__ == "__main__":
    AugmentDataset(datasetPath, augmentedDatasetPath)