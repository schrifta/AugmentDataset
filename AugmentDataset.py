#class Label(object):
class Label:
    """label class."""
    def __init__(self, cls, x, y, w, h):
        self.cls = cls
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def CalcIOU(l1,l2):
    """Calculate IOU of given labels"""
    
    x1s = l1.x-l1.w/2
    x1e = l1.x+l1.w/2
    x2s = l2.x-l2.w/2
    x2e = l2.x+l2.w/2
    y1s = l1.y-l1.h/2
    y1e = l1.y+l1.h/2
    y2s = l2.y-l2.h/2
    y2e = l2.y+l2.h/2
    
    # Calculate horizontal intersection
    if x1s < x2s:
        if x1e < x2s:
            w = 0
        elif x1e < x2e:
            w = x1e-x2s            
        else:
            w = x2e-x2s
    elif x1s < x2e:
        if x1e < x2s:
            w = 0
        elif x1e < x2e:
            w = x1e-x1s
        else:
            w = x2e-x1s
    else:
        w = 0

    if w==0:
        return 0
    
    # Calculate vertical intersection
    if y1s < y2s:
        if y1e < y2s:
            h = 0
        elif y1e < y2e:
            h = y1e-y2s            
        else:
            h = y2e-y2s
    elif y1s < y2e:
        if y1e < y2s:
            h = 0
        elif y1e < y2e:
            h = y1e-y1s
        else:
            h = y2e-y1s
    else:
        h = 0

    if h==0:
        return 0
    
    union = l1.w*l1.h + l2.w*l2.h - w*h
    return w*h / union

def DetectionFitLabels( labels, boxes ):
    """Decide if the detection fit the labels"""
    
    iouThr = 0.75
    labels_count = len(labels)
    boxes_count = len(boxes)
    labels_fit = [0]*labels_count
    boxes_fit = [0]*boxes_count
    #for lbl in labels:
    for i1, lbl in enumerate(labels,start=0):
        #color = (0,255,0) if lbl.cls=='1' else (0,80,255)
        #for det in boxes:
        for i2, det in enumerate(boxes,start=0):
            # color = (0,160,0) if det.cls[0].item()==1 else (0,0,160)
            det_lbl = Label(str(int(det.cls[0].item())),
                            det.xywhn[0][0].item(), det.xywhn[0][1].item(),
                            det.xywhn[0][2].item(), det.xywhn[0][3].item())
            if lbl.cls==det_lbl.cls:
                iou = CalcIOU(lbl,det_lbl)
                if iou > iouThr:
                    labels_fit[i1] = labels_fit[i1] + 1
                    boxes_fit[i2] = boxes_fit[i2] + 1
    
    # Count false detections and miss detections
    miss_count = 0
    false_count = 0
    for ifit, fit in enumerate(labels_fit,start=0):
        if labels[ifit].cls=='1' and fit==0:
            miss_count = miss_count + 1
    for ifit, fit in enumerate(boxes_fit,start=0):
        if boxes[ifit].cls==1 and fit==0:
            false_count = false_count + 1
    
    return miss_count==0 and false_count==0

def GetXmlLables(labelFile):
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
        xmin_ = int(box.find('xmin').text)
        ymin_ = int(box.find('ymin').text)
        xmax_ = int(box.find('xmax').text)
        ymax_ = int(box.find('ymax').text)
        labels.append([name,xmin_,ymin_,xmax_,ymax_])
        xmin=min(xmin,xmin_)
        xmax=max(xmax,xmax_)
        ymin=min(ymin,ymin_)
        ymax=max(ymax,ymax_)
        boundingBox = [xmin,ymin,xmax,ymax]
    return labels, boundingBox

def CalcRotationROI(boundingBox, imgWidth, imgHeight):
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
        if lbl[1]>=roi[0] and lbl[1]<roi[2] and \
           lbl[2]>=roi[1] and lbl[2]<roi[3] and \
           lbl[3]>=roi[0] and lbl[3]<roi[2] and \
           lbl[4]>=roi[1] and lbl[4]<roi[3]:
            p1 = np.array([[lbl[1]],[lbl[2]]])
            p2 = np.array([[lbl[3]],[lbl[4]]])
            tr_p1 = (T@(p1-p0) + p0).tolist()
            tr_p2 = (T@(p2-p0) + p0).tolist()
            tr_labels.append([lbl[0],int(tr_p1[0][0]),int(tr_p1[1][0]), int(tr_p2[0][0]),int(tr_p2[1][0])])
    return tr_labels

def SaveLabelsToText( labels, imgWidth, imgHeight, labelsFile ):
    with open(labelsFile, "w") as f:
        for lbl in labels:
            clss = 1 if lbl[0]=='sperm' else (0 if lbl[0]=='nonsperm' else -1)
            x = (lbl[3]+lbl[1])/2/imgWidth
            w = (lbl[3]-lbl[1])/imgWidth
            y = (lbl[4]+lbl[2])/2/imgHeight
            h = (lbl[4]-lbl[2])/imgHeight
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
    #from ultralytics import YOLO
    #import torch
    #torch.cuda.is_available()
    #torch.cuda.set_device(0)
    #from xml.etree import ElementTree as ET
    
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
    # Each image is rotaged by 90, 180 and 270 degrees,
    # and horizontally and vertically reflected.
    # From each image has 6 image versions (including original image).
    
    # Create a random distribution list, to assign augumented iamges to the
    # train, val and test directories.
    jpgQuality = 100
    totalCount = len(imgList)*6
    indexes = list(range(totalCount))
    random.seed(42)
    random.shuffle(indexes)
    s1 = int(0.7*totalCount)
    s2 = int(0.9*totalCount)
    target = [0]*totalCount
    for i,j in enumerate(indexes,start=0):
        target[i] = 0 if j<s1 else 1 if j<s2 else 2
    
    for i, elem in enumerate(imgList,start=0):
        
        # Read the input image
        image = cv2.imread(os.path.join(datasetPath, elem))
        height, width = image.shape[:2]
        
        # Read the input labels
        xmlFile = os.path.join(datasetPath, elem.replace('.png', '.xml'))
        labels, boundingBox = GetXmlLables(xmlFile)
        if len(labels)==0 or len(boundingBox)==0:
            continue
        roi = CalcRotationROI(boundingBox, width, height)

        # original
        idx = 6*i
        #trans = [[1,0],[0,1]] # identity
        #rotatedLabels = RotateLabels(labels,width,height,roi,trans)
        rotatedLabels = labels
        
        if len(rotatedLabels):
            subdir = subdirs[target[idx]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '.txt'))
    
            cv2.imwrite(targetImageFile, image, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        
        # 90 degrees
        trans = [[0,-1],[1,0]] # 90 deg
        rotatedLabels = RotateLabels(labels,width,height,roi,trans)

        if len(rotatedLabels):
            subdir = subdirs[target[idx+1]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '-90.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-90.txt'))

            tmpImage = np.zeros((height, width,3), np.uint8)
            tmpImage[roi[1]:roi[3],roi[0]:roi[2]] = \
                cv2.rotate(image[roi[1]:roi[3],roi[0]:roi[2]],cv2.ROTATE_90_CLOCKWISE)
    
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break


        # 180 degrees
        trans = [[-1,0],[0,-1]] # 180 deg
        rotatedLabels = RotateLabels(labels,width,height,roi,trans)

        if len(rotatedLabels):
            subdir = subdirs[target[idx+2]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '-180.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-180.txt'))
    
            tmpImage = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break

        
        # 270 degrees
        trans = [[0,1],[-1,0]] # 270 deg
        rotatedLabels = RotateLabels(labels,width,height,roi,trans)
        
        if len(rotatedLabels):
            subdir = subdirs[target[idx+3]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '-270.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-270.txt'))

            tmpImage = np.zeros((height, width,3), np.uint8)
            tmpImage[roi[1]:roi[3],roi[0]:roi[2]] = \
                cv2.rotate(image[roi[1]:roi[3],roi[0]:roi[2]],cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break

        # horizontal reflection
        trans = [[-1,0],[0,1]] # x reflectin
        rotatedLabels = RotateLabels(labels,width,height,roi,trans)

        if len(rotatedLabels):
            subdir = subdirs[target[idx+4]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '-hor.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-hor.txt'))

            tmpImage = cv2.flip(image, 1 )
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break
        
        # vertical reflection
        trans = [[1,0],[0,-1]] # y reflection
        rotatedLabels = RotateLabels(labels,width,height,roi,trans)

        if len(rotatedLabels):
            subdir = subdirs[target[idx+5]]
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\images"
            targetImageFile = os.path.join(targetPath, elem.replace('.png', '-ver.jpg'))
            targetPath = augmentedDatasetPath+"\\"+subdir+"\\labels"
            targetLabelFile = os.path.join(targetPath,elem.replace('.png', '-ver.txt'))

            tmpImage = cv2.flip(image, 0 )
            cv2.imwrite(targetImageFile, tmpImage, [cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
            SaveLabelsToText( rotatedLabels, width, height, targetLabelFile )
            
            if showIndicatedImages:
                if -1==ShowSavedFiles( targetImageFile,targetLabelFile ):
                    break


datasetPath = 'C:\Projects\Cytobit\Spr#12-1\Spr#12+iteration10'
augmentedDatasetPath = 'D:\CytobitData\DataSets\My\Spr#12-1'
if __name__ == "__main__":
    AugmentDataset(datasetPath, augmentedDatasetPath)