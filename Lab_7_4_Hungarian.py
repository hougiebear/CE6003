import os
import math
import re
import io
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import base64
from IPython.display import clear_output, Image, display
from scipy.spatial import distance

class Tracker(): # Class to keep track of trackers
    def __init__(self):
        # Initialise tracker's history
        self.id = 0                             # tracker's id
        self.box = np.zeros(shape=(2,2))        # bounding box co-ordinates
        self.numHits = 0                        # number of detection matches
        self.numMisses = 0                      # number of missed detections

        self.xState = np.matrix('0. 0. 0. 0.').T

        # Process matrix, assuming constant velocity model (x, y, x_dot, y_dot)
        self.F = np.matrix('''
                       1. 0. 1. 0.;
                       0. 1. 0. 1.;
                       0. 0. 1. 0.;
                       0. 0. 0. 1.
                       ''')

        # Measurement matrix, assumig we can only measure the co-ordinates
        self.H = np.matrix('''
                       1. 0. 0. 0.;
                       0. 1. 0. 0.
                       ''')

        # Initialise to all highly uncertain
        self.P = np.matrix(np.eye(4)*100)

        # Self motion -  we won't work the motion term in this example
        self.motion = np.matrix('0. 0. 0. 0.').T

        # Initialise the process covariance
        self.Q = np.matrix(np.eye(4))

        # Initialise the measurement covariance
        self.R = np.zeros(shape=(2,2))

    def kalmanFilter(self, box, R):
        # build z-term by getting box centres
        z = np.zeros(shape=(1,2))
        z[0][0] = (box[0][0] + box[1][0])/2
        z[0][1] = (box[0][1] + box[1][1])/2

        x = self.xState

        # Update step
        S = self.H*self.P*self.H.T + R
        K = self.P*self.H.T*S.I         # Kalman Gain
        y = np.matrix(z).T - self.H*x   # residual term
        x = x + K*y
        I = np.matrix(np.eye(self.F.shape[0]))
        self.P = (I - K*self.H)*self.P

        # Predict Step
        # Predict x and P based on measurement
        x = self.F*x + self.motion
        self.P = self.F*self.P*self.F.T +self.Q

        self.xState = x

    def box2xstate(self, box):
        # convert np.(2x2), [[x1, y1], [x2, y2]]
        # to state vector state_x [x, y, x_dot, y_dot]
        # by finding centre of box and using that as x,y
        self.xState[0] = (box[0][0] + box[1][0]) / 2 # centre x
        self.xState[1] = (box[0][1] + box[1][1]) / 2 # centre y

    def xstate2box(self):
        # use our  xState to update our box
        # by finding our box's centre, extracting
        # the new centre from xState and moving
        # our box by delta centres
        newCentre = np.zeros(shape=(2,1))
        newCentre[0] = self.xState[0]
        newCentre[1] = self.xState[1]
        oldCentre = np.zeros(shape=(2,1))
        oldCentre[0] = (self.box[0][0] + self.box[1][0]) / 2 # centre x
        oldCentre[1] = (self.box[0][1] + self.box[1][1]) / 2 # centre y

        return self.adjustBBox(self.box, oldCentre, newCentre)

    def adjustBBox(self, box, origCentre, newCentre):
        # Just move any box from oldCentre to newCentre
        delta = newCentre - origCentre
        adjustedBox = np.zeros(shape=(2,2))
        adjustedBox[0][0] = box[0][0] + delta[0]
        adjustedBox[0][1] = box[0][1] + delta[1]
        adjustedBox[1][0] = box[1][0] + delta[0]
        adjustedBox[1][1] = box[1][1] + delta[1]
        return adjustedBox

IMGDIR="./images/lab7/multiple-objects/"

# Find pngs and bounding boxes for pngs
pngDir = IMGDIR
bbDir = IMGDIR


def getPngsAndBoxes():
    global pngDir
    global bbDir

    pngFolder = os.fsencode(pngDir)
    bbFolder = os.fsencode(bbDir)

    pngFiles = []
    for filename in os.listdir(pngFolder):
        if filename.decode().endswith(".png"):
            pngFiles.append(pngDir + filename.decode())
    pngFiles.sort()

    for filename in os.listdir(bbFolder):
        if filename.decode().endswith(".boxes"):
            bbFilename = bbDir + filename.decode()

    bbfh = open(bbFilename, "r")
    bbLines = bbfh.readlines()
    bbfh.close()

    return bbLines, pngFiles

def parseDetections(bBoxes, pngFile, img,minBoxArea):

    index = int(re.findall(r'\d+', pngFile)[-1])

    imgH, imgW = img.shape[:2]

    centreList = []
    boxList = []
    confList = []

    for line in bBoxes:
        # ((x_plus_w+x)/2)/image.shape[1] # width
        # ((y_plus_h+y)/2)/image.shape[0] # height
        # (x_plus_w - x)/image.shape[1]
        # (y_plus_h - y)/image.shape[0]
        lineArray = np.genfromtxt(io.StringIO(line), delimiter=",")
        lineIndex = int(lineArray[0])
        if lineIndex == index:
            centre = np.zeros(shape=(2,1))
            box = np.zeros(shape=(2,2))
            conf = 0.000001 # hack to avoid div by zero
            centre = lineArray[2:4]
            halfW = lineArray[4] * imgW / 2
            halfH = lineArray[5] * imgH / 2
            conf += lineArray[6]
            centre[0] *= imgW
            centre[1] *= imgH
            box[0][0] = centre[0] - halfW   # x1
            box[0][1] = centre[1] - halfH   # y1
            box[1][0] = centre[0] + halfW   # x2
            box[1][1] = centre[1] + halfH   # y2
            boxW = halfW * 2
            boxH = halfH * 2
            boxArea = boxW * boxH
            if boxArea > minBoxArea:                # dump small boxes
                confList.append(conf)
                boxList.append(box)
                centreList.append(centre.tolist())

    return centreList, boxList, confList

#
# Helper Function
#
# Draw a box with a label.  Default label is 'untracked'
#
def drawBoxLabel(img, bbox, color=(0,255,255), label="Untracked"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontSize = 1.2

    cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), color, 8)
    cv2.putText(img, label, (int(bbox[0][0])-25,int(bbox[0][1])-25), font, fontSize, color, 8, cv2.LINE_AA)


def euclidean_distance(xDist, yDist):
    # square root of x squared plus y squared
    cost = xDist ** 2 + yDist ** 2
    cost = math.sqrt(cost)

    return cost


def mahalanobis_distance(box1, box2, ):
    covariance_matrix = np.cov(np.vstack([box1, box2]).T)

    inverse_covariance = np.linalg.inv(covariance_matrix)
    # NOTE: attempted algorithm using scipy source but decided to go
    # with scipy implementation
    # delta = box1 - box2
    # distance = np.dot(np.dot(delta.T, inverse_covariance), delta.T)
    xDist = distance.mahalanobis(box1[0], box2[0], inverse_covariance)
    yDist = distance.mahalanobis(box1[1], box2[1], inverse_covariance)

    # square root of x squared plus y squared
    cost = xDist ** 2 + yDist ** 2
    cost = math.sqrt(cost)

    return cost


def boxCost(box1, box2, threshold_flag, x_thresh, y_thresh, distance_flag):
    # width and height of box1
    # get centre of box1
    w1 = box1[1][0] - box1[0][0]
    w1 = w1 / 2
    cx1 = box1[0][0] + w1
    h1 = box1[1][1] - box1[0][1]
    h1 = h1 / 2
    cy1 = box1[0][1] + h1

    # width and height of box2
    # get centre of box2
    w2 = box2[1][0] - box2[0][0]
    w2 = w2 / 2
    cx2 = box2[0][0] + w2
    h2 = box2[1][1] - box2[0][1]
    h2 = h2 / 2
    cy2 = box2[0][1] + h2

    xDist = abs(cx2 - cx1)
    yDist = abs(cy2 - cy1)

    # Experiment 2.3: Additional Experiment to cap the distances
    # Very Crude
    if threshold_flag == True:
        if xDist >= x_thresh:
            xDist = float(x_thresh)
        if yDist >= y_thresh:
            yDist = float(y_thresh)

    # Experiment 3.1: Mahalanobis Distance
    if distance_flag == 'mahalanobis':
        cost = mahalanobis_distance(box1, box2)
    else:
        cost = euclidean_distance(xDist, yDist)

    # Invert to get what we need for
    # this implementation of Hungarian
    if cost == 0:
        cost = 1
    else:
        cost = 1 / cost  # bigger cost if closer
    return cost


def assignDetectionsToTrackers(trackers, detections, ndThreshold, threshold_flag, x_thresh, y_thresh, distance_flag):
    # Build a cost matrix - all zeros (size determined by num trackers and detections
    # Set it up as float to match our cost function
    costMatrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

    # Fill the cost matrix with 'prices' derived from the
    # cost term
    # A cost for every combination of tracker and new detection
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            costMatrix[t, d] = boxCost(trk, det, threshold_flag, x_thresh, y_thresh, distance_flag)

    # Produce matches
    # Solve the maximising of the sum of cost asignment using the
    # Hungarian algorithm (aka Munkres algorithm)
    matchedRowIdx, matchedColIdx = linear_sum_assignment(-costMatrix)

    # First of all find any tracker that didn't find a date
    # with a new detection at all
    # add it to the unmatchedTrackers list
    # Maybe that object has gone away ...
    unmatchedTrackers, unmatchedDetections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matchedRowIdx):
            unmatchedTrackers.append(t)

    # Now find any detection that didn't find a date
    # with an old trackeer  at all
    # add it to the unmatchedDetections list
    # Maybe its a new object
    for d, det in enumerate(detections):
        if (d not in matchedColIdx):
            unmatchedDetections.append(d)

    # Now, look at the matches in more detail
    # Maybe there's a few matches that are not
    # going to work
    matches = []

    # If the cost is than nd_theshold then
    # override the match - its not good enough
    # If you change the cost function, you'll probably
    # need to change ndThreshold as well
    for m, _ in enumerate(matchedRowIdx):
        if (costMatrix[matchedRowIdx[m], matchedColIdx[m]] < ndThreshold):
            # Nope, not really a match
            # add the detection to unmatched detections list
            # add the tracker to unmatched tracker list
            unmatchedTrackers.append(matchedRowIdx[m])
            unmatchedDetections.append(matchedColIdx[m])
        else:
            # Its a match
            # Record details of the match - tracker index and detection index
            match = np.empty((1, 2), dtype=int)
            match[0][0] = matchedRowIdx[m]
            match[0][1] = matchedColIdx[m]
            # Add to matches list
            matches.append(match)

    # Clean and return
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatchedDetections), np.array(unmatchedTrackers)

def demoHungarian(trackerId,writer,trackers,
                  maxAge, ndThreshold, minBoxArea,
                  threshold_flag, x_thresh, y_thresh, distance_flag):

    # Initialise state to no position
    x = np.matrix('0. 0. 0. 0.').T
    # Initialise state uncertainty covariance
    P = np.matrix(np.eye(4))*100

    # Create an empty box
    box = np.zeros(shape=(2,2))

    # Get lists of files and bounding boxes
    bbLines, pngFiles = getPngsAndBoxes()

    # Main loop - do this for every image in the directory
    for pngFile in pngFiles:
        #print ("handling .." + os.path.basename(pngFile))
        # Load the file
        img = cv2.imread(pngFile)

        # Gather a list of new boxes and confidence values
        # We'll use this confidence in R term of Kalman
        # Derive R from yolo confidence level in detection
        _, newBoxes, newConfs = parseDetections(bbLines, pngFile, img,minBoxArea)

        # Build our known boxes list by extracting it from our list
        # of tracker objects - each tracker has a box its minding for us
        knownBoxes = []

        if(len(trackers) > 0):
            for trk in trackers:
                knownBoxes.append(trk.box)

        # Now we have a list of old boxes being tracked and a
        # list of new boxes.
        # Hand over to assignment function to build our
        # three lists - matched, unmatched detections and unmatched trackers
        matched, unmatchedDetections, unmatchedTrackers \
        = assignDetectionsToTrackers(knownBoxes, newBoxes,ndThreshold,
                                     threshold_flag, x_thresh, y_thresh, distance_flag)

        # Deal with matched detections
        if(matched.size > 0):
            for trkIdx, detIdx in matched:
                # there was a match
                # new data for tracked object
                box = newBoxes[detIdx]
                conf = newConfs[detIdx]
                R = np.eye(2)
                R *= 1/conf
                # find tracker in list
                tmpTrk = trackers[trkIdx]
                # update its data and run a kalman filter
                tmpTrk.kalmanFilter(box, R)
                tmpTrk.box = box
                knownBoxes[trkIdx] = tmpTrk.box
                tmpTrk.numHits += 1
                tmpTrk.numMisses = 0

        # Deal with unmatched detections
        if (len(unmatchedDetections)>0):
            for idx in unmatchedDetections:
                box  = newBoxes[idx]
                tmpTrk = Tracker() # create a new tracker
                tmpTrk.box = box
                tmpTrk.box = tmpTrk.xstate2box()
                tmpTrk.id = trackerId # assign ID to tracker
                trackerId += 1
                trackers.append(tmpTrk)
                knownBoxes.append(tmpTrk.box)

        # Deal with unmatched tracks
        if (len(unmatchedTrackers)>0):
            for trkIdx in unmatchedTrackers:
                tmpTrk = trackers[trkIdx]
                tmpTrk.numMisses += 1
                tmpTrk.box = tmpTrk.xstate2box()
                knownBoxes[trkIdx] = tmpTrk.box

        # The list of tracks to be displayed
        for trk in trackers:
            if ((trk.numHits >= minHits) and (trk.numMisses <= maxAge)):
                drawBoxLabel(img, trk.box, label="Tracked " + str(trk.id))

        # clean up deleted tracks
        trackers = [x for x in trackers if x.numMisses <= maxAge]

        # Resize and show the image
        vidout = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))

        # Build a frame of our output video
        if writer is None:
            # Initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            writer = cv2.VideoWriter('video.webm', fourcc, 30, (vidout.shape[1], vidout.shape[0]), True)

        # Write the output frame to disk
        writer.write(vidout)

    # Release the file pointers
    writer.release()


def arrayShow(imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))


def video_processing(videoBodge):
    if (videoBodge == 0):
        from IPython.display import HTML
        from base64 import b64encode
        webm = open('video.webm', 'rb').read()
        data_url = "data:video/webm;base64," + b64encode(webm).decode()
    else:
        video = cv2.VideoCapture("video.webm")
        while (video.isOpened()):
            clear_output(wait=True)
            ret, frame = video.read()
            if (ret == False):
                break
            lines, columns, _ = frame.shape
            img = arrayShow(frame)
            display(img)
            time.sleep(1)

    # Display Video
    display_video = HTML("""
    <video width=200 controls>
          <source src="%s" type="video/webm">
    </video>
    """ % data_url)

    return display_video


# Initialise an OpenCV video writer object
writer = None
maxAge = 4 # number of consecutive frames containing an unmatched detection before a track is deleted
minHits = 1 # number of consecutive matches needed to establish a new track
ndThreshold = 0.0003    # if the cost of a 'match' between a detection and a tracker
                        # is below this - its not the same object and it needs its own tracker

minBoxArea = 40000      # Don't track boxes below this value - too small

# a list for tracker ids
trackerId = 1   # increment this to give an identity to new objects in the image stream
# Main Object - our tracker list
trackers = [] # the tracker list

threshold_flag = False
x_thresh =0
y_thresh =0
distance_flag ='euclidean'
videoBodge = 0
demoHungarian(trackerId,writer,trackers,maxAge,ndThreshold,minBoxArea,
              threshold_flag, x_thresh, y_thresh, distance_flag)

hungarian_demo_video = video_processing(videoBodge)
hungarian_demo_video