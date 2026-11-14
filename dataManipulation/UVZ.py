# Best performance using python 2 with ros

import enum
import numpy as np
import pandas as pd
import collections
import sys
import rosbag
import os
import cv2

from cv_bridge import CvBridge
BRIDGE = CvBridge()

#  0,1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
#{-1,0,-1,5,5,6,7,8,9,9,10,11,12,13,13,14,15,16,-1,17,17,18,19,20, 1 ,1 ,2, 2, 3, 4}

def extract(bagFile, outFile):
    '''Extracts the UVZ, XZY, and images from a bag file
    '''
    bag = rosbag.Bag(bagFile)
    uvzDF = pd.DataFrame()
    xyzDF = pd.DataFrame()
    imageCount = 0
    for topic, msg, t in bag.read_messages():
        if 'image' in topic.lower():
            image = BRIDGE.imgmsg_to_cv2(msg)
            cv2.imwrite(os.path.join(outFile, 'image{:05d}.png'.format(imageCount)), image)
            imageCount += 1
        else:
            sxy, suv = extractJointPos(msg)
            uvzDF = uvzDF.append(suv, ignore_index=True)
            xyzDF = xyzDF.append(sxy, ignore_index=True)

    uvzDF.to_csv(os.path.join(outFile, 'UVZ.csv'))
    xyzDF.to_csv(os.path.join(outFile, 'XYZ.csv'))

def extractJointPos(msg_poseArray):
    '''Extracts the positions from a msgs_PoseArray message as a pandas series
    msg_poseArray: the ros message containing the positions
    
    returns the position data as a series
    returns the uvz data as a series
    '''
    poseArray = msg_poseArray.poses
    colNames = 'joint{}_{}'
    validPoses = [-1,0,-1,5,5,6,7,8,9,9,10,11,12,13,13,14,15,16,-1,17,17,18,19,20, 1 ,1 ,2, 2, 3, 4]
    validSize = 21

    resultXY = pd.Series()
    resultUV = pd.Series()
    for poseNum, pos in enumerate(poseArray):
        if poseNum < validSize:
            for ind in findAll(validPoses, poseNum):
                resultUV[colNames.format(ind, 'x')] = pos.position.x
                resultUV[colNames.format(ind, 'y')] = pos.position.y
                resultUV[colNames.format(ind, 'z')] = pos.position.z
        
        else:
            for ind in findAll(validPoses, poseNum-validSize):
                resultXY[colNames.format(ind, 'x')] = pos.position.x
                resultXY[colNames.format(ind, 'y')] = pos.position.y
                resultXY[colNames.format(ind, 'z')] = pos.position.z

    return resultXY, resultUV

def findAll(list, elm):
    '''Returns a list of all indexes where matches occur
    '''
    result = []
    for i, e in enumerate(list):
        if e == elm:
            result.append(i)
    return result


if __name__ == '__main__':
    extract('uvTest.bag', 'testingsUVExtract')