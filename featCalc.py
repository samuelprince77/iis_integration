#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for calculating features from 3D landmarks.
"""

__version__ = "0.5.1"
__author__ = (
    "Robin Larsson, Daniel Norell,"
    " Sebastian Nödtvedt & Robert Rosborg"
)

import numpy as np
from scipy.spatial import distance

# landmarkToIndex: Dictionary mapping 3D landmarks to indices.
lTI = {
    1: 5,
    3: 4,
    5: 3,
    6: 2,
    8: 1,
    10: 0,
    15: 14,
    17: 13,
    19: 12,
    20: 9,
    23: 8,
    26: 7,
    29: 6,
    32: 17,
    35: 16,
    38: 15,
    41: 20,
    45: 19,
    48: 18,
}


def calcLipStates(scale, point1, point2):
    """
    Calculate lip states.

    Args:
        scale (float): The scale to divide the distances with.
        point1 (numpy.ndarray): 1D array containing the coordinates of landmark 32 as float.
        point2 (numpy.ndarray): 1D array containing the coordinates of landmark 38 as float.

    Returns:
        float: The scaled euclidean distance between point1 and point2.
    """
    return distance.euclidean(point1, point2) / scale


def calcMouthOpen(scale, point1, point2):
    """
    Calculate mouth open.

    Args:
        scale (float): The scale to divide the distances with.
        point1 (numpy.ndarray): 1D array containing the coordinates of landmark 35 as float.
        point2 (numpy.ndarray): 1D array containing the coordinates of landmark 41 as float.

    Returns:
        float: The scaled euclidean distance between point1 and point2.
    """
    return distance.euclidean(point1, point2) / scale


def calcFaceZoom(scale, point1, point2):
    """
    Calculate face zoom.

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 23 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 26 as float.

    Returns:
        float: The scaled euclidean distance between point1 and point2.
    """
    return distance.euclidean(point1, point2) / scale


def calcSeparation(scale, point1, point2, point3, point4,
                   point5, point6, point7, point8):
    """
    Calculate eye-eyebrow separation.

    Args:
        scale(float): The scale to divide the distances with.
        point1 (numpy.ndarray): 1D array containing the coordinates of landmark 5 as float.
        point2 (numpy.ndarray): 1D array containing the coordinates of landmark 17 as float.
        point3 (numpy.ndarray): 1D array containing the coordinates of landmark 3 as float.
        point4 (numpy.ndarray): 1D array containing the coordinates of landmark 15 as float.
        point5 (numpy.ndarray): 1D array containing the coordinates of landmark 6 as float.
        point6 (numpy.ndarray): 1D array containing the coordinates of landmark 19 as float.
        point7 (numpy.ndarray): 1D array containing the coordinates of landmark 8 as float.
        point8 (numpy.ndarray): 1D array containing the coordinates of landmark 20 as float.

    Returns:
        float: The distance between point1 and point2, and point3 and point4,
               the mean distance between point5 and pont6 and lastly point7 and point8 is added.
    """
    dist1 = distance.euclidean(point1, point2) / scale
    dist2 = distance.euclidean(point3, point4) / scale
    dist3 = distance.euclidean(point5, point6) / scale
    dist4 = distance.euclidean(point7, point8) / scale

    distMean = np.mean([dist3, dist4])

    result = dist1 + dist2 + distMean

    return result


def calcFirstLipShape(scale, point1, point2, point3, point4):
    """
    Calculate first lip shape (around).

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 32 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 35 as float.
        point3(numpy.ndarray): 1D array containing the coordinates of landmark 38 as float.
        point4(numpy.ndarray): 1D array containing the coordinates of landmark 41 as float.

    Returns:
        float: The distance between all the points.
    """
    dist1 = distance.euclidean(point1, point2) / scale
    dist2 = distance.euclidean(point2, point3) / scale
    dist3 = distance.euclidean(point3, point4) / scale
    dist4 = distance.euclidean(point4, point1) / scale

    result = dist1 + dist2 + dist3 + dist4

    return result


def calcSecondLipShape(scale, point1, point2, point3):
    """
    Calculate second lip shape (left eye).

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 26 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 32 as float.
        point3(numpy.ndarray): 1D array containing the coordinates of landmark 38 as float.

    Returns:
        float: The distance point1 to point2 and point1 to point3.
    """
    dist1 = distance.euclidean(point1, point2) / scale
    dist2 = distance.euclidean(point1, point3) / scale

    result = dist1 + dist2

    return result


def calcEyebrow(scale, point1, point2):
    """
    Calculate eyebrow.

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 5 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 6 as float.

    Returns:
        float: The scaled euclidean distance point1 to point2.
    """
    return distance.euclidean(point1, point2) / scale


def calcEyebrowRaise(scale, point1, point2, point3, point4):
    """
    Calculate eyebrow raise.

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 5 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 17 as float.
        point3(numpy.ndarray): 1D array containing the coordinates of landmark 6 as float.
        point4(numpy.ndarray): 1D array containing the coordinates of landmark 19 as float.

    Returns:
        float: The eyebrow that is raised the most compared to the other.

    """
    dist1 = distance.euclidean(point1, point2) / scale
    dist2 = distance.euclidean(point3, point4) / scale

    result = max(dist1, dist2)

    return result


def calcHeadYaw(scale, point1, point2, point3, point4):
    """
    Calculate head yaw.

    Args:
        scale(float): The scale to divide the distances with.
        point1(numpy.ndarray): 1D array containing the coordinates of landmark 15 as float.
        point2(numpy.ndarray): 1D array containing the coordinates of landmark 17 as float.
        point3(numpy.ndarray): 1D array containing the coordinates of landmark 19 as float.
        point4(numpy.ndarray): 1D array containing the coordinates of landmark 20 as float.

    Returns:
        float: The scaled euclidean distance between point1 to point2 and point3 to point 4.
    """
    dist1 = distance.euclidean(point1, point2) / scale
    dist2 = distance.euclidean(point3, point4) / scale

    result = dist1 / dist2

    return result

def calcNoseDistances(points):
    """
    Calculate the distances from the other landmarks to the nose.

    Args:
        points(numpy.ndarray): 2D array containing the coordinates of landmarks as floats.

    Returns:
        float: The scaled euclidean distance between the nose and respective points.
    """
    noseIndex = lTI[17]
    distances = []

    for i in range(len(points)):
        if i != noseIndex:
            distances.append(distance.euclidean(points[noseIndex], points[i]))

    return distances


def calcFeats(data, normalize=True):
    """
    Calculate features from the 3D landmarks in data.

    Args:
        data(numpy.ndarray): 2D array containing the features.
        normalize(boolean): Decides if scale will either be the mean of the nose distance or 1.0.

    Returns:
        numpy.ndarray: A 2D array of the features.
    """
    nFeatures = 9
    nData = len(data)
    features = np.zeros((nData, nFeatures))

    for i in range(nData):
        scale = np.mean(calcNoseDistances(data[i])) if normalize else 1.0

        features[i, 0] = calcLipStates(scale, data[i][lTI[32]], data[i][lTI[38]])

        features[i, 1] = calcMouthOpen(scale, data[i][lTI[35]], data[i][lTI[41]])

        features[i, 2] = calcFaceZoom(scale, data[i][lTI[23]], data[i][lTI[26]])

        features[i, 3] = calcSeparation(scale, data[i][lTI[5]], data[i][lTI[17]],
                                        data[i][lTI[3]], data[i][lTI[15]],
                                        data[i][lTI[6]], data[i][lTI[19]],
                                        data[i][lTI[8]], data[i][lTI[20]])

        features[i, 4] = calcFirstLipShape(scale, data[i][lTI[32]],
                                           data[i][lTI[35]],
                                           data[i][lTI[38]],
                                           data[i][lTI[41]])

        features[i, 5] = calcSecondLipShape(scale, data[i][lTI[26]],
                                            data[i][lTI[32]],
                                            data[i][lTI[38]])

        features[i, 6] = calcEyebrow(scale, data[i][lTI[5]], data[i][lTI[6]])

        features[i, 7] = calcEyebrowRaise(scale, data[i][lTI[5]], data[i][lTI[17]],
                                          data[i][lTI[6]], data[i][lTI[19]])

        features[i, 8] = calcHeadYaw(scale, data[i][lTI[15]], data[i][lTI[17]],
                                     data[i][lTI[19]], data[i][lTI[20]])

    return features


def calcSimpleFeats(data, normalize=True):
    """
    Calculate simple features from the 3D landmarks in data.

    Each feature is the distance from a landmark to the nose tip landmark.

    Args:
        data(numpy.ndarray): 2D array containing the features.
        normalize(boolean): Decides if scale will either be the mean of the nose distance or 1.0.

    Returns:
         numpy.ndarray: A 2D array of the features.

    """
    nFeatures = 21
    nSamples = len(data)
    features = np.zeros((nSamples, nFeatures))

    for i in range(nSamples):
        scale = np.mean(calcNoseDistances(data[i])) if normalize else 1.0
        features[i] = calcNoseDistances(data[i])

        for j in range(nFeatures):
            features[i, j] = features[i, j] / scale

    return features
