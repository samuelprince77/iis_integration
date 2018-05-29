#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for parsing lm3 files and returning 3D landmarks.
"""

__version__ = "0.3.4"
__author__ = (
    "Robin Larsson, Daniel Norell,"
    " Sebastian Nödtvedt & Robert Rosborg"
)


def parseLm3File(filePath):
    """
    Parse the lm3 file at filePath and return the 3D landmarks.

    Args:
        filePath(string): A string of the filepath to the lm3 file.

    Returns:
        array: A array of the 3D landmarks.
    """
    landmarkNames = {
        "Outer left eyebrow",
        "Middle left eyebrow",
        "Inner left eyebrow",
        "Inner right eyebrow",
        "Middle right eyebrow",
        "Outer right eyebrow",
        "Outer left eye corner",
        "Inner left eye corner",
        "Inner right eye corner",
        "Outer right eye corner",
        "Nose saddle left",
        "Nose saddle right",
        "Left nose peak",
        "Nose tip",
        "Right nose peak",
        "Left mouth corner",
        "Upper lip outer middle",
        "Right mouth corner",
        "Upper lip inner middle",
        "Lower lip inner middle",
        "Lower lip outer middle",
        "Chin middle",
    }

    with open(filePath) as f:
        lines = f.readlines()

    landmarks = []

    for i in range(4, len(lines), 2):
        if lines[i - 1].rstrip() in landmarkNames:
            landmark = [float(j) for j in lines[i].split()]
            landmarks.append(landmark)

    nlandmarks = len(landmarks)
    if nlandmarks != 22:
        return False

    return landmarks
