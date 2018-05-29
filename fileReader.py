#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for reading and parsing all lm3 files in a directory.
"""

__version__ = "0.3.5"
__author__ = (
    "Robin Larsson, Daniel Norell,"
    " Sebastian Nödtvedt & Robert Rosborg"
)

import os

from lm3Parser import parseLm3File


def readLm3Files(dataDir):
    """
    Read and parse all lm3 files in dataDir.

    Args:
        dataDir(string): Name of the directory of the files wiht facial features.

    Raises:
         KeyError: Raises an exception, if no files of type .lm3 is within the corresponding directory.

    Returns:
        float: A tuble of the data and labes with the corresponding image names.
    """
    location = os.getcwd()
    location += "/" + dataDir
    lm3Files = []
    data = []
    labels = []
    imageNames = []

    for lm3File in os.listdir(location):
        try:
            if lm3File.endswith(".lm3"):
                lm3Files.append(str(lm3File))

        except Exception as e:
            print("No files found here!")
            raise e

    for lm3File in lm3Files:
        lm3Data = parseLm3File(dataDir + "/" + lm3File)
        if lm3Data:
            data.append(lm3Data)
            labels.append(lm3File.split("_")[2])
            imageNames.append(lm3File)

    return (data, labels, imageNames)
