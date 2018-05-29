#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for parsing argument vectors and setting variabels accordingly.
"""

__version__ = "0.2.1"
__author__ = "Daniel Norell"


def dictAssign(varDict, k, v):
    """
    Set the value of k in varDict to v.
    """
    varDict[k] = v


def dictToggle(varDict, k, v):
    """
    Set the value of k in varDict to True or False depending on v.
    """
    varDict[k] = v.lower() in {
        "true",
        "t",
        "on",
    }


def parseArgv(argv, varDict, funcDict):
    """
    Parse argv and use funcDict to set the variabels in varDict accordingly.
    """
    for arg in argv[1:]:
        kv = arg.split("=")
        if len(kv) == 2:
            k = kv[0].strip().lower()
            v = kv[1].strip()

            if k in funcDict:
                funcDict[k](varDict, k, v)
