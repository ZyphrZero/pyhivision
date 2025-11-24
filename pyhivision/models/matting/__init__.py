#!/usr/bin/env python

"""Matting models"""
from pyhivision.models.matting.birefnet import BiRefNetModel
from pyhivision.models.matting.modnet import (
    HivisionModNetModel,
    ModNetModel,
    ModNetPhotographicModel,
)
from pyhivision.models.matting.rmbg import RMBGModel

__all__ = [
    'ModNetModel',
    'ModNetPhotographicModel',
    'HivisionModNetModel',
    'BiRefNetModel',
    'RMBGModel',
]
