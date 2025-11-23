#!/usr/bin/env python

"""Matting models"""
from models.matting.birefnet import BiRefNetModel
from models.matting.modnet import (
    HivisionModNetModel,
    ModNetModel,
    ModNetPhotographicModel,
)
from models.matting.rmbg import RMBGModel

__all__ = [
    'ModNetModel',
    'ModNetPhotographicModel',
    'HivisionModNetModel',
    'BiRefNetModel',
    'RMBGModel',
]
