#!/usr/bin/env python

"""Detection models"""
from models.detection.mtcnn import MTCNNModel
from models.detection.retinaface import RetinaFaceModel

__all__ = [
    'MTCNNModel',
    'RetinaFaceModel',
]
