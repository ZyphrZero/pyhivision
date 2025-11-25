#!/usr/bin/env python

"""Processors"""

from pyhivision.processors.adjustment import AdjustmentProcessor
from pyhivision.processors.alignment import AlignmentProcessor
from pyhivision.processors.background import BackgroundProcessor
from pyhivision.processors.beauty import BeautyProcessor
from pyhivision.processors.detection import DetectionProcessor
from pyhivision.processors.layout import LayoutProcessor
from pyhivision.processors.matting import MattingProcessor
from pyhivision.processors.template import TemplateProcessor
from pyhivision.processors.thin_face import ThinFaceProcessor
from pyhivision.processors.watermark import WatermarkProcessor

__all__ = [
    'MattingProcessor',
    'DetectionProcessor',
    'AdjustmentProcessor',
    'AlignmentProcessor',
    'BeautyProcessor',
    'BackgroundProcessor',
    'WatermarkProcessor',
    'TemplateProcessor',
    'LayoutProcessor',
    'ThinFaceProcessor',
]
