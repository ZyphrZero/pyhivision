#!/usr/bin/env python

"""Processors"""

from processors.adjustment import AdjustmentProcessor
from processors.background import BackgroundProcessor
from processors.beauty import BeautyProcessor
from processors.detection import DetectionProcessor
from processors.layout import LayoutProcessor
from processors.matting import MattingProcessor
from processors.template import TemplateProcessor
from processors.thin_face import ThinFaceProcessor
from processors.watermark import WatermarkProcessor

__all__ = [
    'MattingProcessor',
    'DetectionProcessor',
    'AdjustmentProcessor',
    'BeautyProcessor',
    'BackgroundProcessor',
    'WatermarkProcessor',
    'TemplateProcessor',
    'LayoutProcessor',
    'ThinFaceProcessor',
]
