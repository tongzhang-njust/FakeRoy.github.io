import random
import os
import numpy as np
import cv2
from PIL import Image
from PIL import  ImageFile
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import glob

import time
import xml.etree.ElementTree as ET
ImageFile.LOAD_TRUNCATED_IMAGES = True

imagepath="D:/github desktop/FakeRoy.github.io/resources/PersonalInformation/photos/"
imgsavepath="D:/github desktop/FakeRoy.github.io/resources/PersonalInformation/photos/resized/"

