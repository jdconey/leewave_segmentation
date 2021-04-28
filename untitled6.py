# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:43:15 2021

@author: mm16jdc
"""

import os

m2=[]
lw2 = []
masklist = os.listdir('C:/Users/mm16jdc/Documents/ukv_data/data/segmentation_full/masks2')
for k in masklist:
    m2.append(k[:20])
lwpathlist = os.listdir('C:/Users/mm16jdc/Documents/ukv_data/data/segmentation_full/lw_paths')
for k in lwpathlist:
    lw2.append(k[:20])