import numpy as np
import os
import glob

path_time = "/home/qinghai/testing/time/"
path_flux = "/home/qinghai/testing/flux/"

name_time = []
name_flux = []

for filename in glob.glob(os.path.join(path_time, '*.txt')):
    name_time.append(filename)
print name_time

for filename in glob.glob(os.path.join(path_flux, '*.txt')):
    name_flux.append(filename)
print name_flux
