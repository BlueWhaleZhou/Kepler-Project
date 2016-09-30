import pandas as pd
import numpy as np
import shutil
import glob
source = '/home/qinghai/research/kepler/0021_superEarths/'
target = '/home/qinghai/research/kepler/'

info = pd.read_csv('/home/qinghai/research/kepler/0021_superEarths_info.csv')
print info

for i in range(len(info)):
    star_id = '00' + str(int(info.values[i][0]))
    period = str(int(info.values[i][1]))
    for filename in glob.glob(os.path.join(source, star_id + '*.fits')):
        shutil.copyfile(filename, target + period + 'days/')
