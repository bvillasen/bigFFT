import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.fft import fftn, ifftn


inDir = '/home/bruno/Desktop/data/fft/'
outDir = '/home/bruno/Desktop/data/fft/'
name_base = 'h5'
outFileName = 'data.h5'

dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find('.h5.') > 0 ) ) ]
dataFiles = np.sort( dataFiles )
nFiles = len( dataFiles )


def split_name( file_name):
  nSap, name, nBox = file_name.split('.')
  return [int(nSap), int(nBox)]

files_names = np.array([ split_name( file_name ) for file_name in dataFiles ])
snaps, boxes = files_names.T
snaps = np.unique( snaps )
boxes = np.unique( boxes )
nSnapshots = len( snaps )
nBoxes = len( boxes )

print "Number of boxes: {0}".format(nBoxes)
print "Number of snapshots: {0}".format(nSnapshots)

snap_id = 0
box_id = 15
inFileName = '{0}.{1}.{2}'.format(snap_id, name_base, box_id)
inFile = h5py.File( inDir + inFileName, 'r')

data = inFile['density'][...]

data_gather = inFile['slab_gather'][...]
slab_data = inFile['slab'][...]

data2_gather = inFile['slab2_gather'][...]
slab2_data = inFile['slab2'][...]

data3_gather = inFile['slab3_gather'][...]
slab3_data = inFile['slab3'][...]

data4_gather = inFile['slab4_gather'][...]
slab4_data = inFile['slab4'][...]
inFile.close()
