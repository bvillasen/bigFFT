import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

#
# # Get all boxes from first snap
# snapNum = 0
# boxNum = 0
# boxesNames = []
# for fileName in dataFiles:
#   nSnap, nBox = split_name( dataFiles[boxNum], name_base )
#   boxesNames.append( str(nSnap) + '.' + name_base + '.' + str(boxNum) )
#   if nSnap != snapNum: break
#   boxNum += 1
# nBoxes = len( boxesNames )
# nSnapshots = nFiles / nBoxes
print "Number of boxes: {0}".format(nBoxes)
print "Number of snapshots: {0}".format(nSnapshots)

snap_id = 0
box_id = 0
inFileName = '{0}.{1}.{2}'.format(snap_id, name_base, box_id)
inFile = h5py.File( inDir + inFileName, 'r')
head = inFile.attrs
dims_all = head['dims']
dims_local = head['dims_local']
nproc_z, nproc_y, nproc_x = head['proc_grid']
inFile.close()

outFile = h5py.File( outDir + outFileName, 'w')
  # nSnap = 0
for nSnap in range(nSnapshots):
  fileSnap = outFile.create_group( str(nSnap) )
  keys = [ 'density', 'slab4' ]
  print ' snap: {0}  {1}'.format( nSnap, keys )
  for key in keys:
    # print key
    data_all = np.zeros( dims_all, dtype=np.complex )
    for nBox in range(nBoxes):
      inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
      inFile = h5py.File( inDir + inFileName, 'r')
      head = inFile.attrs
      procID_z, procID_y, procID_x = head['proc_coord']
      procStart_z, procStart_y, procStart_x = head['proc_offset']
      procEnd_z, procEnd_y, procEnd_x = head['proc_offset'] + head['dims_local']
      # procDomain_l = head['proc_domain_l']
      # print head['proc_offset'], head['proc_offset'] + head['dims_local'], procDomain_l
      data_local = inFile[key][...]
      data_all[ procStart_z:procEnd_z, procStart_y:procEnd_y, procStart_x:procEnd_x] = data_local
      # data_all[pro ]
      inFile.close()
    fileSnap.create_dataset( key, data=data_all )

outFile.close()


#
#
#
# data_all = np.zeros( dims_all, dtype=np.float32 )
# key = 'grav_density'
# # delta_x = head['dx']
# # dz, dy, dx = delta_x
# offset = head['offset']
# data_local = inFile[key][...]
# # inFile.close()

#Get


# outFileName = 'allData.h5'
# outFile = h5py.File( outDir + outFileName, 'w')
#
# nFiles = len( dataFiles )
# for nSnap in range( nFiles ):
#   inFileName = dataFiles[nSnap]
#   inFile = h5py.File( inDir + inFileName, 'r')
#   density = inFile['density'][...]
#   key = '{0:03}_dens'.format(nSnap)
#   outFile[key] = density.astype( np.float32 )
#   inFile.close()
#
# outFile.close()
