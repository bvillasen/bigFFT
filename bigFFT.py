import sys, time, os
import numpy as np
import h5py as h5
from mpi4py import MPI

#Add Modules from other directories
devDir = '/home/bruno/Desktop/Dropbox/Developer/pyCUDA/'
toolsDirectory = devDir + "tools"
sys.path.extend( [toolsDirectory ] )
from tools import *
from mpiTools import *

nPoints = 128
useDevice = 0
usingAnimation = False
outDir = '/home/bruno/Desktop/data/fft/'
# ensureDirectory( outDir )

#Initialize MPI
MPIcomm = MPI.COMM_WORLD
pId = MPI.COMM_WORLD.Get_rank()
nProcess = MPIcomm.Get_size()
name = MPI.Get_processor_name()

if pId == 0:
  print "\nMPI-CUDA 3D FFT"
  print " nProcess: {0}\n".format(nProcess)
  time.sleep(0.1)
MPIcomm.Barrier()

nP_z, nP_y, nP_x = 2, 2, 2
pId_z, pId_y, pId_x = get_mpi_id_3D( pId, nP_x, nP_y )
pCoords = np.array([ pId_z, pId_y, pId_x ])
pParity = (pId_x + pId_y + pId_z) % 2

msg = ' Process coords: [ {0} {1} {2} ]'.format( pId_z, pId_y, pId_x )
print_mpi( msg , pId, nProcess, MPIcomm )

fileName = '0.h5.{0}'.format(pId)
outFile = h5.File( outDir + fileName, 'w')
head = outFile.attrs

#set simulation volume dimentions
nz_total = nPoints
ny_total = nPoints
nx_total = nPoints
dims_total = np.array([ nz_total, ny_total, nx_total ])

nz_local = nz_total / nP_z
ny_local = ny_total / nP_y
nx_local = nx_total / nP_x
dims_local = np.array([ nz_local, ny_local, nx_local ])

head['dims'] = dims_total
head['dims_local'] = dims_local
head['proc_grid']  = np.array([ nP_z, nP_y, nP_x ])
head['proc_coord'] = pCoords
head['proc_offset'] = pCoords * dims_local

#Global size
Lz, Ly, Lx = 1., 1., 1.
zMin, yMin, xMin = 0., 0., 0.

Lz_p, Ly_p, Lx_p = Lz/nP_z, Ly/nP_y, Lx/nP_x
dz_p, dy_p, dx_p = Lz_p/nz_local, Ly_p/ny_local, Lx_p/nx_local
zMin_p = zMin + pId_z*Lz_p + 0.5*dz_p
yMin_p = yMin + pId_y*Ly_p + 0.5*dy_p
xMin_p = xMin + pId_x*Lx_p + 0.5*dx_p
zMax_p = zMin_p + (nz_local-1)*dz_p
yMax_p = yMin_p + (ny_local-1)*dy_p
xMax_p = xMin_p + (nx_local-1)*dx_p
Z_p, Y_p, X_p = np.mgrid[ zMin_p:zMax_p:nz_local*1j, yMin_p:yMax_p:ny_local*1j, xMin_p:xMax_p:nx_local*1j ]

# msg = ' Process range: [ ( {0:0.3f} ,  {1:0.3f} ) , ( {2:0.3f} ,  {3:0.3f} ) , ( {4:0.3f} ,  {5:0.3f} ) ]'.format( zMin_p, zMax_p, yMin_p, yMax_p, xMin_p, xMax_p )
# print_mpi( msg , pId, nProcess, MPIcomm )

sphereR = 0.25
center_z = 0.5
center_y = 0.5
center_x = 0.5
r = np.sqrt( (Z_p - center_z)**2 + (Y_p - center_y)**2 + (X_p - center_x)**2 )
sphere = r < sphereR
rho = 0.6 * np.ones( [nz_local, ny_local, nx_local ])
rho[sphere] = 1


# Slabs made in the X-Y plane, Z axis is divided among processes
salab_axis = 0  # 0->Z, 1->Y, 2->X
slab_id = pId_z
nSlabs = nP_z
slab_np_x = nP_x
slab_np_y = nP_y
slab_procs_ids = [ pid for pid in range(nProcess) if get_mpi_id_3D(pid, nP_x, nP_y)[salab_axis] == slab_id ]
# msg = 'Slab ids: {0}'.format(slab_procs_ids)
# print_mpi( msg , pId, nProcess, MPIcomm )

slab_comm = MPIcomm.Split( slab_id , pId_z )
slab_pId = slab_comm.Get_rank()
slab_nProcs = slab_comm.Get_size()
msg = 'Slab_id: {0},  pId_slab: {1}, slab_nProcs: {2}'.format(slab_id, slab_pId, slab_nProcs)
print_mpi( msg , pId, nProcess, MPIcomm )
slab_nx = nx_total
slab_ny = ny_total
slab_nz = nz_local / slab_nProcs
slab_nx_local = nx_local
slab_ny_local = ny_local
# slab_pId = pId % nSlabs

data = rho

slab_data_gather = -1*np.ones( [slab_nProcs, slab_nz, slab_ny_local, slab_nx_local] )
for i,pId_temp in enumerate(slab_procs_ids):
  slab_domain_z_start, slab_domain_z_end = i*slab_nz, (i+1)*slab_nz
  slab_data_local =  data[slab_domain_z_start:slab_domain_z_end,:,:].copy()
  # print ( '[pId: {0}] Gathering slab {1}'.format(pId, slab_id))
  slab_comm.Gather( slab_data_local, slab_data_gather, root=i)
    # time.sleep(0.1)
  slab_comm.Barrier()
# slab_domain_z_start, slab_domain_z_end = slab_pId*slab_nz, (slab_pId+1)*slab_nz
# # msg = 'Slab_z:[ {0} , {1} ] '.format(slab_domain_z_start, slab_domain_z_end )
# # print_mpi( msg , pId, nProcess, MPIcomm )
# slab_data_local = -pId * data[slab_domain_z_start:slab_domain_z_end,:,:].copy()
# slab_data_gather = np.zeros( [slab_nProcs, slab_nz, slab_ny_local, slab_nx_local] )
# slab_comm.Allgather( slab_data_local, slab_data_gather )

slab_data = np.zeros( [slab_nz, slab_ny, slab_nx ])
for i,pId_temp in enumerate(slab_procs_ids):
  pId_z_temp, pId_y_temp, pId_x_temp = get_mpi_id_3D( pId_temp, nP_x, nP_y )
  slab_domain_x_start, slab_domain_x_end = pId_x_temp*nx_local, (pId_x_temp+1)*nx_local
  slab_domain_y_start, slab_domain_y_end = pId_y_temp*ny_local, (pId_y_temp+1)*ny_local
  slab_data[:,slab_domain_y_start:slab_domain_y_end,slab_domain_x_start:slab_domain_x_end] = slab_data_gather[i].copy()

stride = 1
outFile.create_dataset('density', data=rho[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('slab_gather', data=slab_data_gather[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('slab', data=slab_data[::stride,::stride,::stride].astype(np.float32))

MPIcomm.Barrier()
slab_send = slab_data.copy()
# slab_send = pId * np.ones_like( slab_data )

slab2_nx = nx_total
slab2_ny = ny_total / nProcess
slab2_nz = nz_total
slab2_nz_gather = nz_total/nProcess
slab2_gather = np.zeros([nProcess, slab2_nz_gather, slab2_ny, slab2_nx])
for pId_temp in range(nProcess):
  slab2_domain_y_start, slab2_domain_y_end = pId_temp*slab2_ny, (pId_temp+1)*slab2_ny
  slab2_data_local = slab_send[:,slab2_domain_y_start:slab2_domain_y_end,:].copy()
  MPIcomm.Gather( slab2_data_local, slab2_gather, root=pId_temp  )
  MPIcomm.Barrier()

slab2_data = np.zeros( [slab2_nz, slab2_ny, slab2_nx ])
for pId_temp in range(nProcess):
  slab2_domain_z_start, slab2_domain_z_end = pId_temp*slab2_nz_gather, (pId_temp+1)*slab2_nz_gather
  slab2_data[slab2_domain_z_start:slab2_domain_z_end,:,:] = slab2_gather[pId_temp].copy()

outFile.create_dataset('slab2_gather', data=slab2_gather[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('slab2', data=slab2_data[::stride,::stride,::stride].astype(np.float32))

slab2_send = slab2_data.copy()
slab3_gather = np.zeros([ nProcess, slab_nz, slab2_ny, slab_nx ])
for pId_temp in range(nProcess):
  slab2_domain_z_start, slab2_domain_z_end = pId_temp*slab2_nz_gather, (pId_temp+1)*slab2_nz_gather
  slab3_data_local = slab2_send[ slab2_domain_z_start : slab2_domain_z_end, :, : ]
  MPIcomm.Gather( slab3_data_local, slab3_gather, root=pId_temp  )
  MPIcomm.Barrier()

# slab3_data = np.zeros( [slab_nz, slab_ny, slab_nx ])
for pId_temp in range(nProcess):
  slab2_domain_y_start, slab2_domain_y_end = pId_temp*slab2_ny, (pId_temp+1)*slab2_ny
  slab_data[:,slab2_domain_y_start:slab2_domain_y_end,:] = slab3_gather[pId_temp].copy()

outFile.create_dataset('slab3_gather', data=slab3_gather[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('slab3', data=slab_data[::stride,::stride,::stride].astype(np.float32))

slab3_send = slab_data.copy()
slab4_gather = -1*np.ones( [slab_nProcs, slab_nz, slab_ny_local, slab_nx_local] )
for i,pId_temp in enumerate(slab_procs_ids):
  pId_z_temp, pId_y_temp, pId_x_temp = get_mpi_id_3D( pId_temp, nP_x, nP_y )
  slab_domain_x_start, slab_domain_x_end = pId_x_temp*nx_local, (pId_x_temp+1)*nx_local
  slab_domain_y_start, slab_domain_y_end = pId_y_temp*ny_local, (pId_y_temp+1)*ny_local
  slab4_data_local = slab3_send[:, slab_domain_y_start:slab_domain_y_end, slab_domain_x_start:slab_domain_x_end].copy()
  slab_comm.Gather( slab4_data_local, slab4_gather, root=i )

slab4_data = np.zeros( [nz_local, ny_local, nx_local] )
for i,pId_temp in enumerate(slab_procs_ids):
  slab_domain_z_start, slab_domain_z_end = i*slab_nz, (i+1)*slab_nz
  slab4_data[slab_domain_z_start:slab_domain_z_end,:,:] = slab4_gather[i]

outFile.create_dataset('slab4_gather', data=slab4_gather[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('slab4', data=slab4_data[::stride,::stride,::stride].astype(np.float32))


outFile.close()
