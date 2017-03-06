#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <hdf5.h>
#include <mpi.h>
#include <fftw3.h>
#include <unistd.h>
#include <ctime>

using namespace std;

/*Global MPI Variables*/
int procID; /*process rank*/
int nproc;  /*number of processes in global comm*/
int nProc[3];      //number of process in each axis [Z,Y,X]
int procCoord[3];  //process coordinates [Z,Y,X]
int procOffset[3];      //process offset [Z,Y,X]
double procDomain_l[3];
double procDomain_r[3];
int nproc_x, nproc_y, nproc_z;
int procID_x, procID_y, procID_z;
int root;   /*rank of root process*/

int n[3];

int procID_node; /*process rank on node*/
int nproc_node;  /*number of MPI processes on node*/

MPI_Comm world; /*global communicator*/
MPI_Comm node;  /*global communicator*/

// Get the name of the processor
char processor_name[MPI_MAX_PROCESSOR_NAME];
int name_len;

void InitializeChollaMPI(int *pargc, char **pargv[]);

void get_mpi_id_3D( int pID, int np_x, int np_y, int &pID_x, int &pID_y, int &pID_z ){
  pID_x = pID % np_x;
  pID_z = pID / ( np_x * np_y );
  pID_y = ( pID - pID_z*np_y*np_x ) / np_x;
}

void Write_Header_HDF5(hid_t file_id, int nx_total, int ny_total, int nz_total,
      int nx_local, int ny_local, int nz_local);

void Write_Data_HDF5(hid_t file_id, string data_name, int nx, int ny, int nz, double *field_re );

void Write_Grid_HDF5(hid_t file_id, int nx, int ny, int nz, double *field_re );


int main(int argc, char **argv){

  /* Initialize MPI and PFFT */
  // MPI_Init(&argc, &argv);
  InitializeChollaMPI(&argc, &argv);
  get_mpi_id_3D( procID, nproc_x, nproc_y, procID_x, procID_y, procID_z);
  MPI_Barrier( world );
  procCoord[0] = procID_z;
  procCoord[1] = procID_y;
  procCoord[2] = procID_x;
  for (size_t i = 0; i < nproc; i++) {
    if (procID == i)  printf("Processor %d:  ( %d , %d , %d )\n", procID, procID_z, procID_y, procID_x );
    usleep(1e5);
  }

  /* Set size of FFT and process mesh */
  int nx_total, ny_total, nz_total;
  nz_total = 128;
  ny_total = 128;
  nx_total = 128;
  n[0] = nz_total; n[1] = ny_total; n[2] = nx_total;

  int nx_local, ny_local, nz_local, n_cells_local;
  nz_local = nz_total / nproc_z;
  ny_local = ny_total / nproc_y;
  nx_local = nx_total / nproc_x;
  n_cells_local = nz_local * ny_local * nx_local;
  if ( procID == 0 ) printf( " nCells total: %d %d %d\n", nz_total, ny_total, nx_total );
  if ( procID == 0 ) printf( " nCells local: %d %d %d\n", nz_local, ny_local, nx_local );


  string fileName = "0";
  string outDir = "/home/bruno/Desktop/data/bigFFT_output/";
  // create the filename
  ostringstream outFileName;
  outFileName << outDir << fileName << ".h5." << procID;
  // cout << outFileName.str() << endl;

  // Create a new file collectively
  hid_t   file_id;
  herr_t  status;
  file_id = H5Fcreate(outFileName.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


  double x_min_total, x_min_local, y_min_total, y_min_local, z_min_total, z_min_local;
  double length_x, length_y, length_z;
  double dx, dy, dz;
  x_min_total = 0;
  y_min_total = 0;
  z_min_total = 0;
  length_x = 1;
  length_y = 1;
  length_z = 1;
  dx = length_x / nx_total;
  dy = length_y / ny_total;
  dz = length_z / nz_total;
  x_min_local = procID_x * (length_x / nproc_x);
  y_min_local = procID_y * (length_y / nproc_y);
  z_min_local = procID_z * (length_z / nproc_z);

  procDomain_l[0] = z_min_local + 0.5*dz;
  procDomain_l[1] = y_min_local + 0.5*dy;
  procDomain_l[2] = x_min_local + 0.5*dx;
  procDomain_r[0] = z_min_local + 0.5*dz + (nz_local-1)*dz;
  procDomain_r[1] = y_min_local + 0.5*dy + (ny_local-1)*dy;
  procDomain_r[2] = x_min_local + 0.5*dx + (nx_local-1)*dx;
  procOffset[0] = procID_z * nz_local;
  procOffset[1] = procID_y * ny_local;
  procOffset[2] = procID_x * nx_local;

  for (size_t i = 0; i < nproc; i++) {
    if (procID == i)  printf("Processor %d: [ %f , %f ] [ %f %f ] [ %f %f ]\n", procID, procDomain_l[0], procDomain_r[0], procDomain_l[1], procDomain_r[1], procDomain_l[2], procDomain_r[2]);
    usleep(1e5);
  }

  Write_Header_HDF5( file_id, nx_total, ny_total, nz_total, nx_local, ny_local, nz_local );


  double *in_re ;
  in_re   = (double *) malloc(n_cells_local*sizeof(double));


  // set the initial values of the conserved variables
  int i, j, k, id;
  double x_pos, y_pos, z_pos;

  double center_x = 0.5;
  double center_y = 0.5;
  double center_z = 0.5;
  double sigma_x = 1.0;
  double sigma_y = 1.0;
  double sigma_z = 1.0;
  double r, dens;
  for (k=0; k<nz_local; k++) {
    for (j=0; j<ny_local; j++) {
      for (i=0; i<nx_local; i++) {
        id = i + j*nx_local + k*nx_local*ny_local;

        // get the centered cell positions at (i,j,k)
        // Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        x_pos = x_min_local + i*dx + 0.5*dx;
        y_pos = y_min_local + j*dy + 0.5*dy;
        z_pos = z_min_local + k*dz + 0.5*dz;

        r = sqrt( ( x_pos - center_x )*( x_pos - center_x )/(sigma_x*sigma_x) +
            ( y_pos - center_y )*( y_pos - center_y )/(sigma_y*sigma_y) +
            ( z_pos - center_z )*( z_pos - center_z )/(sigma_z*sigma_z) );
        // dens = exp(-r*r);
        // dens = 0.6;
        // if (r < 0.2 ) dens = 1.0;
        dens = r;
        in_re[id] = dens;
      }
    }
  }

  // string data_name = "/density_1";
  Write_Data_HDF5( file_id, "/density", nx_local, ny_local, nz_local, in_re );


  // First slab transfer
  // Slabs are made in the X-Y plane, Z axis is divided among processes
  int slab_id = procID_z;
  int nProcess_slab = nproc_x * nproc_y;
  MPI_Comm slab_comm;
  MPI_Comm_split( world, slab_id, procID_z, &slab_comm );
  int slab_pId;
  MPI_Comm_rank( slab_comm, &slab_pId );
  for (size_t i = 0; i < nproc; i++) {
    if (procID == i)  printf("Processor %d: slab_id %d  slab_pId %d\n", procID, slab_id, slab_pId );
    usleep(1e5);
  }


  if( procID == 0 ) printf("Sending to first slab\n" );
  int slab_nz = nz_total/nproc;
  double *slab_gather;
  double *slab_local;
  int send_size = slab_nz * nx_local * ny_local;
  slab_gather = (double *) malloc( nProcess_slab * send_size *sizeof(double) );
  slab_local  = (double *) malloc( send_size *sizeof(double) );
  int z_start, z_end;
  int idx_slab, idx;
  for( int np=0; np<nProcess_slab; np++ ){
    // z_start = np*slab_nz;
    // z_end = (np+1)*slab_nz;
    // Get the local data in to slab_local
    for( k=0; k<slab_nz; k++){
      for( j=0; j<ny_local; j++){
        for(i=0; i<nx_local; i++ ){
          idx = i + j*nx_local + (k+np*slab_nz)*nx_local*ny_local;
          idx_slab = i + j*nx_local + k*nx_local*ny_local;
          slab_local[idx_slab] = in_re[idx];
        }
      }
    }
    MPI_Gather( slab_local, send_size, MPI_DOUBLE, slab_gather, send_size, MPI_DOUBLE, np, slab_comm );
  }

  // order the data to in slab
  if( procID == 0 ) printf(" Writing first slab\n" );
  double *slab_data;
  slab_data = (double *) malloc( nProcess_slab * send_size *sizeof(double) );
  int pId_temp, pId_z_temp, pId_y_temp, pId_x_temp;
  int slab_x_start, slab_y_start; // slab_x_end, slab_y_end;
  int idx_gather, idx_data;
  for ( int np=0; np<nProcess_slab; np++ ){
    pId_temp = slab_id*nProcess_slab + np;
    get_mpi_id_3D( pId_temp, nproc_x, nproc_y, pId_x_temp, pId_y_temp, pId_z_temp );
    slab_x_start = pId_x_temp * nx_local;
    // slab_x_end   = (pId_x_temp+1) * nx_local;
    slab_y_start = pId_y_temp * ny_local;
    // slab_y_end   = (pId_y_temp+1) * ny_local;
    for( k=0; k<slab_nz; k++){
      for( j=0; j<ny_local; j++){
        for(i=0; i<nx_local; i++ ){
          idx_gather = np*send_size + i + j*nx_local + k*nx_local*ny_local;
          idx_data   = (slab_x_start + i) + (slab_y_start + j)*nx_total + k*nx_total*ny_total;
          slab_data[idx_data] = slab_gather[idx_gather];
        }
      }
    }
  }
  Write_Data_HDF5( file_id, "/slab", nx_total, ny_total, slab_nz, slab_data  );

  if( procID == 0 ) printf("Sending to second slab\n" );
  double *slab2_gather;
  double *slab2_local;
  int slab2_ny = ny_total/nproc;
  int send_size_2 = slab_nz * nx_total * slab2_ny;
  slab2_gather = (double *) malloc( nproc * send_size_2 *sizeof(double) );
  slab2_local  = (double *) malloc( send_size_2 *sizeof(double) );
  for ( int np=0; np<nproc; np++ ){
    slab_y_start = np*slab2_ny;
    for( k=0; k<slab_nz; k++){
      for( j=0; j<slab2_ny; j++){
        for(i=0; i<nx_total; i++ ){
          idx = i + (j+slab_y_start)*nx_total + k*nx_total*ny_total;
          idx_slab = i + j*nx_total + k*nx_total*slab2_ny;
          slab2_local[idx_slab] = slab_data[idx];
        }
      }
    }
    MPI_Gather( slab2_local, send_size_2, MPI_DOUBLE, slab2_gather, send_size_2, MPI_DOUBLE, np, world );
  }


  // order the data to in slab
  if( procID == 0 ) printf(" Writing second slab\n" );
  double *slab2_data;
  slab2_data = (double *) malloc( nproc * send_size_2 *sizeof(double) );
  int slab_z_start;
  for ( int np=0; np<nproc; np++ ){
    pId_temp = np;
    get_mpi_id_3D( pId_temp, nproc_x, nproc_y, pId_x_temp, pId_y_temp, pId_z_temp );
    slab_z_start = np*slab_nz;
    for( k=0; k<slab_nz; k++){
      for( j=0; j<slab2_ny; j++){
        for(i=0; i<nx_total; i++ ){
          idx_gather = np*send_size_2 + i + j*nx_total + k*nx_total*slab2_ny;
          idx_data   = i + j*nx_total + (k+slab_z_start)*nx_total*slab2_ny;
          slab2_data[idx_data] = slab2_gather[idx_gather];
        }
      }
    }
  }
  Write_Data_HDF5( file_id, "/slab2", nx_total, slab2_ny, nz_total, slab2_data  );

  // Close the file
  status = H5Fclose(file_id);

  MPI_Finalize();
  return 0;
}


void InitializeChollaMPI(int *pargc, char **pargv[])
{
  /*initialize MPI*/
  MPI_Init(pargc, pargv);
  /*set process ids in comm world*/
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  /*find number of processes in comm world*/
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  // nproc_z = sqrt(nproc);
  // nproc_y = sqrt(nproc);
  nproc_z = 2;
  nproc_y = 2;
  nproc_x = 2;

  nProc[0] = nproc_z;
  nProc[1] = nproc_y;
  nProc[2] = nproc_x;
  // procID_x = nproc % nproc_x;
  // procID_y = nproc / nproc_x;
  // procID_z = 0;

  MPI_Get_processor_name(processor_name, &name_len);
  /*print a cute message*/
  printf("Processor %d of %d: Node: %s \n", procID, nproc, processor_name);
  /* set the root process rank */
  root = 0;
  /* set the global communicator */
  world = MPI_COMM_WORLD;

  /*set up node communicator*/
  // node = MPI_Comm_node(&procID_node, &nproc_node);
}

void Write_Header_HDF5(hid_t file_id, int nx_total, int ny_total, int nz_total,
      int nx_local, int ny_local, int nz_local)
{
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;
  int       int_data[3];
  double    double_data[3];

  // Now 3D attributes
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  int_data[0] = nz_total;
  int_data[1] = ny_total;
  int_data[2] = nx_total;

  attribute_id = H5Acreate(file_id, "dims", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);


  int_data[0] = nz_local;
  int_data[1] = ny_local;
  int_data[2] = nx_local;

  attribute_id = H5Acreate(file_id, "dims_local", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  int_data[0] = procCoord[0];
  int_data[1] = procCoord[1];
  int_data[2] = procCoord[2];

  attribute_id = H5Acreate(file_id, "proc_coord", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  int_data[0] = nProc[0];
  int_data[1] = nProc[1];
  int_data[2] = nProc[2];

  attribute_id = H5Acreate(file_id, "proc_grid", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  int_data[0] = procOffset[0];
  int_data[1] = procOffset[1];
  int_data[2] = procOffset[2];

  attribute_id = H5Acreate(file_id, "proc_offset", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_INT, int_data);
  status = H5Aclose(attribute_id);

  double_data[0] = procDomain_l[0];
  double_data[1] = procDomain_l[1];
  double_data[2] = procDomain_l[2];

  attribute_id = H5Acreate(file_id, "proc_domain_l", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, double_data);
  status = H5Aclose(attribute_id);


  // Close the dataspace
  status = H5Sclose(dataspace_id);

}

void Write_Data_HDF5(hid_t file_id, string data_name, int nx, int ny, int nz, double *field_re )
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id;
  double      *dataset_buffer;
  herr_t    status;

  int       nx_dset = nx;
  int       ny_dset = ny;
  int       nz_dset = nz;
  hsize_t   dims[3];
  dataset_buffer = (double *) malloc(nz_dset*ny_dset*nx_dset*sizeof(double));

  // Create the data space for the datasets
  dims[0] = nz_dset;
  dims[1] = ny_dset;
  dims[2] = nx_dset;
  dataspace_id = H5Screate_simple(3, dims, NULL);

  // Copy the density array to the memory buffer
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = i + j*nx + k*nx*ny;
        buf_id = k + j*nz + i*nz*ny;
        // dataset_buffer[buf_id] = field[id];
        dataset_buffer[id] = field_re[id];
      }
    }
  }
  // Create a dataset id for density
  dataset_id = H5Dcreate(file_id, data_name.c_str(), H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // Free the dataset id
  status = H5Dclose(dataset_id);

  free(dataset_buffer);
}



void Write_Grid_HDF5(hid_t file_id, int nx, int ny, int nz, double *field_re )
{
  int i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id;
  double      *dataset_buffer;
  herr_t    status;

  int       nx_dset = nx;
  int       ny_dset = ny;
  int       nz_dset = nz;
  hsize_t   dims[3];
  dataset_buffer = (double *) malloc(nz_dset*ny_dset*nx_dset*sizeof(double));

  // Create the data space for the datasets
  dims[0] = nz_dset;
  dims[1] = ny_dset;
  dims[2] = nx_dset;
  dataspace_id = H5Screate_simple(3, dims, NULL);

  // Copy the density array to the memory buffer
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        id = i + j*nx + k*nx*ny;
        buf_id = k + j*nz + i*nz*ny;
        // dataset_buffer[buf_id] = field[id];
        dataset_buffer[id] = field_re[id];
      }
    }
  }
  // Create a dataset id for density
  dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // Free the dataset id
  status = H5Dclose(dataset_id);

  // // Copy the density array to the memory buffer
  // for (k=0; k<nz; k++) {
  //   for (j=0; j<ny; j++) {
  //     for (i=0; i<nx; i++) {
  //       id = i + j*nx + k*nx*ny;
  //       buf_id = k + j*nz + i*nz*ny;
  //       // dataset_buffer[buf_id] = field[id];
  //       dataset_buffer[id] = field_im[id];
  //     }
  //   }
  // }
  // // Create a dataset id for density
  // dataset_id = H5Dcreate(file_id, "/in_im", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
  // status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // // Free the dataset id
  // status = H5Dclose(dataset_id);
  //
  //
  //
  // // Copy the density array to the memory buffer
  // for (k=0; k<nz; k++) {
  //   for (j=0; j<ny; j++) {
  //     for (i=0; i<nx; i++) {
  //       id = i + j*nx + k*nx*ny;
  //       buf_id = k + j*nz + i*nz*ny;
  //       // dataset_buffer[buf_id] = field[id];
  //       dataset_buffer[id] = transf_re[id];
  //     }
  //   }
  // }
  // // Create a dataset id for density
  // dataset_id = H5Dcreate(file_id, "/fft_re", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
  // status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // // Free the dataset id
  // status = H5Dclose(dataset_id);
  //
  // // Copy the density array to the memory buffer
  // for (k=0; k<nz; k++) {
  //   for (j=0; j<ny; j++) {
  //     for (i=0; i<nx; i++) {
  //       id = i + j*nx + k*nx*ny;
  //       buf_id = k + j*nz + i*nz*ny;
  //       // dataset_buffer[buf_id] = field[id];
  //       dataset_buffer[id] = transf_im[id];
  //     }
  //   }
  // }
  // // Create a dataset id for density
  // dataset_id = H5Dcreate(file_id, "/fft_im", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // // Write the density array to file  // NOTE: NEED TO FIX FOR FLOAT REAL!!!
  // status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // // Free the dataset id
  // status = H5Dclose(dataset_id);

  free(dataset_buffer);
}