EXEC = bigFFT


CC	= mpicc
C++   = mpic++

CFLAGS = -I

HDF5_INCLUDE = -I/usr/include/hdf5/serial/
HDF5_LIBS = -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/

# PFFTINC = /home/bruno/apps/pfft-1.0.8-alpha/include
#PFFTLIB = /home/bruno/apps/pfft-1.0.8-alpha/lib
FFTWINC = /home/bruno/apps/fftw-3.3.5/include
FFTWLIB = /home/bruno/apps/fftw-3.3.5/lib

INCL = -I./ $(HDF5_INCLUDE)
LIBS = -lm $(HDF5_LIBS)

testmake: bigFFT.cpp
	mpic++ -O2 bigFFT.cpp -I$(FFTWINC) $(INCL) -L$(FFTWLIB) -lfftw3_omp -lfftw3 -fopenmp $(LIBS) -o $(EXEC)
