NVCC		= nvcc
CC		= g++
#CU_FLAGS	= -O3 -g -deviceemu
CU_FLAGS	= -O3 -g --ptxas-options=-v -arch compute_20 -code sm_20
CC_FLAGS	= -O3 -m64 -Wall

CU_SOURCES	= convolution.cu
CC_SOURCES	= timer.cc

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CU_EV_PTX	= $(CU_EV_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o:		%.cc
		$(CC) $(CC_FLAGS) -c $< -o $@

%.ptx:		%.cu
		$(NVCC) $(CU_FLAGS) --ptx $< -o $@

myconvolution:	$(CU_OBJECTS) $(CC_OBJECTS)
		$(NVCC) $^ -o $@

clean:
		rm -f *.o myconvolution
