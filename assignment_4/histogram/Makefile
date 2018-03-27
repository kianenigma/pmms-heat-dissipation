NVCC		= nvcc
CC		= g++
#CU_FLAGS	= -O3 -g -deviceemu
CU_FLAGS	= -O3 -g -arch compute_20 -code sm_20 
#--ptxas-options=-v -Wno-deprecated-gpu-targets
#CU_FLAGS=-O2 -arch=sm_20
CC_FLAGS	= -O3 -m64 -Wall

CU_SOURCES	= histogram.cu
CU_EV_SOURCES	= histogram-events.cu
CC_SOURCES	= timer.cc

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_EV_OBJECTS	= $(CU_EV_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CU_EV_PTX	= $(CU_EV_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)


%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o:		%.cc
		$(CC) $(CC_FLAGS) -c $< -o $@

%.ptx:		%.cu
		$(NVCC) $(CU_FLAGS) --ptx $< -o $@

myhistogram:	$(CU_OBJECTS) $(CC_OBJECTS)
		$(NVCC) $^ -o $@

ptx:            $(CU_PTX)

myhistogram-events:	$(CU_EV_OBJECTS) $(CC_OBJECTS) 
			$(NVCC) $^ -o $@ 

clean:
		rm -f *.o *.ptx myhistogram myhistogram-events histo_gpu
