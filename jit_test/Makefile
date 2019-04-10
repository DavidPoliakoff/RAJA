default: cpu

CALI_DIR ?= $(HOME)
CLANG_JIT ?= clang++
RAJA_DIR ?= $(HOME)
CUDA_FLAGS ?= -L/usr/tce/packages/cuda/cuda-9.2.148/lib64 -x cuda --cuda-gpu-arch=sm_70 -lcudart
CALIPER_FLAGS = -I$(CALI_DIR)/include -L$(CALI_DIR)/lib64 -lcaliper -lcaliper-reader -lcaliper-common -DUSE_CALIPER -B/usr/workspace/wsrzd/dzpolia/src/RADIOACTIVE_GPL/build2/gold

all: cpu 
cpu: simple_test simple_test_nojit

simple_test: simple_test.cpp
	$(CLANG_JIT) -L$(RAJA_DIR)/lib -lRAJA -DGTEST_HAS_DEATH_TEST=0 -I$(RAJA_DIR)/include -isystem /usr/workspace/wsrzd/dzpolia/src/raja/blt/thirdparty_builtin/googletest-master-2018-02-01/googletest/include -I/usr/workspace/wsrzd/dzpolia/src/raja/include -I/usr/workspace/wsrzd/dzpolia/src/raja/build/include -lRAJA -lstdc++ -fjit -O3  -Wall -Wextra $(CUDA_FLAGS) -fPIE -DUSE_JIT -std=c++14 simple_test.cpp -lstdc++ $(CALIPER_FLAGS) -lRAJA -o simple_test

simple_test_nojit: simple_test.cpp
	$(CLANG_JIT) -L$(RAJA_DIR)/lib -lRAJA -DGTEST_HAS_DEATH_TEST=0 -I$(RAJA_DIR)/include -lRAJA -lstdc++ -fjit -O3  -Wall -Wextra  -fPIE $(CUDA_FLAGS) -DNO_JIT -std=c++14 simple_test.cpp $(CALIPER_FLAGS) -lstdc++ -lRAJA -o simple_test_nojit
