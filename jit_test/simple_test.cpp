#include<RAJA/RAJA.hpp>
#include <chrono>
#include <tuple>
using clock_type = std::chrono::high_resolution_clock;
#ifdef USE_CALIPER
#include <caliper/Annotation.h>
#include <caliper/cali.h>
cali::Annotation func_annot("function",CALI_ATTR_NESTED);
#endif

#ifdef USE_JIT
#define JIT_ENABLED true
#else
#define JIT_ENABLED false
#endif

void enable_caliper(const char* fn){
#ifdef USE_CALIPER
  cali_config_preset("CALI_SERVICES_ENABLE","event:recorder:aggregate:timestamp");
  cali_config_preset("CALI_RECORDER_FILENAME",fn);
#endif
}

void begin(const char* name){
#ifdef USE_CALIPER
  func_annot.begin(name);
#endif
}

void end(){
#ifdef USE_CALIPER
  func_annot.end();
#endif
}

template<typename T>
void set_global(const char* key, T value){
#ifdef USE_CALIPER
  cali::Annotation(key,CALI_ATTR_GLOBAL | CALI_ATTR_SKIP_EVENTS).set(value);
#endif
}
#define MAT2D(r,c,size) r*size+c

using namespace RAJA::statement;
using RAJA::seq_exec;

using MatMulPolicy = RAJA::KernelPolicy<
      For<
        0,seq_exec,
        For<
          1,seq_exec,
          For<
            2, seq_exec,
              For<3, seq_exec,Lambda<0>>
          >
        >
      >
    >;

template<typename Policy, long... ends, typename Kernel>
[[clang::jit]] void affine_jit_kernel_full(Kernel&& kernel){
  static auto rs =
    camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
    );
  RAJA::kernel<Policy>(
      rs,
      std::forward<Kernel>(kernel)
  );
}

template<typename Policy, typename Kernel, typename... Args>
void affine_jit_kernel_difficult_helper2(Kernel&& kernel, Args... args){
   affine_jit_kernel_full<Policy,*std::end(args)...>(
       std::forward<Kernel>(kernel)
   );
}

template<typename Policy, typename TupleLike, typename Kernel, std::size_t... indices>
void affine_jit_kernel_difficult_helper(TupleLike IndexTuple, Kernel&& kernel, std::index_sequence<indices...>){
  affine_jit_kernel_difficult_helper2<Policy>(
   std::forward<Kernel>(kernel),
   camp::get<indices>(IndexTuple)...
  );
    
}

template<typename Policy, typename TupleLike, typename Kernel>
void affine_jit_kernel_difficult(TupleLike IndexTuple, Kernel&& kernel){
  affine_jit_kernel_difficult_helper<Policy,TupleLike>(
      std::forward<TupleLike>(IndexTuple),
      std::forward<Kernel>(kernel),
      std::make_index_sequence<camp::tuple_size<TupleLike>::value>()
  );
}

template<std::size_t... ends, typename Kernel>
[[clang::jit]] void affine_jit_kernel(Kernel&& kernel) {
  static auto rs =
    camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
    );
  RAJA::kernel<MatMulPolicy>(
      rs,
      std::forward<Kernel>(kernel)
  );
}

template<std::size_t n_matrices, std::size_t size>
[[clang::jit]] void affine_jit_kernel_simplified(float* out_matrix, float* input_matrix1, float* input_matrix2) {
  static auto rs = 
      camp::make_tuple(
        RAJA::RangeSegment(0,n_matrices),
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size)
      );
  RAJA::kernel<MatMulPolicy>(
      rs,
      [=](const std::size_t matrices, const std::size_t i,const std::size_t j,const std::size_t k){
        (void)matrices;
        out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
      }  
  );
}

template<typename Policy, typename... Args, typename Kernel>
void affine_kernel(Kernel&& k, Args... ends){
  RAJA::kernel<Policy>(
      camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
      ),
      k
  );
}

#ifndef USE_JIT 
#ifndef NO_JIT
#define USE_JIT
#endif // NO_JIT
#endif // USE_JIT

int main(int argc, char* argv[]){
  srand(time(nullptr));
  auto start_click = clock_type::now();
  int run_id = rand();
  (void)argc;
  std::size_t size = atoi(argv[1]);
  std::size_t batch_size = atoi(argv[2]);
  std::cout << "Size is "<<size<<", batch size "<< batch_size<<"\n";
  std::size_t repeats = 5000000000;
  if(argc>2){
     repeats = atoi(argv[3]) * 10000000;
  }
  char file_name[1024];
  sprintf(file_name, "%lu_%lu_%lu%s.cali",size,batch_size,repeats, JIT_ENABLED ? "_jit" : "");
  enable_caliper(file_name);
  begin("main");
  set_global("total_iterations", (double)repeats);
  set_global("size",(int)size);
  set_global("batch_size",(int)batch_size);
  set_global("run_id",run_id);
  set_global("has_jit",JIT_ENABLED);
  float* out_matrix = (float*)malloc(sizeof(float)*size*size );
  float* input_matrix1 = (float*)malloc(sizeof(float)*size*size);
  float* input_matrix2 = (float*)malloc(sizeof(float)*size*size);
  for(std::size_t r = 0; r < size * size; r++){
    input_matrix1[r] = (1.0f*rand())/RAND_MAX;
    input_matrix2[r] = (1.0f*rand())/RAND_MAX;
  } 
  for(std::size_t rep = 0; rep<(repeats/batch_size);rep++){
#ifdef USE_JIT
    affine_jit_kernel_difficult<MatMulPolicy>(
        camp::make_tuple(
          RAJA::RangeSegment(0,batch_size),
          RAJA::RangeSegment(0,size),
          RAJA::RangeSegment(0,size),
          RAJA::RangeSegment(0,size)
        ),
        [=](const std::size_t matrices, const std::size_t i, const std::size_t j, const std::size_t k){
            (void)matrices;
            out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
        }
    );
#endif 
  }
  auto mid_click = clock_type::now();
  for(std::size_t rep = 0; rep<(repeats/batch_size);rep++){
    affine_kernel<RAJA::KernelPolicy<
      For<
        0,seq_exec,
      For<
        1,seq_exec,
        For<
          2, seq_exec,
            For<3, seq_exec,Lambda<0>>
        >
      >
    >>>(
        [=](const std::size_t matrices, const std::size_t i, const std::size_t j, const std::size_t k){
          (void)matrices;
          out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
        },
        batch_size,
        size,
        size,
        size
      );
  } // end for loop
  auto end_click = clock_type::now();
  auto total_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_click - start_click).count();
  auto nonjit_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_click - mid_click).count();
  auto jit_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(mid_click - start_click).count();
  set_global("total_time",total_time);
  set_global("jit_time",jit_time); 
  set_global("traditional_time",nonjit_time);
  set_global("speedup",nonjit_time/jit_time);
  end();
}
