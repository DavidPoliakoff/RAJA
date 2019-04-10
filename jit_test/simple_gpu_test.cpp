#include <chrono>
#ifdef USE_CALIPER
#include <caliper/Annotation.h>
#include <caliper/cali.h>
cali::Annotation func_annot("function",CALI_ATTR_NESTED);
#endif
using clock_type = std::chrono::high_resolution_clock;

#ifdef USE_JIT
#define JIT_ENABLED true
#else
#define JIT_ENABLED false
#endif
#include <RAJA/RAJA.hpp>

template<typename Body>
__global__ void forall_kernel(Body&& in, int upbound){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id<upbound){
    in(id);
  }
}

//template<typename Exec, typename Body>
//void forall(RAJA::RangeSegment is, Body bd){
//  auto start = *std::begin(is);
//  auto end = *std::end(is);
//  forall_kernel<<<(end+255)/256,256>>>(std::move(bd),end);
//}
template<typename Exec, typename Body>
void forall(RAJA::RangeSegment is, Body bd){
  auto start = *std::begin(is);
  auto end = *std::end(is);
  for(auto idx : is){
      bd(idx);
  }
}
void enable_caliper(const char* fn){
#ifdef USE_CALIPER
  cali_config_preset("CALI_SERVICES_ENABLE","event:recorder:aggregate:timestamp:report");
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
#define DEVICE
#define CUDA_BLOCK_SIZE 256
#define cu_device __device__
#define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#define cu_reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>
#define pforall(i,max,body) \
RAJA::forall<cu_exec>(0,max,[=]cu_device(RAJA::Index_type i) {body});
#define   ijN(i,j,N) (i)+(N)*(j)
#define  ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define      ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define      ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define     _ijkNM(i,j,k,N,M) (j)+(N)*((k)+(M)*(i))
#define     ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define    _ijklNM(i,j,k,l,N,M)  (j)+(N)*((k)+(N)*((l)+(M)*(i)))
#define    ijklmNM(i,j,k,l,m,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*(m))))
#define   _ijklmNM(i,j,k,l,m,N,M) (j)+(N)*((k)+(N)*((l)+(N)*((m)+(M)*(i))))
#define ijklmnNM(i,j,k,l,m,n,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*((m)+(M)*(n)))))
template<int NUM_DOFS_1D,
         int NUM_QUAD_1D>
[[clang::jit]] void rMassMultAdd3DJitEmpty(
   const int numElements,
   const double* dofToQuad,
   const double* dofToQuadD,
   const double* quadToDof,
   const double* quadToDofD,
   const double* oper,
   const double* solIn,
   double* __restrict solOut)
{
   //forall(e,numElements,
  forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0,numElements),[=]DEVICE(RAJA::Index_type e)
     {

     });
   //);
}
template<int NUM_DOFS_1D,
         int NUM_QUAD_1D>
[[clang::jit]] void rMassMultAdd3DJit(
   const int numElements,
   const double* dofToQuad,
   const double* dofToQuadD,
   const double* quadToDof,
   const double* quadToDofD,
   const double* oper,
   const double* solIn,
   double* __restrict solOut)
{
   //forall(e,numElements,
  forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0,numElements),[=]DEVICE(RAJA::Index_type e)
     {
        double sol_xyz[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] = 0;
              }
           }
        }
#pragma unroll
        for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
        {
           double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xy[qy][qx] = 0;
              }
           }
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              double sol_x[NUM_QUAD_1D];
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_x[qx] = 0;
              }
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
                 }
              }
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xy[qy][qx] += wy * sol_x[qx];
                 }
              }
           }
           for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
           {
              const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
                 }
              }
           }
        }
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
              }
           }
        }
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           double sol_xy[NUM_DOFS_1D][NUM_DOFS_1D];
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_xy[dy][dx] = 0;
              }
           }
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              double sol_x[NUM_DOFS_1D];
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_x[dx] = 0;
              }
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 const double s = sol_xyz[qz][qy][qx];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
                 }
              }
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_xy[dy][dx] += wy * sol_x[dx];
                 }
              }
           }
           for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
           {
              const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
                 }
              }
           }
        }
     });
   //);
}

#define NUM_QUAD_1D_SIZE 16
#define NUM_DOFS_1D_SIZE 16

void rMassMultAdd3D(
   const int NUM_DOFS_1D,
   const int NUM_QUAD_1D,
   const int numElements,
   const double* dofToQuad,
   const double* dofToQuadD,
   const double* quadToDof,
   const double* quadToDofD,
   const double* oper,
   const double* solIn,
   double* __restrict solOut)
{
   //forall(e,numElements,
    forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0,numElements),[=]DEVICE(RAJA::Index_type e)
     {
        double sol_xyz[NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE];
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] = 0;
              }
           }
        }
#pragma unroll
        for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
        {
           double sol_xy[NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE];
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xy[qy][qx] = 0;
              }
           }
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              double sol_x[NUM_QUAD_1D_SIZE];
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_x[qx] = 0;
              }
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
                 }
              }
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xy[qy][qx] += wy * sol_x[qx];
                 }
              }
           }
           for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
           {
              const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
                 }
              }
           }
        }
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
              }
           }
        }
#pragma unroll
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           double sol_xy[NUM_DOFS_1D_SIZE][NUM_DOFS_1D_SIZE];
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_xy[dy][dx] = 0;
              }
           }
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              double sol_x[NUM_DOFS_1D_SIZE];
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_x[dx] = 0;
              }
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 const double s = sol_xyz[qz][qy][qx];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
                 }
              }
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_xy[dy][dx] += wy * sol_x[dx];
                 }
              }
           }
           for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
           {
              const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
                 }
              }
           }
        }
     }
   );
}

void rMassMultAdd3DEmpty(
   const int NUM_DOFS_1D,
   const int NUM_QUAD_1D,
   const int numElements,
   const double* dofToQuad,
   const double* dofToQuadD,
   const double* quadToDof,
   const double* quadToDofD,
   const double* oper,
   const double* solIn,
   double* __restrict solOut)
{
   //forall(e,numElements,
    forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0,numElements),[=]DEVICE(RAJA::Index_type e)
     {
        double sol_xyz[NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE];
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] = 0;
              }
           }
        }
        for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
        {
           double sol_xy[NUM_QUAD_1D_SIZE][NUM_QUAD_1D_SIZE];
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xy[qy][qx] = 0;
              }
           }
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              double sol_x[NUM_QUAD_1D_SIZE];
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_x[qx] = 0;
              }
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
                 }
              }
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xy[qy][qx] += wy * sol_x[qx];
                 }
              }
           }
           for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
           {
              const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
              {
                 for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                 {
                    sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
                 }
              }
           }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
              }
           }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
        {
           double sol_xy[NUM_DOFS_1D_SIZE][NUM_DOFS_1D_SIZE];
           for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
           {
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_xy[dy][dx] = 0;
              }
           }
           for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
           {
              double sol_x[NUM_DOFS_1D_SIZE];
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
              {
                 sol_x[dx] = 0;
              }
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
              {
                 const double s = sol_xyz[qz][qy][qx];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
                 }
              }
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    sol_xy[dy][dx] += wy * sol_x[dx];
                 }
              }
           }
           for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
           {
              const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
              {
                 for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                 {
                    solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
                 }
              }
           }
        }
     }
   );
}

int oper_size(int num_dofs, int num_quads, int num_elements){
  return num_elements * num_quads ;
}
int q2d_size(int num_dofs, int num_quads, int num_elements){
  return num_dofs * num_dofs * num_dofs * num_quads * num_quads * num_quads;
}
int d2q_size(int num_dofs, int num_quads, int num_elements){
  return num_dofs * num_dofs * num_dofs * num_quads * num_quads * num_quads;
}
int sol_size(int num_dofs, int num_quads, int num_elements){
  return num_elements * num_dofs * num_dofs * num_dofs * num_quads * num_quads * num_quads;
}
double* cmal(std::size_t size){
  void* in;
  cudaMalloc(&in, size);
  return (double*)in;
}
double* hmal(std::size_t size){
  double* in = (double*)malloc(size);
  for(std::size_t iter = 0; iter < (size/sizeof(double)) ; ++iter){
    in[iter] = (float)rand() / RAND_MAX;
  }
  return in;
}
void htd(void* in, void* out, int size){
  cudaMemcpy(in,out,sizeof(double)*size,cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]){
  //void rMassMultAdd3D(
  //const int NUM_DOFS_1D,
  //const int NUM_QUAD_1D,
  //const int numElements,
  //const double* dofToQuad,
  //const double* dofToQuadD,
  //const double* quadToDof,
  //const double* quadToDofD,
  //const double* oper,
  //const double* solIn,
  //double* __restrict solOut
  int repeats  = atoi(argv[1]);
  int num_elements = atoi(argv[2]);
  int num_dofs = atoi(argv[3]);
  int num_quads = atoi(argv[4]);
  char file_name[1024];
  sprintf(file_name, "%s_%d_%d_%d_%d.cali","fem",repeats,num_elements,num_dofs,num_quads);
  enable_caliper(file_name);
  begin("main");
  set_global("iterations",(double)repeats);
  set_global("elements",(double)num_elements);
  set_global("dofs",(double)num_dofs);
  set_global("quads",(double)num_quads);
  auto setup_click = clock_type::now();
  double *h_oper, *d_oper;
  double *h_dofToQuad, *d_dofToQuad;
  double *h_quadToDof, *d_quadToDof;
  double *h_solIn, *d_solIn;
  double *h_solOut, *d_solOut;
  std::cout << "Pre setup: repeats( "<<repeats<<" ), dofs ( "<<num_dofs<<" ), quads ( "<<num_quads<<" ), elements ( "<<num_elements<<" )"<<"\n";
  h_oper = (double*)hmal(sizeof(double)*oper_size(num_dofs,num_quads,num_elements));
  h_dofToQuad = (double*)hmal(sizeof(double)*d2q_size(num_dofs,num_quads,num_elements));
  h_quadToDof = (double*)hmal(sizeof(double)*q2d_size(num_dofs,num_quads,num_elements));
  h_solIn = (double*)hmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));
  h_solOut = (double*)hmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));

  d_oper = (double*)hmal(sizeof(double)*oper_size(num_dofs,num_quads,num_elements));
  d_dofToQuad = (double*)hmal(sizeof(double)*d2q_size(num_dofs,num_quads,num_elements));
  d_quadToDof = (double*)hmal(sizeof(double)*q2d_size(num_dofs,num_quads,num_elements));
  d_solIn = (double*)hmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));
  d_solOut = (double*)hmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));
 
  //d_oper = (double*)cmal(sizeof(double)*oper_size(num_dofs,num_quads,num_elements));
  //d_dofToQuad = (double*)cmal(sizeof(double)*d2q_size(num_dofs,num_quads,num_elements));
  //d_quadToDof = (double*)cmal(sizeof(double)*q2d_size(num_dofs,num_quads,num_elements));
  //d_solIn = (double*)cmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));
  //d_solOut = (double*)cmal(sizeof(double)*sol_size(num_dofs,num_quads,num_elements));

  //htd(h_oper,d_oper,oper_size(num_dofs,num_quads,num_elements));
  //htd(h_dofToQuad,d_dofToQuad,d2q_size(num_dofs,num_quads,num_elements));
  //htd(h_quadToDof,d_quadToDof,q2d_size(num_dofs,num_quads,num_elements));
  //htd(h_solIn,d_solIn,sol_size(num_dofs,num_quads,num_elements));
  std::cout << "Pre Jit\n";
  rMassMultAdd3DJit<num_dofs,num_quads>(num_elements, d_dofToQuad,nullptr,d_quadToDof,nullptr,d_oper,d_solIn,d_solOut);
  std::cout << "Pre Jit Cycle\n";
  auto start_click = clock_type::now();
  for(int iter = 0;iter < repeats;iter++){
    rMassMultAdd3DJit<num_dofs,num_quads>(num_elements, d_dofToQuad,nullptr,d_quadToDof,nullptr,d_oper,d_solIn,d_solOut);
  }
  auto mid_click = clock_type::now();
  std::cout << "Pre NonJit Cycle\n";
  for(int iter = 0;iter < repeats;iter++){
    rMassMultAdd3D(num_dofs,num_quads,num_elements, d_dofToQuad,nullptr,d_quadToDof,nullptr,d_oper,d_solIn,d_solOut);
  }
  std::cout << "Post Compute\n";
  auto end_click = clock_type::now();
  auto setup_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(start_click - setup_click).count();
  auto total_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_click - start_click).count();
  auto nonjit_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_click - mid_click).count();
  auto jit_time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(mid_click - start_click).count();
  set_global("setup_time",setup_time / 1000000.0);
  set_global("total_time",total_time / 1000000.0);
  set_global("jit_time",jit_time / 1000000.0); 
  set_global("traditional_time",nonjit_time / 1000000.0);
  set_global("speedup",nonjit_time/jit_time);
  end();
  std::cout << "Done\n";
}
