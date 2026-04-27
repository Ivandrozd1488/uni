// benchmark_hpc.cpp — Comprehensive HPC performance validation.
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
// immintrin.h is x86/x64-only — guard for cross-platform builds (e.g., ARM64 macOS)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "core/linalg.hpp"
#include "core/hpc_kernels.hpp"
#include "models/pinn/pinn_fast.hpp"

using Clock=std::chrono::high_resolution_clock;
using Ms=std::chrono::duration<double,std::milli>;
struct Timer{Clock::time_point t0=Clock::now();double ms()const{return Ms(Clock::now()-t0).count();}void reset(){t0=Clock::now();}};
struct BR{std::string name;double mean_ms,min_ms;int iters;};
template<typename F>BR ben(const std::string&nm,int w,int it,F&&fn){for(int i=0;i<w;++i)fn();double mn=1e18;Timer t;for(int i=0;i<it;++i){Timer t2;fn();double m=t2.ms();if(m<mn)mn=m;}return{nm,t.ms()/it,mn,it};}
void pr(const BR&r){std::cout<<std::left<<std::setw(55)<<r.name<<" mean="<<std::fixed<<std::setprecision(4)<<r.mean_ms<<"ms  min="<<r.min_ms<<"ms\n";}

void bench_matmul(){
  std::cout<<"\n=== §1 MATRIX MULTIPLICATION ===\n";
  for(int N:{64,128,256,512,1024}){
    core::Matrix A(N,N),B(N,N);
    for(int i=0;i<N;++i)for(int j=0;j<N;++j){A(i,j)=0.01*(i+1.0)/N;B(i,j)=0.01*(j+1.0)/N;}
    int it=N<=128?100:(N<=512?20:5);
    auto r=ben("matmul "+std::to_string(N)+"x"+std::to_string(N),3,it,[&]{auto C=A*B;(void)C;});
    pr(r);double gf=2.0*N*N*N/(r.min_ms*1e6);
    std::cout<<"  => "<<std::setprecision(2)<<gf<<" GFLOP/s\n";
  }
  {core::Matrix A(512,64,0.1);core::Vector v(64,1.0);
   auto r=ben("matvec 512x64",5,200,[&]{auto y=A*v;(void)y;});pr(r);}
}

void bench_scaling(){
  std::cout<<"\n=== §2 THREAD SCALING (matmul 512x512) ===\n";
  constexpr int N=512;
  core::Matrix A(N,N),B(N,N);
  for(int i=0;i<N;++i)for(int j=0;j<N;++j){A(i,j)=0.01*(i+1.0)/N;B(i,j)=0.01*(j+1.0)/N;}
  int mx=1;
#ifdef _OPENMP
  mx=omp_get_max_threads();
#endif
  double base=0;
  for(int nt=1;nt<=std::min(mx,16);nt*=2){
#ifdef _OPENMP
    omp_set_num_threads(nt);
#endif
    auto r=ben("  threads="+std::to_string(nt),2,10,[&]{auto C=A*B;(void)C;});
    pr(r);double gf=2.0*N*N*N/(r.min_ms*1e6);
    if(nt==1)base=r.min_ms;
    double sp=base/r.min_ms,ef=sp/nt*100;
    std::cout<<"  => "<<std::setprecision(2)<<gf<<" GF/s  speedup="<<sp<<"x  efficiency="<<std::setprecision(1)<<ef<<"%\n";
  }
#ifdef _OPENMP
  omp_set_num_threads(mx);
#endif
}

void bench_fused(){
  std::cout<<"\n=== §3 FUSED GEMV+BIAS+TANH vs SEPARATE ===\n";
  constexpr std::size_t NI=64,NO=32;
  alignas(64) double W[NO*NI],x[NI],bias[NO],out[NO],z[NO],a[NO];
  for(std::size_t i=0;i<NO*NI;++i)W[i]=0.01*((i%100)-50);
  for(std::size_t i=0;i<NI;++i)x[i]=0.5;
  for(std::size_t i=0;i<NO;++i)bias[i]=0.01*i;

  auto r1=ben("SEPARATE: gemv + bias + tanh (64->32)",100,100000,[&]{
    for(std::size_t i=0;i<NO;++i){double acc=0;for(std::size_t j=0;j<NI;++j)acc+=W[i*NI+j]*x[j];z[i]=acc+bias[i];}
    for(std::size_t i=0;i<NO;++i)a[i]=std::tanh(z[i]);volatile double v=a[0];(void)v;});
  pr(r1);
  auto r2=ben("FUSED two-phase GEMV+bias+tanh (64->32)",100,100000,[&]{
    hpc::fused_gemv_bias_act(W,x,bias,out,NO,NI,hpc::FusedAct::TANH);volatile double v=out[0];(void)v;});
  pr(r2);
  std::cout<<"  => HPC fused speedup: "<<std::setprecision(2)<<r1.min_ms/r2.min_ms<<"x\n";
  double maxerr=0;for(std::size_t i=0;i<NO;++i)maxerr=std::max(maxerr,std::abs(out[i]-a[i]));
  std::cout<<"  => Numerical error: "<<std::scientific<<maxerr<<"\n";
}

void bench_prefetch(){
  std::cout<<"\n=== §4 REGISTER-BLOCKED + PREFETCH vs BASELINE TILED ===\n";
  constexpr std::size_t N=512;
  double*A=hpc::alloc64(N*N),*B=hpc::alloc64(N*N),*C1=hpc::alloc64(N*N),*C2=hpc::alloc64(N*N);
  for(std::size_t i=0;i<N*N;++i){A[i]=0.001*(i%1000);B[i]=0.002*(i%500);}

  auto r1=ben("HPC GEMM 512x512 (6x16 reg-block+prefetch)",2,10,[&]{
    std::memset(C1,0,N*N*8);hpc::gemm_hpc(A,B,C1,N,N,N,N,N,N);});
  pr(r1);double gf1=2.0*N*N*N/(r1.min_ms*1e6);

  auto r2=ben("Baseline tiled 512x512 (scalar ikj)",2,10,[&]{
    std::memset(C2,0,N*N*8);
    constexpr std::size_t TM=32,TK=64,TN=64;
    for(std::size_t i0=0;i0<N;i0+=TM)for(std::size_t k0=0;k0<N;k0+=TK)for(std::size_t j0=0;j0<N;j0+=TN){
    std::size_t ie=std::min(i0+TM,N),ke=std::min(k0+TK,N),je=std::min(j0+TN,N);
    for(std::size_t i=i0;i<ie;++i)for(std::size_t k=k0;k<ke;++k){
      double aik=A[i*N+k];if(aik==0.0)continue;
      for(std::size_t j=j0;j<je;++j)C2[i*N+j]+=aik*B[k*N+j];}}});
  pr(r2);double gf2=2.0*N*N*N/(r2.min_ms*1e6);

  double maxerr=0;for(std::size_t i=0;i<N*N;++i)maxerr=std::max(maxerr,std::abs(C1[i]-C2[i]));
  std::cout<<"  => HPC:  "<<std::setprecision(2)<<gf1<<" GF/s\n";
  std::cout<<"  => Baseline:  "<<gf2<<" GF/s\n";
  std::cout<<"  => Speedup: "<<r2.min_ms/r1.min_ms<<"x\n";
  std::cout<<"  => Max error:  "<<std::scientific<<maxerr<<"\n";
  hpc::free64(A);hpc::free64(B);hpc::free64(C1);hpc::free64(C2);
}

void bench_nesting(){
  std::cout<<"\n=== §5 NESTED PARALLELISM PROOF ===\n";
  // Prove that inner kernels are serial by timing inside vs outside parallel
  constexpr std::size_t NI=64,NO=32;
  alignas(64) double W[NO*NI],x[NI],bias[NO],out[NO];
  for(std::size_t i=0;i<NO*NI;++i)W[i]=0.01*i;
  for(std::size_t i=0;i<NI;++i)x[i]=0.5;
  for(std::size_t i=0;i<NO;++i)bias[i]=0.01;

  auto r1=ben("fused_gemv_bias_act OUTSIDE parallel",100,100000,[&]{
    hpc::fused_gemv_bias_act(W,x,bias,out,NO,NI,hpc::FusedAct::TANH);volatile double v=out[0];(void)v;});
  pr(r1);

  double inside_min=1e18;
  for(int rep=0;rep<10;++rep){
    Timer t;
    #pragma omp parallel
    {
    alignas(64) double lout[32];
    #pragma omp for schedule(static)
    for(int b=0;b<1000;++b)
      hpc::fused_gemv_bias_act(W,x,bias,lout,NO,NI,hpc::FusedAct::TANH);
    }
    double ms=t.ms()/1000;
    if(ms<inside_min)inside_min=ms;
  }
  std::cout<<std::left<<std::setw(55)<<"fused_gemv_bias_act INSIDE parallel (per-call)"
     <<" min="<<std::fixed<<std::setprecision(4)<<inside_min<<"ms\n";
  std::cout<<"  => Inside/outside ratio: "<<std::setprecision(2)<<inside_min/r1.min_ms
     <<"x (should be ~1.0 = no nesting overhead)\n";
}

void bench_pinn(){
  std::cout<<"\n=== §6 PINN ENGINE ===\n";
  constexpr double PI=3.14159265358979;
  constexpr int NC=100,NB=2;
  std::vector<double> cx(NC),fv(NC);
  for(int i=0;i<NC;++i){cx[i]=(i+1.0)/(NC+1);fv[i]=PI*PI*std::sin(PI*cx[i]);}
  std::vector<double> bx={0.0,1.0},bu={0.0,0.0};

  {pinn::PinnFastNet net({1,64,32,1},42,1e-3);
   auto r=ben("PINN train_step (100 coll, 2 BC)",3,50,[&]{net.train_step_poisson1d(cx,fv,bx,bu);});pr(r);}
  {pinn::PinnFastNet net({1,64,32,1},42,1e-3);
   std::vector<double> tx(1000);for(int i=0;i<1000;++i)tx[i]=i/999.0;
   auto r=ben("PINN predict 1000pts (parallel batch)",3,50,[&]{auto p=net.predict(tx);(void)p;});pr(r);}
  {pinn::PinnFastNet net({1,64,32,1},42,1e-3);
   auto r=ben("PINN forward_scalar (single point)",10,50000,[&]{double x=0.5;volatile double u=net.forward_scalar(&x);(void)u;});
   pr(r);std::cout<<"  => "<<std::setprecision(3)<<r.min_ms*1000.0<<" us/call\n";}
  {pinn::PinnFastNet net({1,64,32,1},42,1e-3);Timer t;
   for(int i=0;i<1000;++i)net.train_step_poisson1d(cx,fv,bx,bu);double ms=t.ms();
   std::cout<<std::left<<std::setw(55)<<"PINN 1000 iters end-to-end"
      <<" total="<<std::setprecision(1)<<ms<<"ms  per_iter="<<std::setprecision(4)<<ms/1000<<"ms\n";
   double me=0;for(int i=1;i<20;++i){double x=i/20.0;double p=net.forward_scalar(&x);double e=std::sin(PI*x);me=std::max(me,std::abs(p-e));}
   std::cout<<"  => L-inf error: "<<std::scientific<<me<<"\n";}
}

#if HPC_HAS_AVX512
// microkernel_6x16_packed is only defined when AVX-512 is available
void bench_microkernel(){
  std::cout<<"\n=== §7 6x16 MICRO-KERNEL ===\n";
  constexpr std::size_t K=256;
  alignas(64) double Ap[K*HPC_MR],Bp[K*HPC_NR],C[HPC_MR*HPC_NR];
  for(std::size_t i=0;i<K*HPC_MR;++i)Ap[i]=0.01;
  for(std::size_t i=0;i<K*HPC_NR;++i)Bp[i]=0.02;
  auto r=ben("microkernel_6x16_packed (K=256)",100,100000,[&]{
    std::memset(C,0,sizeof(C));hpc::microkernel_6x16_packed(Ap,Bp,C,K,HPC_NR);volatile double v=C[0];(void)v;});
  pr(r);double gf=6.0*16*K*2.0/(r.min_ms*1e6);
  std::cout<<"  => "<<std::setprecision(2)<<gf<<" GF/s (12 ZMM, 0 spills)\n";
}
#endif // HPC_HAS_AVX512

int main(){
  std::cout<<"================================================================\n";
  std::cout<<" unified_ml HPC Performance Benchmark (FINAL)\n";
  std::cout<<"================================================================\n";
  std::cout<<"Build: -O3 -march=native -ffast-math -fopenmp (AVX-512)\n";
#ifdef _OPENMP
  std::cout<<"OpenMP threads: "<<omp_get_max_threads()<<"\n";
#endif
  std::cout<<"Register blocking: 6x16 (12 ZMM accumulators)\n";
  std::cout<<"Macro-tiling: MC="<<HPC_MC<<" NC="<<HPC_NC<<" KC="<<HPC_KC<<"\n";
  std::cout<<"Leaf kernel principle: NO nested parallelism\n";
  std::cout<<"Forward strategy: two-phase (GEMV+bias -> vectorized tanh)\n";

  bench_matmul();bench_scaling();bench_fused();bench_prefetch();
  bench_nesting();bench_pinn();
#if HPC_HAS_AVX512
  bench_microkernel();
#else
  std::cout<<"\n=== §7 6x16 MICRO-KERNEL ===\n";
  std::cout<<"  Skipped — AVX-512 not available on this platform/CPU\n";
#endif

  std::cout<<"\n================================================================\n";
  std::cout<<" ALL BENCHMARKS COMPLETE\n";
  std::cout<<"================================================================\n";
  return 0;
}
