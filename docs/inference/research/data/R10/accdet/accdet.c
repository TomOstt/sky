// Bitwise determinism of Accelerate dgemm / dpotrf / dgemv across runs and thread counts.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <Accelerate/Accelerate.h>
static uint64_t fnv(const double *x, long n){ uint64_t h=1469598103934665603ULL; const unsigned char*p=(const unsigned char*)x; for(long i=0;i<n*8;i++){h^=p[i]; h*=1099511628211ULL;} return h; }
static uint64_t s=88172645463325252ULL; static double rnd(void){ s^=s<<13; s^=s>>7; s^=s<<17; return (double)(s>>11)/9007199254740992.0-0.5; }
int main(int argc,char**argv){
  int n = argc>1? atoi(argv[1]) : 512;
  double *A=malloc(sizeof(double)*n*n),*B=malloc(sizeof(double)*n*n),*C=malloc(sizeof(double)*n*n),*K=malloc(sizeof(double)*n*n),*x=malloc(sizeof(double)*n),*y=malloc(sizeof(double)*n);
  for(long i=0;i<(long)n*n;i++){A[i]=rnd();B[i]=rnd();}
  for(int i=0;i<n;i++) x[i]=rnd();
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,1.0,A,n,B,n,0.0,C,n);
  printf("dgemm %d %016llx\n", n, (unsigned long long)fnv(C,(long)n*n));
  cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,1.0,A,n,x,1,0.0,y,1);
  printf("dgemv %d %016llx\n", n, (unsigned long long)fnv(y,n));
  // SPD K = A A^T + n I
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,n,n,1.0,A,n,A,n,0.0,K,n);
  for(int i=0;i<n;i++) K[(long)i*n+i]+=n;
  __LAPACK_int nn=n, info=0; char lo='L';
  dpotrf_(&lo,&nn,K,&nn,&info);
  printf("dpotrf %d %016llx info=%d\n", n, (unsigned long long)fnv(K,(long)n*n), (int)info);
  return 0;
}
