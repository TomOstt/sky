// Rectangular, transposed and beta!=0 variants: is cblas_dgemm == sequential k-order FMA chain?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
static unsigned long long s=88172645463325252ULL; static double rnd(void){ s^=s<<13; s^=s>>7; s^=s<<17; return (double)(s>>11)/9007199254740992.0-0.5; }
int main(int argc,char**argv){
  int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]); int transB = argc>4 ? atoi(argv[4]) : 0;
  double *A=malloc(sizeof(double)*m*k),*B=malloc(sizeof(double)*k*n),*C=malloc(sizeof(double)*m*n);
  for(long i=0;i<(long)m*k;i++) A[i]=rnd();
  for(long i=0;i<(long)k*n;i++) B[i]=rnd();
  // transB: B stored as n x k (row-major), use B^T
  cblas_dgemm(CblasRowMajor,CblasNoTrans,transB?CblasTrans:CblasNoTrans,m,n,k,1.0,A,k,B,transB?k:n,0.0,C,n);
  long eq=0, tot=(long)m*n;
  for(int i=0;i<m;i++) for(int j=0;j<n;j++){
    double f=0.0;
    for(int p=0;p<k;p++){ double b = transB ? B[(long)j*k+p] : B[(long)p*n+j]; f=fma(A[(long)i*k+p],b,f); }
    eq += (C[(long)i*n+j]==f);
  }
  printf("m=%d n=%d k=%d transB=%d  equal_to_seq_fma=%ld/%ld\n", m,n,k,transB, eq, tot);
  return 0;
}
