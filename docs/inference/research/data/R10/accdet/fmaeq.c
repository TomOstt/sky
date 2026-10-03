// Is Accelerate's dgemm bitwise equal to a sequential k-order FMA chain (c = fma(a,b,c), c0=+0)?
// Or to the unfused chain (c = c + a*b), which is Sky's reproducible definition?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
static unsigned long long s=88172645463325252ULL; static double rnd(void){ s^=s<<13; s^=s>>7; s^=s<<17; return (double)(s>>11)/9007199254740992.0-0.5; }
int main(int argc,char**argv){
  int n = argc>1? atoi(argv[1]) : 256;
  double *A=malloc(sizeof(double)*n*n),*B=malloc(sizeof(double)*n*n),*C=malloc(sizeof(double)*n*n);
  for(long i=0;i<(long)n*n;i++){A[i]=rnd();B[i]=rnd();}
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,1.0,A,n,B,n,0.0,C,n);
  long eq_fma=0, eq_plain=0, tot=(long)n*n; double maxrel=0;
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){
    double f=0.0, p=0.0;
    for(int k=0;k<n;k++){ f=fma(A[(long)i*n+k],B[(long)k*n+j],f); volatile double prod=A[(long)i*n+k]*B[(long)k*n+j]; p=p+prod; }
    double c=C[(long)i*n+j];
    eq_fma += (c==f); eq_plain += (c==p);
    double r=fabs(c-f)/(fabs(f)+1e-300); if(r>maxrel) maxrel=r;
  }
  printf("n=%d entries=%ld  equal_to_seq_fma=%ld  equal_to_seq_unfused=%ld  max_rel_diff_vs_fma=%.3g\n", n, tot, eq_fma, eq_plain, maxrel);
  return 0;
}
