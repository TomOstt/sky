#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){ long n=atol(argv[1]); double*x=calloc(1000000,sizeof(double)); double s=0; for(long i=0;i<n;i++){ s = s + x[i] + 1.0; } printf("%.17g\n",s); return 0;}
