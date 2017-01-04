// VSNR ADMM 2D
//
// COMPILE WITH
// mex '-L/usr/local/lib' -lfftw3_omp -lfftw3 -lm VSNR_ADMM_2D.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

#include <math.h>
#include "mex.h"
#include "fftw3.h"
#include <omp.h>
//#include <stdlib.h>
//#include <time.h>

// Computes out=u1.*u2
void product_carray(fftw_complex *u1,fftw_complex *u2,fftw_complex *out,int n){
    #pragma omp parallel for 
    for (int i=0;i<n;++i){
        out[i][0]=u1[i][0]*u2[i][0]-u1[i][1]*u2[i][1];
        out[i][1]=u1[i][1]*u2[i][0]+u1[i][0]*u2[i][1];
    }
}

// Displays a real array
void disp_array(double *u,int n0,int n1){
    for (int i=0;i<n1;++i){
        for (int j=0;j<n0;++j){
            printf("%1.4f     ",u[i+j*n0]);
        }
        printf("\n");
    }
    printf("\n\n");
}

// Displays a real array
void disp_array2(double *u,int n){
    for (int i=0;i<n;++i){
            printf("%1.4f     ",u[i]);
    }
    printf("\n\n");
}

// Displays a complex array
void disp_carray(fftw_complex *u,int n0,int n1){
    int ii;
    for (int i=0;i<n1;++i){
        for (int j=0;j<n0;++j){
            if (i<=n1/2){
                printf("%1.4f+i%1.4f     ",u[i+j*n0][0],u[i+j*n0][1]);
            }
            else{
                ii=n1-i;
                printf("%1.4f+i%1.4f     ",u[ii+j*n0][0],-u[ii+j*n0][1]);
            }
        }
        printf("\n");
    }
    printf("\n\n");
}

// Displays a complex array as a vector
void disp_carray2(fftw_complex *u,int n){
    for (int i=0;i<n;++i){
        printf("%1.4f+i%1.4f     ",u[i][0],u[i][1]);
    }
    printf("\n\n");
}


// Displays a complex array
void normalize(double *u,int n){
    #pragma omp parallel for 
    for (int i=0;i<n;++i){
        u[i]=u[i]/n;
    }
}

// Main function
void VSNR_ADMM(double *u0, double *psi, int n0, int n1,int nit,double beta,double *u){
    fftw_plan plan_u0,plan_psi;
    fftw_complex *fpsi,*fu0;
    fftw_complex *fd1,*fd2,*fphi1,*fphi2,*fphi;
    double *d1u0,*d2u0;
    
    int n=n0*n1;
    int i,j;
    int n_=n0*(n1/2+1);
    
    printf("VSNR2D - ADMM. Working with %i threads \n",omp_get_max_threads());
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    fpsi=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    fu0=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    d1u0= (double*) malloc(sizeof(double)*n);
    d2u0= (double*) malloc(sizeof(double)*n);
    
    // Plans for the main loop
    double *tmp1= (double*) malloc(sizeof(double)*n);
    double *tmp2= (double*) malloc(sizeof(double)*n);
    fftw_complex *ftmp1=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    fftw_complex *ftmp2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    
    fftw_plan plan_tmp1 = fftw_plan_dft_r2c_2d(n0,n1,tmp1,ftmp1, FFTW_MEASURE);
    fftw_plan plan_tmp2 = fftw_plan_dft_r2c_2d(n0,n1,tmp2,ftmp2, FFTW_MEASURE);
    fftw_plan plan_ftmp1 = fftw_plan_dft_c2r_2d(n0,n1,ftmp1,tmp1, FFTW_MEASURE);
    fftw_plan plan_ftmp2 = fftw_plan_dft_c2r_2d(n0,n1,ftmp2,tmp2, FFTW_MEASURE);
    
    /*fftw_plan plan_tmp1 = fftw_plan_dft_r2c_2d(n0,n1,tmp1,ftmp1, FFTW_ESTIMATE);
    fftw_plan plan_tmp2 = fftw_plan_dft_r2c_2d(n0,n1,tmp2,ftmp2, FFTW_ESTIMATE);
    fftw_plan plan_ftmp1 = fftw_plan_dft_c2r_2d(n0,n1,ftmp1,tmp1, FFTW_ESTIMATE);
    fftw_plan plan_ftmp2 = fftw_plan_dft_c2r_2d(n0,n1,ftmp2,tmp2, FFTW_ESTIMATE);*/
    
    // Should use FFTW_MEASURE for iterative algorithms, but beware of initialization
    plan_u0 = fftw_plan_dft_r2c_2d(n0,n1,u0,fu0, FFTW_ESTIMATE);
    plan_psi = fftw_plan_dft_r2c_2d(n0,n1,psi,fpsi, FFTW_ESTIMATE);
    fftw_execute(plan_u0); // fu0=fftn(u0)
    fftw_execute(plan_psi); // fpsi=fftn(psi)
    fftw_destroy_plan(plan_u0);
    fftw_destroy_plan(plan_psi);
    
    // Computes d1 and fd1
    double *d1 = (double*) calloc(n0*n1,sizeof(double));
    d1[0]=1;d1[n1-1]=-1;
    fd1=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    fftw_plan plan_d1 = fftw_plan_dft_r2c_2d(n0,n1,d1,fd1, FFTW_ESTIMATE);
    fftw_execute(plan_d1); // fd1=fftn(d1)
    fftw_destroy_plan(plan_d1);
    free(d1);    
    
    // Computes d2 and fd2
    double *d2 = (double*) calloc(n0*n1,sizeof(double));
    fd2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    d2[0]=1;d2[(n0-1)*n1]=-1;
    fftw_plan plan_d2 = fftw_plan_dft_r2c_2d(n0,n1,d2,fd2, FFTW_ESTIMATE);
    fftw_execute(plan_d2); // fd2=fftn(d2)
    fftw_destroy_plan(plan_d2);
    free(d2);
    
    // Computes d1u0
    product_carray(fd1,fu0,ftmp1,n_);
    fftw_plan plan_d1u0 = fftw_plan_dft_c2r_2d(n0,n1,ftmp1,d1u0, FFTW_ESTIMATE);
    fftw_execute(plan_d1u0); // d1u0=ifftn(fd1.*fu0)
    normalize(d1u0,n);
    fftw_destroy_plan(plan_d1u0);   
    
    // Computes d2u0
    product_carray(fd2,fu0,ftmp2,n_);
    fftw_plan plan_d2u0 = fftw_plan_dft_c2r_2d(n0,n1,ftmp2,d2u0, FFTW_ESTIMATE);
    fftw_execute(plan_d2u0); // d2u0=ifftn(fd2.*fu0)
    normalize(d2u0,n);
    fftw_destroy_plan(plan_d2u0);
   
    fftw_free(fu0); // This is unused until the end
    
    // Computes fphi1 and fphi2
    fphi1=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    fphi2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    product_carray(fd1,fpsi,fphi1,n_); //fphi1=fpsi.*fd1
    product_carray(fd2,fpsi,fphi2,n_); //fphi2=fpsi.*fd2
        
    fftw_free(fd1);
    fftw_free(fd2);
    
    // Computes fphi
    fphi=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);

    #pragma omp parallel for 
    for (i=0;i<n_;++i){
        fphi[i][0]=1 + beta*(fphi1[i][0]*fphi1[i][0] + fphi1[i][1]*fphi1[i][1] + fphi2[i][0]*fphi2[i][0] + fphi2[i][1]*fphi2[i][1]);
        fphi[i][1]=0;
    }

    // Initialization
    double *x= (double*) calloc(n,sizeof(double));
    double *y1= (double*) calloc(n,sizeof(double));
    double *y2= (double*) calloc(n,sizeof(double));
    double *lambda1= (double*) calloc(n,sizeof(double));
    double *lambda2= (double*) calloc(n,sizeof(double));
    double a1,b1,a2,b2;
    fftw_complex *fx=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_);
    double ng,t1,t2;
    
    // Main algorithm
    for (int k=0;k<nit;++k){
        ///////////////////////////////////////////////////////////
        // First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        ///////////////////////////////////////////////////////////
        // ftmp1=conj(fphi1).*(fftn(-lambda1+beta*y1));
        // ftmp2=conj(fphi2).*(fftn(-lambda2+beta*y2));   
        #pragma omp parallel for 
        for (i=0;i<n;++i){
            tmp1[i]=-lambda1[i]+beta*y1[i];
            tmp2[i]=-lambda2[i]+beta*y2[i];
        }
        
        fftw_execute(plan_tmp1);
        fftw_execute(plan_tmp2);

        #pragma omp parallel for private(a1,a2,b1,b2)
        for (i=0;i<n_;++i){
            a1=fphi1[i][0];b1=fphi1[i][1];
            a2=ftmp1[i][0];b2=ftmp1[i][1];
            ftmp1[i][0]=a1*a2+b1*b2;
            ftmp1[i][1]=b2*a1-b1*a2;

            a1=fphi2[i][0];b1=fphi2[i][1];
            a2=ftmp2[i][0];b2=ftmp2[i][1];
            ftmp2[i][0]=a1*a2+b1*b2;
            ftmp2[i][1]=b2*a1-b1*a2;
        }
        
        // fx=(tmp1+tmp2)/fphi;
        #pragma omp parallel for 
        for (i=0;i<n_;++i){
            fx[i][0]=(ftmp1[i][0]+ftmp2[i][0])/fphi[i][0];
            fx[i][1]=(ftmp1[i][1]+ftmp2[i][1])/fphi[i][0];
        }
                
        ///////////////////////////////////////////////////////////
        // Second step y update : y=prox_{f1/beta}(Ax+lambda/beta)
        ///////////////////////////////////////////////////////////
        product_carray(fphi1,fx,ftmp1,n_);
        product_carray(fphi2,fx,ftmp2,n_);
        fftw_execute(plan_ftmp1); // tmp1 = Ax1
        normalize(tmp1,n);
        fftw_execute(plan_ftmp2); // tmp2 = Ax2
        normalize(tmp2,n);
        
        #pragma omp parallel for private(t1,t2,ng)
        for (i=0;i<n;++i){
            t1=d1u0[i]-(tmp1[i]+lambda1[i]/beta);
            t2=d2u0[i]-(tmp2[i]+lambda2[i]/beta);
            ng=sqrt(t1*t1+t2*t2);            
                        
            if (ng>1.0/beta){
                y1[i]=d1u0[i]-t1*(1.0-1.0/(beta*ng));
                y2[i]=d2u0[i]-t2*(1.0-1.0/(beta*ng));
            }
            else {
                y1[i]=d1u0[i];
                y2[i]=d2u0[i];
            }
        }
        
        ///////////////////////////////////////////////////////////
        // Third step lambda update
        ///////////////////////////////////////////////////////////
        #pragma omp parallel for 
        for (i=0;i<n;++i){
            lambda1[i]=lambda1[i]+beta*(tmp1[i]-y1[i]);
            lambda2[i]=lambda2[i]+beta*(tmp2[i]-y2[i]);
        } 
    }
    
    // Free plans
    fftw_destroy_plan(plan_tmp1);
    fftw_destroy_plan(plan_tmp2);
    fftw_destroy_plan(plan_ftmp1);
    fftw_destroy_plan(plan_ftmp2);
    
    // Last but not the least : u=u0-psi*x
    product_carray(fx,fpsi,ftmp1,n_);
    fftw_plan plan_fu = fftw_plan_dft_c2r_2d(n0,n1,ftmp1,u, FFTW_ESTIMATE);
    fftw_execute(plan_fu);
    normalize(u,n);
    fftw_destroy_plan(plan_fu);
    #pragma omp parallel for 
    for (i=0;i<n;++i){
        u[i]=u0[i]-u[i];
    }
    
    // Free memory
    fftw_free(fpsi);
    fftw_free(fphi);
    fftw_free(fphi1);
    fftw_free(fphi2);
    fftw_free(ftmp1);
    fftw_free(ftmp2);
    fftw_free(fx);
    
    free(d1u0);
    free(d2u0);
    free(x);
    free(y1);
    free(y2);
    free(lambda1);
    free(lambda2);
    free(tmp1);
    free(tmp2);
}

// Entry point for Matlab
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Ouput : u
    // Input : u0, psi, nit, beta
    
    // Check for proper input
    switch(nrhs) {
        case 4 : /*mexPrintf("Good call.\n");*/
            break;
        default: mexErrMsgTxt("Bad number of inputs.\n");
        break;
    }
    if (nlhs > 1) {mexErrMsgTxt("Too many outputs.\n");}
    
    time_t start;
    int n0,n1,nit,i;
    double beta, *u0,*psi,*u;
    
    // Get input arguments
    // Note that n0 and n1 are reversed because of row major in C VS column major format in Matlab
    n1=mxGetM(prhs[0]); //number of rows
    n0=mxGetN(prhs[0]); //number of columns
    u0=mxGetPr(prhs[0]);
    psi=mxGetPr(prhs[1]);
    nit=(int)*mxGetPr(prhs[2]);
    beta=*mxGetPr(prhs[3]);
    
    // Create output arguments
    plhs[0] = mxCreateDoubleMatrix(n1,n0,mxREAL);
    u=mxGetPr(plhs[0]);
    
    // Main function
    VSNR_ADMM(u0,psi,n0,n1,nit,beta,u);
}