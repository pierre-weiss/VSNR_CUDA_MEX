// VSNR ADMM 2D ON GPU WITH MATLAB

#include <math.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

// Computes out=u1.*u2
__global__ void product_carray(cufftDoubleComplex *u1,cufftDoubleComplex *u2,cufftDoubleComplex *out,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        out[i].x=u1[i].x*u2[i].x-u1[i].y*u2[i].y;
        out[i].y=u1[i].y*u2[i].x+u1[i].x*u2[i].y;
        
        i+=blockDim.x*gridDim.x;
    }
}

// Normalize an array
__global__ void normalize( cufftDoubleReal* u, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        u[i]=u[i]/n;
        
        i+=blockDim.x*gridDim.x;
    }
}

// Adds two vectors w=u+v
__global__ void Add( double* u, double *v, double *w, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        w[i]=u[i]+v[i];
        
        i+=blockDim.x*gridDim.x;
    }
}

// Substracts two vectors w=u-v
__global__ void Substract( double* u, double *v, double *w, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        w[i]=u[i]-v[i];
        
        i+=blockDim.x*gridDim.x;
    }
}

// Sets finite difference 1
__global__ void setd1( double* d1, int n, int n1) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        if (i==0) {d1[i]=1;}
        else if (i==n1-1) {d1[i]=-1;}
        else {d1[i]=0;}
        
        i+=blockDim.x*gridDim.x;
    }
}

// Sets finite difference 2
__global__ void setd2( double* d2, int n, int n1) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        if (i==0) {d2[i]=1;}
        else if (i==n-n1) {d2[i]=-1;}
        else {d2[i]=0;}
        
        i+=blockDim.x*gridDim.x;
    }
}

// Compute Phi
__global__ void Compute_Phi(cufftDoubleComplex* fphi1, cufftDoubleComplex* fphi2,cufftDoubleComplex *fphi, double beta, int n) {
    int i=0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        fphi[i].x=1 + beta*(fphi1[i].x*fphi1[i].x + fphi1[i].y*fphi1[i].y + fphi2[i].x*fphi2[i].x + fphi2[i].y*fphi2[i].y);
        fphi[i].y=0;
        
        i+=blockDim.x*gridDim.x;
    }
}

// Computes tmpi=-lambdai+beta*yi
__global__ void betay_m_lambda(cufftDoubleReal* lambda1,cufftDoubleReal* lambda2,cufftDoubleReal* y1,cufftDoubleReal* y2,cufftDoubleReal* tmp1,cufftDoubleReal* tmp2, double beta,int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        tmp1[i]=-lambda1[i]+beta*y1[i];
        tmp2[i]=-lambda2[i]+beta*y2[i];
        
        i+=blockDim.x*gridDim.x;
    }
}

// Computes w=conj(u)*v
__global__ void conju_x_v(cufftDoubleComplex* u,cufftDoubleComplex* v,cufftDoubleComplex* w,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    double a1,a2,b1,b2;
    
    while (i < n){
        a1=u[i].x;b1=u[i].y;
        a2=v[i].x;b2=v[i].y;
        w[i].x=a1*a2+b1*b2;
        w[i].y=b2*a1-b1*a2;
                
        i+=blockDim.x*gridDim.x;
    }
}

// fx=(tmp1+tmp2)/fphi;
__global__ void updatefx(cufftDoubleComplex* ftmp1,cufftDoubleComplex* ftmp2,cufftDoubleComplex* fphi,cufftDoubleComplex* fx, int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        fx[i].x=(ftmp1[i].x+ftmp2[i].x)/fphi[i].x;
        fx[i].y=(ftmp1[i].y+ftmp2[i].y)/fphi[i].x;
        i+=blockDim.x*gridDim.x;
    }
}

__global__ void updatey(cufftDoubleReal* d1u0,cufftDoubleReal* d2u0,cufftDoubleReal* tmp1,cufftDoubleReal* tmp2,cufftDoubleReal* lambda1,cufftDoubleReal* lambda2,cufftDoubleReal* y1,cufftDoubleReal* y2,double beta,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    double ng,t1,t2;
    while (i < n){
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
        i+=blockDim.x*gridDim.x;
    }
}

__global__ void updatelambda(cufftDoubleReal* lambda,cufftDoubleReal* tmp,cufftDoubleReal* y,double beta,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    while (i < n){
        lambda[i]=lambda[i]+beta*(tmp[i]-y[i]);
        i+=blockDim.x*gridDim.x;
    }
}

// Displays a real array as a vector
void disp_array2(double *u,int n){
    double* copy_u = (double*)malloc(n*sizeof(double));
    cudaMemcpy(copy_u, u, n*sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<n;++i){
            printf("%1.4f     ",copy_u[i]);
    }
    printf("\n\n");
    
    free(copy_u);

}

// Displays a complex array as a vector
void disp_carray2(cufftDoubleComplex *u,int n){    
    double2* copy_u = (double2*)malloc(n*sizeof(double2));
    cudaMemcpy(copy_u, u, n*sizeof(double2), cudaMemcpyDeviceToHost);

    for (int i=0;i<n;++i){
        printf("%1.4f+i%1.4f     ",copy_u[i].x,copy_u[i].y);
    }
    printf("\n\n");  
    free(copy_u);
}

// Main function
void VSNR_ADMM_GPU(double *u0,double *psi, int n0, int n1,int nit,double beta,double *u,int dimGrid, int dimBlock){
    
    cufftHandle plan_D2Z,plan_Z2D;
    cufftDoubleComplex *fpsi,*fu0;
    cufftDoubleComplex *fd1,*fd2,*fphi1,*fphi2,*fphi,*ftmp1,*ftmp2,*fx;
    cufftDoubleReal *d1,*d2,*d1u0,*d2u0,*tmp1,*tmp2,*y1,*y2,*lambda1,*lambda2;
    
    int n=n0*n1;
    int n_=n0*(n1/2+1);
    
    printf("VSNR2D - ADMM - GPU \n");
    
    cudaMalloc((void**)&fpsi,sizeof(cufftDoubleComplex)*n_);
    cudaMalloc((void**)&fu0,sizeof(cufftDoubleComplex)*n_);
    cudaMalloc((void**)&d1u0,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&d2u0,sizeof(cufftDoubleReal)*n);
    
    // Allocation for the main loop
    cudaMalloc((void**)&tmp1,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&tmp2,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&ftmp1,sizeof(cufftDoubleComplex)*n_);
    cudaMalloc((void**)&ftmp2,sizeof(cufftDoubleComplex)*n_);
    
    cufftPlan2d(&plan_D2Z,n0,n1,CUFFT_D2Z);
    cufftPlan2d(&plan_Z2D,n0,n1,CUFFT_Z2D);
    cufftExecD2Z(plan_D2Z,u0,fu0); // fu0=fftn(u0)
    cufftExecD2Z(plan_D2Z,psi,fpsi); // fpsi=fftn(psi)
      
    // Computes d1 and fd1
    cudaMalloc((void**)&d1,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&fd1,sizeof(cufftDoubleComplex)*n_);
    setd1<<<dimGrid,dimBlock>>>( d1, n, n1);  //d1[0]=1;d1[n1-1]=-1;
    cufftExecD2Z(plan_D2Z,d1,fd1);
    cudaFree(d1);
    
    // Computes d2 and fd2
    cudaMalloc((void**)&d2,sizeof(cufftDoubleReal)*n);
    cudaMemset(d2,0,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&fd2,sizeof(cufftDoubleComplex)*n_);
    setd2<<<dimGrid,dimBlock>>>( d2, n, n1);  //d2[0]=1;d2[(n0-1)*n1]=-1;
    cufftExecD2Z(plan_D2Z,d2,fd2);
    cudaFree(d2);
      
    // Computes d1u0
    product_carray<<<dimGrid,dimBlock>>>(fd1,fu0,ftmp1,n_);
    cufftExecZ2D(plan_Z2D,ftmp1,d1u0);  // d1u0=ifftn(fd1.*fu0)
    normalize<<<dimGrid,dimBlock>>>(d1u0,n);
    
    // Computes d2u0
    product_carray<<<dimGrid,dimBlock>>>(fd2,fu0,ftmp2,n_);
    cufftExecZ2D(plan_Z2D,ftmp2,d2u0);  // d2u0=ifftn(fd2.*fu0)
    normalize<<<dimGrid,dimBlock>>>(d2u0,n);
    cudaFree(fu0); // This is unused until the end
    
    // Computes fphi1 and fphi2  
    cudaMalloc((void**)&fphi1,sizeof(cufftDoubleComplex)*n_);
    cudaMalloc((void**)&fphi2,sizeof(cufftDoubleComplex)*n_);
    product_carray<<<dimGrid,dimBlock>>>(fd1,fpsi,fphi1,n_); //fphi1=fpsi.*fd1
    product_carray<<<dimGrid,dimBlock>>>(fd2,fpsi,fphi2,n_); //fphi2=fpsi.*fd2   

    //disp_carray2(fpsi,n_);
    //disp_array2(psi,n);
    //disp_carray2(fd2,n_);
    
    cudaFree(fd1);
    cudaFree(fd2);
    
    
    // Computes fphi
    cudaMalloc((void**)&fphi,sizeof(cufftDoubleComplex) *n_);
    Compute_Phi<<<dimGrid,dimBlock>>>(fphi1, fphi2, fphi,beta,n_);    
    
    // Initialization
    cudaMalloc((void**)&y1,sizeof(cufftDoubleReal)*n);
    cudaMemset(y1,0,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&y2,sizeof(cufftDoubleReal)*n);
    cudaMemset(y2,0,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&lambda1,sizeof(cufftDoubleReal)*n);
    cudaMemset(lambda1,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&lambda2,sizeof(cufftDoubleReal)*n);
    cudaMemset(lambda2,0,sizeof(cufftDoubleReal)*n);
    cudaMalloc((void**)&fx,sizeof(cufftDoubleComplex)*n_);
    
    // Main algorithm
    for (int k=0;k<nit;++k){
        ///////////////////////////////////////////////////////////
        // First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        ///////////////////////////////////////////////////////////
        // ftmp1=conj(fphi1).*(fftn(-lambda1+beta*y1));
        // ftmp2=conj(fphi2).*(fftn(-lambda2+beta*y2));
        betay_m_lambda<<<dimGrid,dimBlock>>>(lambda1,lambda2,y1,y2,tmp1,tmp2,beta,n);
        cufftExecD2Z(plan_D2Z,tmp1,ftmp1);
        cufftExecD2Z(plan_D2Z,tmp2,ftmp2);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi1,ftmp1,ftmp1,n_);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi2,ftmp2,ftmp2,n_);
        updatefx<<<dimGrid,dimBlock>>>(ftmp1,ftmp2,fphi,fx,n_);
        
        ///////////////////////////////////////////////////////////
        // Second step y update : y=prox_{f1/beta}(Ax+lambda/beta)
        ///////////////////////////////////////////////////////////
        product_carray<<<dimGrid,dimBlock>>>(fphi1,fx,ftmp1,n_);
        product_carray<<<dimGrid,dimBlock>>>(fphi2,fx,ftmp2,n_);
        cufftExecZ2D(plan_Z2D,ftmp1,tmp1); // tmp1 = Ax1
        normalize<<<dimGrid,dimBlock>>>(tmp1,n);
        cufftExecZ2D(plan_Z2D,ftmp2,tmp2); // tmp2 = Ax2
        normalize<<<dimGrid,dimBlock>>>(tmp2,n);
        updatey<<<dimGrid,dimBlock>>>(d1u0,d2u0,tmp1,tmp2,lambda1,lambda2,y1,y2,beta,n);

        ///////////////////////////////////////////////////////////
        // Third step lambda update
        ///////////////////////////////////////////////////////////
        updatelambda<<<dimGrid,dimBlock>>>(lambda1,tmp1,y1,beta,n);
        updatelambda<<<dimGrid,dimBlock>>>(lambda2,tmp2,y2,beta,n);

    }
    
    // Last but not the least : u=u0-psi*x
    product_carray<<<dimGrid,dimBlock>>>(fx,fpsi,ftmp1,n_);
    cufftExecZ2D(plan_Z2D,ftmp1,u);
    normalize<<<dimGrid,dimBlock>>>(u,n);
    Substract<<<dimGrid,dimBlock>>>(u0,u,u,n);
    
    // Free memory
    cudaFree(fpsi);
    cudaFree(fphi);
    cudaFree(fphi1);
    cudaFree(fphi2);
    cudaFree(ftmp1);
    cudaFree(ftmp2);
    cudaFree(fx);
    
    cudaFree(d1u0);
    cudaFree(d2u0);
    cudaFree(y1);
    cudaFree(y2);
    cudaFree(lambda1);
    cudaFree(lambda2);
    cudaFree(tmp1);
    cudaFree(tmp2);
    
    cufftDestroy(plan_D2Z);
    cufftDestroy(plan_Z2D);
}

// Entry point for Matlab
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Ouput : u
    // Input : u0, psi, nit, beta, dimGird, dimBlock
    
    int n0,n1,nit,dimGrid,dimBlock;
    double beta;
    mwSize const *dim;
    mxGPUArray *gu0,*gpsi;
    double *u0,*psi;
    mxGPUArray *gu;
    double *u;
    char const * const errId = "parallel:gpu:VSNR_ADMM_2D_GPU_DOUBLE:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file. Should be DOUBLE gpuArray.";
    
    // Initialize the MathWorks GPU API.
    mxInitGPU();
    
    // Check for proper input
    switch(nrhs) {
        case 6 : /*mexPrintf("Good call.\n");*/
            break;
        default: mexErrMsgTxt("Bad number of inputs.\n");
        break;
    }
    if (nlhs > 1) {mexErrMsgTxt("Too many outputs.\n");}
    
    if (!(mxIsGPUArray(prhs[0]))) mexErrMsgIdAndTxt(errId, errMsg);
    if (!(mxIsGPUArray(prhs[1]))) mexErrMsgIdAndTxt(errId, errMsg);
    
    // Get input arguments
    gu0 = mxGPUCopyFromMxArray(prhs[0]); // Here I would prefer using mxGPUCreateFromMxArray to avoid a copy, but this leads to new complications: I get a const * that messes up all subsequent functions
    gpsi= mxGPUCopyFromMxArray(prhs[1]);
    nit=(int)*mxGetPr(prhs[2]);
    beta=*mxGetPr(prhs[3]);
    dimGrid=(int)*mxGetPr(prhs[4]);
    dimBlock=(int)*mxGetPr(prhs[5]);
    
    // Note that n0 and n1 are reversed because of row major in C VS column major format in Matlab
    dim = mxGPUGetDimensions(gu0);
    n1=dim[0]; //number of rows
    n0=dim[1]; //number of columns
    
    if (mxGPUGetClassID(gu0) != mxDOUBLE_CLASS) mexErrMsgIdAndTxt(errId, errMsg);
    if (mxGPUGetClassID(gpsi) != mxDOUBLE_CLASS) mexErrMsgIdAndTxt(errId, errMsg);
    u0 = (double *)(mxGPUGetData(gu0));
    psi = (double *)(mxGPUGetData(gpsi));
    
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    gu = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(gu0),
            mxGPUGetDimensions(gu0),
            mxGPUGetClassID(gu0),
            mxGPUGetComplexity(gu0),
            MX_GPU_DO_NOT_INITIALIZE);
    u = (double *)(mxGPUGetData(gu));
    
    // Main function
    VSNR_ADMM_GPU(u0,psi,n0,n1,nit,beta,u,dimGrid,dimBlock);
    
    // Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(gu);
    
    // Destroys the copies
    mxGPUDestroyGPUArray(gu0);
    mxGPUDestroyGPUArray(gpsi);
}