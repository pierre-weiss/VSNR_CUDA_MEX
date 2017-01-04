// VSNR ADMM 2D ON GPU WITH MATLAB

#include <math.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

// Computes out=u1.*u2
__global__ void product_carray(cufftComplex *u1,cufftComplex *u2,cufftComplex *out,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        out[i].x=u1[i].x*u2[i].x-u1[i].y*u2[i].y;
        out[i].y=u1[i].y*u2[i].x+u1[i].x*u2[i].y;

        i+=blockDim.x*gridDim.x;
    }
}

// Normalize an array
__global__ void normalize( cufftReal* u, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        u[i]=u[i]/n;

        i+=blockDim.x*gridDim.x;
    }
}

// Adds two vectors w=u+v
__global__ void Add( float* u, float *v, float *w, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        w[i]=u[i]+v[i];

        i+=blockDim.x*gridDim.x;
    }
}

// Substracts two vectors w=u-v
__global__ void Substract( float* u, float *v, float *w, int n) {
    // indices of the thread
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        w[i]=u[i]-v[i];

        i+=blockDim.x*gridDim.x;
    }
}

// Sets finite difference 1
__global__ void setd1( float* d1, int n, int n1) {
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
__global__ void setd2( float* d2, int n, int n1) {
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
__global__ void Compute_Phi(cufftComplex* fphi1, cufftComplex* fphi2,cufftComplex *fphi, float beta, int n) {
    int i=0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        fphi[i].x=1 + beta*(fphi1[i].x*fphi1[i].x + fphi1[i].y*fphi1[i].y + fphi2[i].x*fphi2[i].x + fphi2[i].y*fphi2[i].y);
        fphi[i].y=0;

        i+=blockDim.x*gridDim.x;
    }
}

// Computes tmpi=-lambdai+beta*yi
__global__ void betay_m_lambda(cufftReal* lambda1,cufftReal* lambda2,cufftReal* y1,cufftReal* y2,cufftReal* tmp1,cufftReal* tmp2, float beta,int n) {
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
__global__ void conju_x_v(cufftComplex* u,cufftComplex* v,cufftComplex* w,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;
    float a1,a2,b1,b2;

    while (i < n){
        a1=u[i].x;b1=u[i].y;
        a2=v[i].x;b2=v[i].y;
        w[i].x=a1*a2+b1*b2;
        w[i].y=b2*a1-b1*a2;

        i+=blockDim.x*gridDim.x;
    }
}

// fx=(tmp1+tmp2)/fphi;
__global__ void updatefx(cufftComplex* ftmp1,cufftComplex* ftmp2,cufftComplex* fphi,cufftComplex* fx, int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        fx[i].x=(ftmp1[i].x+ftmp2[i].x)/fphi[i].x;
        fx[i].y=(ftmp1[i].y+ftmp2[i].y)/fphi[i].x;
        i+=blockDim.x*gridDim.x;
    }
}

__global__ void updatey(cufftReal* d1u0,cufftReal* d2u0,cufftReal* tmp1,cufftReal* tmp2,cufftReal* lambda1,cufftReal* lambda2,cufftReal* y1,cufftReal* y2,float beta,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    float ng,t1,t2;
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

__global__ void updatelambda(cufftReal* lambda,cufftReal* tmp,cufftReal* y,float beta,int n){
    int i = 0;
    i = blockIdx.x * blockDim.x + threadIdx.x ;

    while (i < n){
        lambda[i]=lambda[i]+beta*(tmp[i]-y[i]);
        i+=blockDim.x*gridDim.x;
    }
}

// Displays a real array as a vector
void disp_array2(float *u,int n){
    float* copy_u = (float*)malloc(n*sizeof(float));
    cudaMemcpy(copy_u, u, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0;i<n;++i){
        printf("%1.4f     ",copy_u[i]);
    }
    printf("\n\n");

    free(copy_u);

}

// Displays a complex array as a vector
void disp_carray2(cufftComplex *u,int n){    
    float2* copy_u = (float2*)malloc(n*sizeof(float2));
    cudaMemcpy(copy_u, u, n*sizeof(float2), cudaMemcpyDeviceToHost);

    for (int i=0;i<n;++i){
        printf("%1.4f+i%1.4f     ",copy_u[i].x,copy_u[i].y);
    }
    printf("\n\n");  
    free(copy_u);
}

// Main function
void VSNR_ADMM_GPU(float *u0,float *psi, int n0, int n1,int nit,float beta,float *u,int dimGrid, int dimBlock){

    cufftHandle plan_R2C,plan_C2R;
    cufftComplex *fpsi,*fu0;
    cufftComplex *fd1,*fd2,*fphi1,*fphi2,*fphi,*ftmp1,*ftmp2,*fx;
    cufftReal *d1,*d2,*d1u0,*d2u0,*tmp1,*tmp2,*y1,*y2,*lambda1,*lambda2;

    int n=n0*n1;
    int n_=n0*(n1/2+1);

    printf("VSNR2D - ADMM - GPU \n");

    cudaMalloc((void**)&fpsi,sizeof(cufftComplex)*n_);
    cudaMalloc((void**)&fu0,sizeof(cufftComplex)*n_);
    cudaMalloc((void**)&d1u0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&d2u0,sizeof(cufftReal)*n);

    // Allocation for the main loop
    cudaMalloc((void**)&tmp1,sizeof(cufftReal)*n);
    cudaMalloc((void**)&tmp2,sizeof(cufftReal)*n);
    cudaMalloc((void**)&ftmp1,sizeof(cufftComplex)*n_);
    cudaMalloc((void**)&ftmp2,sizeof(cufftComplex)*n_);

    cufftPlan2d(&plan_R2C,n0,n1,CUFFT_R2C);
    cufftPlan2d(&plan_C2R,n0,n1,CUFFT_C2R);
    cufftExecR2C(plan_R2C,u0,fu0); // fu0=fftn(u0)
    cufftExecR2C(plan_R2C,psi,fpsi); // fpsi=fftn(psi)

    // Computes d1 and fd1
    cudaMalloc((void**)&d1,sizeof(cufftReal)*n);
    cudaMalloc((void**)&fd1,sizeof(cufftComplex)*n_);
    setd1<<<dimGrid,dimBlock>>>( d1, n, n1);  //d1[0]=1;d1[n1-1]=-1;
    cufftExecR2C(plan_R2C,d1,fd1);
    cudaFree(d1);

    // Computes d2 and fd2
    cudaMalloc((void**)&d2,sizeof(cufftReal)*n);
    cudaMemset(d2,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&fd2,sizeof(cufftComplex)*n_);
    setd2<<<dimGrid,dimBlock>>>( d2, n, n1);  //d2[0]=1;d2[(n0-1)*n1]=-1;
    cufftExecR2C(plan_R2C,d2,fd2);
    cudaFree(d2);

    // Computes d1u0
    product_carray<<<dimGrid,dimBlock>>>(fd1,fu0,ftmp1,n_);
    cufftExecC2R(plan_C2R,ftmp1,d1u0);  // d1u0=ifftn(fd1.*fu0)
    normalize<<<dimGrid,dimBlock>>>(d1u0,n);

    // Computes d2u0
    product_carray<<<dimGrid,dimBlock>>>(fd2,fu0,ftmp2,n_);
    cufftExecC2R(plan_C2R,ftmp2,d2u0);  // d2u0=ifftn(fd2.*fu0)
    normalize<<<dimGrid,dimBlock>>>(d2u0,n);
    cudaFree(fu0); // This is unused until the end

    // Computes fphi1 and fphi2  
    cudaMalloc((void**)&fphi1,sizeof(cufftComplex)*n_);
    cudaMalloc((void**)&fphi2,sizeof(cufftComplex)*n_);
    product_carray<<<dimGrid,dimBlock>>>(fd1,fpsi,fphi1,n_); //fphi1=fpsi.*fd1
    product_carray<<<dimGrid,dimBlock>>>(fd2,fpsi,fphi2,n_); //fphi2=fpsi.*fd2   

    //disp_carray2(fpsi,n_);
    //disp_array2(psi,n);
    //disp_carray2(fd2,n_);

    cudaFree(fd1);
    cudaFree(fd2);


    // Computes fphi
    cudaMalloc((void**)&fphi,sizeof(cufftComplex) *n_);
    Compute_Phi<<<dimGrid,dimBlock>>>(fphi1, fphi2, fphi,beta,n_);    

    // Initialization
    cudaMalloc((void**)&y1,sizeof(cufftReal)*n);
    cudaMemset(y1,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&y2,sizeof(cufftReal)*n);
    cudaMemset(y2,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&lambda1,sizeof(cufftReal)*n);
    cudaMemset(lambda1,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&lambda2,sizeof(cufftReal)*n);
    cudaMemset(lambda2,0,sizeof(cufftReal)*n);
    cudaMalloc((void**)&fx,sizeof(cufftComplex)*n_);

    // Main algorithm
    for (int k=0;k<nit;++k){
        ///////////////////////////////////////////////////////////
        // First step, x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
        ///////////////////////////////////////////////////////////
        // ftmp1=conj(fphi1).*(fftn(-lambda1+beta*y1));
        // ftmp2=conj(fphi2).*(fftn(-lambda2+beta*y2));
        betay_m_lambda<<<dimGrid,dimBlock>>>(lambda1,lambda2,y1,y2,tmp1,tmp2,beta,n);
        cufftExecR2C(plan_R2C,tmp1,ftmp1);
        cufftExecR2C(plan_R2C,tmp2,ftmp2);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi1,ftmp1,ftmp1,n_);
        conju_x_v<<<dimGrid,dimBlock>>>(fphi2,ftmp2,ftmp2,n_);
        updatefx<<<dimGrid,dimBlock>>>(ftmp1,ftmp2,fphi,fx,n_);

        ///////////////////////////////////////////////////////////
        // Second step y update : y=prox_{f1/beta}(Ax+lambda/beta)
        ///////////////////////////////////////////////////////////
        product_carray<<<dimGrid,dimBlock>>>(fphi1,fx,ftmp1,n_);
        product_carray<<<dimGrid,dimBlock>>>(fphi2,fx,ftmp2,n_);
        cufftExecC2R(plan_C2R,ftmp1,tmp1); // tmp1 = Ax1
        normalize<<<dimGrid,dimBlock>>>(tmp1,n);
        cufftExecC2R(plan_C2R,ftmp2,tmp2); // tmp2 = Ax2
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
    cufftExecC2R(plan_C2R,ftmp1,u);
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

    cufftDestroy(plan_R2C);
    cufftDestroy(plan_C2R);
}
