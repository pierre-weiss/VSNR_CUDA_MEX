%% Simplified function to copy for C implementation
function u=denoise_VSNR_ADMM_GPU_SINGLE(u0,psi,nit,beta)

d1=gpuArray(single(zeros(size(u0))));
d2=gpuArray(single(zeros(size(u0))));

d1(1,1)=1;d1(end,1)=-1;
d2(1,1)=1;d2(1,end)=-1;
fd1=fftn(d1);fd2=fftn(d2);

d1u0=ifftn(fd1.*fftn(u0));
d2u0=ifftn(fd2.*fftn(u0));

fpsi=fftn(psi);
fphi1=fftn(d1).*fpsi;
fphi2=fftn(d2).*fpsi;
fphi=1+beta*(conj(fphi1).*fphi1 + conj(fphi2).*fphi2);

y1=gpuArray(single(zeros(size(u0))));
y2=gpuArray(single(zeros(size(u0))));
lambda1=gpuArray(single(zeros(size(u0))));
lambda2=gpuArray(single(zeros(size(u0))));

for k=1:nit   
    %% First step x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
    tmp1=conj(fphi1).*(fftn(-lambda1+beta*y1));
    tmp2=conj(fphi2).*(fftn(-lambda2+beta*y2));
    
    fx=(tmp1+tmp2)./(fphi);
    
    %% Second step y update : y=prox_{f1/beta}(Ax+lambda/beta)
    Ax1=ifftn(fphi1.*fx);
    Ax2=ifftn(fphi2.*fx);
    tmp1=d1u0-(Ax1+lambda1/beta);
    tmp2=d2u0-(Ax2+lambda2/beta);
    ng=sqrt(tmp1.^2+tmp2.^2);
    I=(ng>1/beta);
    
    y1=d1u0-I.*tmp1.*(1 - 1./(beta*ng));
    y2=d2u0-I.*tmp2.*(1 - 1./(beta*ng));
            
    %% Third step lambda update  
    lambda1=lambda1+beta*(Ax1-y1);
    lambda2=lambda2+beta*(Ax2-y2);    
end

u=u0-ifftn(fx.*fftn(psi));
