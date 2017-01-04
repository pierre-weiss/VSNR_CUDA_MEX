% function u=denoise_VSNR_ADMM2(u0,psi,nit,beta)
%
% This function denoises an image u0, degraded by a noise defined through
% filter psi. It solves an optimization problem with the ADMM.
%
% INPUT : 
% - u0: input 2D grayscale image.
% - psi: input filter to describe the noise.
% - nit: number of iterations. 
% - beta: parameter of the ADMM.
%
% OUTPUT :
% - u: denoised image.
%
% Developer: Pierre Weiss, December 2016.
function u=denoise_VSNR_ADMM2(u0,psi,nit,beta)

d1=zeros(size(u0));
d2=zeros(size(u0));

d1(1,1)=1;d1(end,1)=-1;
d2(1,1)=1;d2(1,end)=-1;
fd1=fftn(d1);fd2=fftn(d2);

d1u0=ifftn(fd1.*fftn(u0));
d2u0=ifftn(fd2.*fftn(u0));

fpsi=fftn(psi);
fphi1=fftn(d1).*fpsi;
fphi2=fftn(d2).*fpsi;

fphi=1+beta*(conj(fphi1).*fphi1 + conj(fphi2).*fphi2);

y1=zeros(size(u0));
y2=zeros(size(u0));
lambda1=zeros(size(u0));
lambda2=zeros(size(u0));

for k=1:nit   
    %% First step x update : (I+beta ATA)x = AT (-lambda+beta*ATy)
    tmp1=-lambda1+beta*y1;
    tmp2=-lambda2+beta*y2;
    ftmp1=fftn(tmp1);
    ftmp2=fftn(tmp2);
    
    ftmp1=conj(fphi1).*(ftmp1);
    ftmp2=conj(fphi2).*(ftmp2);
    fx=(ftmp1+ftmp2)./(fphi);
    
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
