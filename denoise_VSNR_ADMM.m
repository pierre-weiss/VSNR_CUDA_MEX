function [u,lambda,CF]=denoise_VSNR_ADMM(u0,psi,nit,beta)

d1=zeros(size(u0));
d2=zeros(size(u0));
id=zeros(size(u0));
id(1)=1;

d1(1,1)=1;d1(end,1)=-1;
d2(1,1)=1;d2(1,end)=-1;
fd1=fftn(d1);fd2=fftn(d2);
d1u0=ifftn(fd1.*fftn(u0));
d2u0=ifftn(fd2.*fftn(u0));

fpsi=fftn(psi);
fphi1=fftn(d1).*fpsi;
fphi2=fftn(d2).*fpsi;
fphi=fftn(id)+beta*(conj(fphi1).*fphi1 + conj(fphi2).*fphi2);

x=zeros(size(u0));
y1=zeros(size(u0));
y2=zeros(size(u0));
lambda1=zeros(size(u0));
lambda2=zeros(size(u0));

CF=zeros(nit,1);
for k=1:nit
    CF(k)=Cost_Function(x,u0,psi);
    
    %% First step x update : (I+beta ATA)x = -ATlambda+beta*ATy
    tmp1=conj(fd1).*(fftn(-lambda1+beta*y1));
    tmp2=conj(fd2).*(fftn(-lambda2+beta*y2));
    fx=conj(fpsi).*(tmp1+tmp2)./(fphi);
    x=ifftn(fx);  
    
    %% Second step y update : y=prox_{f1/beta}(Ax+lambda/beta)
    Ax1=ifftn(fd1.*fx);
    Ax2=ifftn(fd2.*fx);
    tmp1=d1u0-(Ax1+lambda1/beta);    
    tmp2=d2u0-(Ax2+lambda2/beta);
    ng=sqrt(tmp1.^2+tmp2.^2);
    I=(ng>1/beta);
    
    y1=d1u0;
    y2=d2u0;
    y1(I)=y1(I)-tmp1(I).*(1 - 1./(beta*ng(I)));
    y2(I)=y2(I)-tmp2(I).*(1 - 1./(beta*ng(I)));
    
    %% Third step lambda update  
    lambda1=lambda1+beta*(Ax1-y1);
    lambda1=lambda1+beta*(Ax2-y2);
    
    %% Display
    if mod(k,100)==0
        figure(100);imagesc(u0-ifftn(fftn(x).*fftn(psi)));colormap gray;axis equal;
        title(sprintf('Iteration %i/%i',k,nit))
    end
end

lambda=x;
u=u0-ifftn(fftn(lambda).*fftn(psi));


function val=Cost_Function(lambda,u0,psi)

b=ifftn(fftn(lambda).*fftn(psi));
d1=drond1(u0-b);
d2=drond2(u0-b);
val=sum(sqrt(d1(:).^2+d2(:).^2))+norm(lambda(:))^2;