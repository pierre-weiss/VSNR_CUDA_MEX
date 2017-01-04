% This is a script to test the different parallel implementations of VSNR.
% Developer: Pierre Weiss, January 2016.

n0=512;
n1=512;
[X,Y]=meshgrid(linspace(-1,1,n0),linspace(-1,1,n1));
u=double(sqrt(X.^2+Y.^2)<=0.8);
rng(1);

psi=zeros(size(u));
psi(1,1:10)=1;
psi=psi/sum(psi(:));
lambda=randn(size(u));

b=ifftn(fftn(lambda).*fftn(psi));
u0=u+b;

figure(1);colormap gray;imagesc(u);title('Original image');axis equal
figure(2);colormap gray;imagesc(u0);title('Noisy image');axis equal

%% Comparison APGD / ADMM
% nit=1000;
% tic;[u1,lambda1,CF1]=denoise_VSNR_APGD(u0,10*psi,nit);toc;
% tic;[u2,lambda2,CF2]=denoise_VSNR_ADMM(u0,10*psi,nit);toc;

%figure(4);colormap gray;imagesc(u1);title('denoised APGD');axis equal
%figure(5);colormap gray;imagesc(u2);title('denoised ADMM');axis equal
%figure(6);plot(1:nit,CF1,'r',1:nit,CF2,'g');
%legend('APGD','ADMM')

%% ADMM
gu0d=gpuArray(u0);
gpsid=10*gpuArray(psi);

gu0s=gpuArray(single(u0));
gpsis=10*gpuArray(single(psi));


beta=10;
nit=30;

disp('MATLAB 1 THREAD')
maxNumCompThreads(1);tic;u1=denoise_VSNR_ADMM2(u0,10*psi,nit,beta);toc;time1=toc;
disp('MATLAB 20 THREADS')
maxNumCompThreads(20);tic;u2=denoise_VSNR_ADMM2(u0,10*psi,nit,beta);toc;time2=toc;
disp('C 1 THREAD')
maxNumCompThreads(1);tic;u3=VSNR_ADMM_2D(u0,10*psi,nit,beta);toc;time3=toc;
disp('C 20 THREADS')
maxNumCompThreads(20);tic;u4=VSNR_ADMM_2D(u0,10*psi,nit,beta);toc;time4=toc;

disp('MATLAB GPU DOUBLE')
tic;gu5=denoise_VSNR_ADMM_GPU_DOUBLE(gu0d,gpsid,nit,beta);toc;time5=toc;
u5=real(gather(gu5));

disp('MATLAB GPU SINGLE')
tic;gu6=denoise_VSNR_ADMM_GPU_SINGLE(gu0s,gpsis,nit,beta);toc;time6=toc;
u6=real(gather(gu6));

disp('C GPU SINGLE')
dimGrid=4096;dimBlock=2048;
tic;gu7=VSNR_ADMM_2D_GPU_SINGLE(gu0s,gpsis,nit,beta,dimGrid,dimBlock);toc;time7=toc;
u7=real(gather(gu7));

disp('C GPU DOUBLE')
dimGrid=4096;dimBlock=2048;
tic;gu8=VSNR_ADMM_2D_GPU_DOUBLE(gu0d,gpsid,nit,beta,dimGrid,dimBlock);toc;time8=toc;
u8=real(gather(gu8));

fprintf('Acceleration MATLAB multithread: %3.2f\n',time1/time2)
fprintf('Acceleration C 1 thread: %3.2f\n',time1/time3)
fprintf('Acceleration C multithread: %3.2f\n',time1/time4)
fprintf('Acceleration MATLAB GPU - DOUBLE: %3.2f\n',time1/time5)
fprintf('Acceleration MATLAB GPU - SINGLE : %3.2f\n',time1/time6)
fprintf('Acceleration C GPU SINGLE: %3.2f\n',time1/time7)
fprintf('Acceleration C GPU DOUBLE: %3.2f\n',time1/time8)
