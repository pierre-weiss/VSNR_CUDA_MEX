function mex_all()
% tested on Linux 64-bit with Matlab R2016a and CUDA 7.5

% checks
if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-7.5 /usr/local/cuda"')
end

% need to be in the current directory for mexcuda
oldpath = pwd;
newpath = fileparts(mfilename('fullpath'));
cd(newpath);

% if the mexcuda fails, we are stuck - rethrow error
try
    mex_all_compile();
    cd(oldpath)
catch ME
    cd(oldpath)
    rethrow(ME)
end


function mex_all_compile()

% clean
delete VSNR_ADMM_2D_GPU_TEST.mex*

files = {'VSNR_ADMM_2D_GPU_DOUBLE'};
%files = {'VSNR_ADMM_2D_GPU_DOUBLE'};

for k = 1:numel(files)
    cmd1 = ['/usr/local/cuda-8.0/bin/nvcc -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE' ...
        ' -I"/usr/local/cuda-8.0/include"' ...
        ' -I"/usr/local/MATLAB/R2016b/extern/include"' ...
        ' -I"/usr/local/MATLAB/R2016b/simulink/include"' ...
        ' -I"/usr/local/MATLAB/R2016b/toolbox/distcomp/gpu/extern/include/"' ...
        ' -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50' ...
        ' -std=c++11 --compiler-options=-ansi,-fexceptions,-fPIC,-fno-omit-frame-pointer,-lcufft,-lcufftw,-pthread -O -DNDEBUG' ...
        ' ' files{k} '.cu -o ' files{k} '.o'];
    
    cmd2 = ['/usr/bin/g++ -pthread -Wl,--no-undefined -Wl,--no-as-needed -shared -O' ...
        ' -Wl,--version-script,"/usr/local/MATLAB/R2016b/extern/lib/glnxa64/mexFunction.map"' ...
        ' ' strcat(files{k},'.o') ' -ldl' ...
        ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcusparse.so' ... % -lcusparse
        ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcublas_static.a' ... % -lcublas_static
        ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcusparse_static.a' ... % -lcusparse_static
        ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libculibos.a' ... % -lculibos'
        ' -L/usr/local/cuda-8.0/lib64 -Wl,-rpath-link,/usr/local/MATLAB/R2016b/bin/glnxa64' ...
        ' -L"/usr/local/MATLAB/R2016b/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu -lcufft -lcufftw' ...
        ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so' ... % /usr/local/MATLAB/R2016a/bin/glnxa64/libcudart.so.7.5
        ' -o ' files{k} '.mexa64'];
    
    cmd3 = ['rm -f ' files{k} '.o'];
    
    disp([files{k} '.cu'])
    if system(cmd1); error('%s failed step 1',files{k}); end
    if system(cmd2); error('%s failed step 2',files{k}); end
    if system(cmd3); error('%s failed step 3',files{k}); end
    disp('MEX completed successfully.')
    
end

