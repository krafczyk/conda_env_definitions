# 202603_big_env

This environment is tuned for schumann and it's GTX1080 GPUs. Support for SM6.1 has been dropped in recent CUDA and cudnn and this results in strange errors. Thus I studied the cuda requirements of the various pip packages to discover the best path.

I found I couldn't use plain pip dependencies as written directly as the dependencies aren't properly specified for all frameworks. Specifically tensorflow and pytorch are pinned to specific versions even though they would work with a range of versions. Thus I picked one framework (torch I believe) to pull in the cuda dependencies, and installed the plain tensorflow pip package. Thankfully the TF team have fixed the issue of tensorflow being plain cpu, thus I don't NEED to install the "tensorflow[with-cuda]" version for it to be able to use an existing cuda. Thus The combination of the yaml file with pip requirements file results in a working tri-framework environment.

I also had to move ray to pip, ray-all from conda was also pulling in a competing cuda. This setup has one cuda installation, and all the ML frameworks use that one.

After this, when installing, we have to run `bash pip_cuda_linking.sh`. This is required because tensorflow doesn't look in the right place for some tools currently.
