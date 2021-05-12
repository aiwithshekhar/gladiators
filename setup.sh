unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/heh3kor/cuda_home/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/heh3kor/cuda_home/extras/CUPTI/lib64
export CUDA_HOME=/home/heh3kor/cuda_home


cd mmcv
# FORCE_CUDA=0 for cpu 

FORCE_CUDA="1" MMCV_WITH_OPS="1" pip install -e .
cd ..

cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
cd ..