# use python=3.10
# create a conda environment
# conda create -n pcd python=3.10
# conda activate pcd

# install tensorflow
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing 

# install jax
pip install "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install pytorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# install detectron2
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/detectron2-0.6%2B2a420ed/detectron2-0.6%2B2a420edpt2.0.0cu118-cp310-cp310-linux_x86_64.whl

# install maniskill env
cd third_party/ManiSkill2_real2sim
pip install -e .
cd ..

# install grounded sam 2
cd grounded_sam_2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..

# install SED
cd SED/open_clip
make install
cd ../..

# install yolo world
cd yolo_world
pip install -e .
cd ..

# install octo
cd octo
pip install -e .
cd ..

# install requirements
pip install numpy==1.24.4
pip install -r SED/requirements.txt
pip install -r inpaint_anything/lama/requirements.txt
cd ..
pip install -r requirements_full_install.txt

# reforce the installation of the following packages
pip install timm==0.9.10 accelerate==1.1.1
pip install "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
