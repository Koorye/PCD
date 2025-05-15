cd pretrained

# grounded sam 2
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    wget "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi
if [ ! -f "groundingdino_swinb_cogcoor.pth" ]; then
    wget "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
fi

# sed
# PLEASE download 'sed_model_large.pth' from the google drive: https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link

# inpaint_anything
# PLEASE download 'big-lama' from the google drive: https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing

# install huggingface-cli
pip install huggingface_hub

current_dir=$(pwd)
cd ~/.cache/huggingface/hub
huggingface_cache_dir=$(pwd)
cd $current_dir

# siglip
huggingface-cli download "google/siglip-so400m-patch14-384"
if [ ! -d "siglip-so400m-patch14-384" ]; then
    ln -s ${huggingface_cache_dir}"/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/" ${current_dir}"/siglip-so400m-patch14-384"
fi

# t5-base
huggingface-cli download "google-t5/t5-base"
if [ ! -d "t5-base" ]; then
    ln -s ${huggingface_cache_dir}"/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1" ${current_dir}"/t5-base"
fi

# grounding-dino
huggingface-cli download "IDEA-Research/grounding-dino-base"
if [ ! -d "grounding-dino-base" ]; then
    ln -s ${huggingface_cache_dir}"/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb" ${current_dir}"/grounding-dino-base"
fi

# octo
huggingface-cli download "rail-berkeley/octo-base"
if [ ! -d "octo-base" ]; then
    ln -s ${huggingface_cache_dir}"/models--rail-berkeley--octo-base/snapshots/39d6c88fdbbcf6f841481a7d732f68c612d04609" ${current_dir}"/octo-base"
fi

# openvla
huggingface-cli download "openvla/openvla-7b"
if [ ! -d "openvla-7b" ]; then
    ln -s ${huggingface_cache_dir}"/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0" ${current_dir}"/openvla-7b"
fi

# rdt
huggingface-cli download "robotics-diffusion-transformer/rdt-1b"
if [ ! -d "rdt-1b" ]; then
    ln -s ${huggingface_cache_dir}"/models--robotics-diffusion-transformer--rdt-1b/snapshots/eb09036cc64ca4945051acbd1bd581d30a1d7711" ${current_dir}"/rdt-1b"
fi

# t5-xxl
huggingface-cli download "google/t5-v1_1-xxl"
if [ ! -d "t5-v1_1-xxl" ]; then
    ln -s ${huggingface_cache_dir}"/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2" ${current_dir}"/t5-v1_1-xxl"
fi