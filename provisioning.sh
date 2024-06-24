#!/bin/bash

# This file will be sourced in init.sh

# https://raw.githubusercontent.com/ai-dock/comfyui/main/config/provisioning/default.sh

# Packages are installed after nodes so we can fix them...

PYTHON_PACKAGES=(
    #"opencv-python==4.7.0.72"
    "pillow==10.2.0 insightface onnxruntime onnxruntime-gpu"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
    "https://github.com/audioscavenger/save-image-extended-comfyui"
    "https://github.com/StartHua/Comfyui_joytag"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/shiimizu/ComfyUI_smZNodes"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/wallish77/wlsh_nodes"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/theUpsider/ComfyUI-Logic"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/edenartlab/eden_comfy_pipelines"
    "https://github.com/KoreTeknology/ComfyUI-Universal-Styler"
    "https://github.com/coolzilj/ComfyUI-LJNodes"
    "https://github.com/jags111/efficiency-nodes-comfyui"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/cubiq/PuLID_ComfyUI"
    "https://github.com/mav-rik/facerestore_cf"
)

CHECKPOINT_MODELS=(
    #"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/iniverseMixXLSFWNSFW_guofengTurboV14.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/wildcardxAsian_v10.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/majicmixRealistic_v7-inpainting.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/White_RealisticSimulator_Pony_v20.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/pinkiepiePonyMix_v30BetaFp16.safetensors"
    #"https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/leosamsHelloworldXL_helloworldXL70.safetensors"
    #"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

LORA_MODELS=(
    #"https://civitai.com/api/download/models/16576"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/genshin_v4.safetensors"
    "https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/genshin-ganyu-ingame-ponyxl-lora-nochekaiser.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/breastsizeslideroffset.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/m99_labiaplasty_pussy_2.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/more_details.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/nagachichiD.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/picxer_real.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/pytorch_lora_weights.safetensors"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/Hand_v3_SD1.5.safetensors"
    #"https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/roxy-migurdia-s1-ponyxl-lora-nochekaiser.safetensors"
    "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Age%20Slider%20V2_alpha1.0_rank4_noxattn_last.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Body%20Type_alpha1.0_rank4_noxattn_last.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Breast%20Sag%20Slider_alpha1.0_rank4_noxattn_last.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Breast%20Slider%20-%20Pony_alpha1.0_rank4_noxattn_last.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Clothed%20Slider_alpha1.0_rank4_noxattn_last.safetensors"
    #"https://huggingface.co/GritTin/LoraStableDiffusion/resolve/main/Skin%20Color_alpha1.0_rank4_noxattn_250steps.safetensors"
)

VAE_MODELS=(
    #"https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors"
    "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    #"https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
)

EMBEDDINGS=(
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/epiCPhoto.pt"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/CyberRealistic_Negative-neg.pt"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/DeepNegative_xl_v1.safetensors"
)

ULTRALYTICS=(
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/nipple.pt"
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/pussyV2.pt"
)

CLIP_VISION=(
    "https://huggingface.co/GritTin/modelsStableDiffusion/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
)

IPADAPTER=(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light.bin"
    "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    DISK_GB_AVAILABLE=$(($(df --output=avail -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_USED=$(($(df --output=used -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_ALLOCATED=$(($DISK_GB_AVAILABLE + $DISK_GB_USED))
    provisioning_print_header
    provisioning_get_nodes
    provisioning_install_python_packages
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ckpt" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/lora" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/embeddings" \
        "${EMBEDDINGS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ultralytics/bbox" \
        "${ULTRALYTICS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/clip_vision" \
        "${CLIP_VISION[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ipadapter" \
        "${IPADAPTER[@]}"
    provisioning_print_end
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    micromamba -n comfyui run ${PIP_INSTALL} -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                micromamba -n comfyui run ${PIP_INSTALL} -r "${requirements}"
            fi
        fi
    done
}

function provisioning_install_python_packages() {
    if [ ${#PYTHON_PACKAGES[@]} -gt 0 ]; then
        micromamba -n comfyui run ${PIP_INSTALL} ${PYTHON_PACKAGES[*]}
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    if [[ $DISK_GB_ALLOCATED -ge $DISK_GB_REQUIRED ]]; then
        arr=("$@")
    else
        printf "WARNING: Low disk space allocation - Only the first model will be downloaded!\n"
        arr=("$1")
    fi
    
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
}

provisioning_start