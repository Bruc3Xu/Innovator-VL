set -x

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/mnt/si00068187c7/default/innovator_vl/Innovator-VL}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

cd /mnt/si00068187c7/default/innovator_vl/Innovator-VL/

if [ -f /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh
  conda activate innovator_vl_stable || true
fi

LOAD=$1
SAVE=$2
TP=$3
PP=$4

mkdir -p ./tmp2/
SAVE_LANGUAGE_MODEL=./tmp2/language-mcore
SAVE_VISION_MODEL=./tmp2/vision-model-mcore
SAVE_SIGLIP2=./tmp2/vision-model-siglip2
SAVE_DINOV3=./tmp2/vision-model-dinov3
SAVE_ADAPTER=./tmp2/adapter-mcore
SAVE_PATCH=./tmp2/patch-mcore


# llama: language expert
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --megatron_path $AIAK_MAGATRON_PATH \
    --save_platform=huggingface \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

# siglip2
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/siglip2.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_SIGLIP2 \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

# dinov3
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/dinov3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_DINOV3 \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/innovator_vl/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch
python $CONVERT_CHECKPOINT_PATH/custom/innovator_vl/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/innovator-vl-8b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/innovator_vl/merge_huggingface_hybrid.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --ricevit_path $SAVE_VISION_MODEL \
    --siglip2_path $SAVE_SIGLIP2 \
    --dinov3_path $SAVE_DINOV3 \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE


# rm -rf $SAVE_LANGUAGE_MODEL
# rm -rf $SAVE_VISION_MODEL
# rm -rf $SAVE_SIGLIP2
# rm -rf $SAVE_DINOV3
# rm -rf $SAVE_ADAPTER
# rm -rf $SAVE_PATCH


# rsync -av --partial --exclude='*.safetensors' --exclude='*.index.json' /root/innovator_model_wenzichen/Innovator-VL-8B-Instruct/ /root/innovator_code_wenzichen/Innovator-VL/checkpoints/stage_2_instruct_llava_ov_8b_25M_1101_v1_iter_0105000_hf
