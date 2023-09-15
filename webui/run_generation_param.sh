model_size=$1
data_size=$2

python generate.py \
    --load_8bit \
    --base_model Enoch/llama-${model_size}b-hf \
    --lora_weights Enoch/SoTana-${model_size}B-lora-${data_size} \
    --share_gradio \