CUDA_VISIBLE_DEVICES=0
model_size=$1
data_size=1000000
data_num=1000

current_time=$(date "+%Y%m%d%H%M%S")

llama_model=Enoch/llama-${model_size}b-hf
lora_model_path=Enoch/SoTana-${model_size}B-lora-${data_size}


# Inference with llama-lora
function fine-tuned-inference(){

output_dir=output/Sotana-{model_size}B/${current_time}
mkdir -p ${output_dir}
echo ${output_dir}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3.9 inference.py \
--data_num ${data_num} \
--lora_model_path ${lora_model_path} \
--llama_model ${llama_model} \
--output_dir ${output_dir} 2>&1 |tee ${output_dir}/inference.log
}


fine-tuned-inference
