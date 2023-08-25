model_size=$1 
data_size=100000
CUDA_VISIBLE_DEVICES=0
model_type=SoTana
llama_model=Enoch/llama-${model_size}b-hf
lora_model_path=Enoch/SoTana-${model_size}B-lora-${data_size}
data_num=100



# Inference with llama-lora
function fine-tuned-inference(){
current_time=$(date "+%Y%m%d%H%M%S")
output_dir=output/${model_type}-${model_size}/${current_time}
mkdir -p ${output_dir}
echo ${output_dir}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python inference.py \
--data_num ${data_num} \
--lora_model_path ${lora_model_path} \
--llama_model ${llama_model} \
--output_dir ${output_dir} 2>&1 |tee ${output_dir}/inference.log
}


function fine-tuned-inference-debug(){
current_time=debug
output_dir=output/${model_type}-${model_size}/${current_time}
mkdir -p ${output_dir}
echo ${output_dir}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python inference.py \
--data_num 2 \
--lora_model_path ${lora_model_path} \
--llama_model ${llama_model} \
--output_dir ${output_dir} 2>&1 |tee ${output_dir}/inference.log
}


fine-tuned-inference
#  fine-tuned-inference-debug
