

data_size=100000
batch_size=512
target_modules='[q_proj,v_proj,v_proj,o_proj]' 

model_size=$1
epoch=$2

data_filename=../data-generation/output/100000/data.json  
micro_batch_size=4
learning_rate=0.0001
cutoff_len=512
llama_model=Enoch/llama-${model_size}b-hf
lora_r=8


function single_train() {
current_time=$(date "+%Y%m%d%H%M%S")
output_dir=output/llama-${model_size}B/${current_time}
mkdir -p ${output_dir}
echo ${output_dir}
CUDA_VISIBLE_DEVICES=0 python fine-tuning.py \
    --base_model ${llama_model} \
    --data_path ${data_filename} \
    --output_dir ${output_dir} \
    --batch_size ${batch_size}\
    --micro_batch_size ${micro_batch_size} \
    --num_epochs  ${epoch}  \
    --learning_rate ${learning_rate} \
    --cutoff_len ${cutoff_len} \
    --val_set_size 0 \
    --lora_r ${lora_r} \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ${target_modules} \
    --train_on_inputs \
    --wandb_project "sotana" 2>&1 |tee ${output_dir}/finetune.log
}

function single_train_debug() {
current_time=debug
output_dir=output/llama-${model_size}B/${current_time}
mkdir -p ${output_dir}
echo ${output_dir}
CUDA_VISIBLE_DEVICES=0 python fine-tuning.py \
    --do_debug \
    --base_model ${llama_model} \
    --data_path ${data_filename} \
    --output_dir ${output_dir} \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs  2  \
    --learning_rate ${learning_rate} \
    --cutoff_len 256 \
    --val_set_size 0 \
    --lora_r ${lora_r} \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ${target_modules} \
    --train_on_inputs \
    --wandb_project "sotana-test" 2>&1 |tee ${output_dir}/finetune.log
}

single_train

# single_train_debug


