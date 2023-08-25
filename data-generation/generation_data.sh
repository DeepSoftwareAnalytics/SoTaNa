
data_number=100000
time=$(date "+%Y%m%d%H%M%S")
output_dir=./output/${data_number}/${time}
echo ${output_dir}
mkdir -p ${output_dir}

python -m generate_data \
generate_instruction_following_data \
--output_dir ${output_dir} \
--num_instructions_to_generate ${data_number} 2>&1 | tee ${output_dir}/generate_data.log

