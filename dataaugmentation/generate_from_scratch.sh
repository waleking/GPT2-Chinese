job_dir="doupo"

cd ..

if [ ! -e $job_dir/outputs_from_scratch ]; then
    mkdir $job_dir/outputs_from_scratch
fi

tokenizer_path=$job_dir/config/vocab.txt
model_path=$job_dir/model/final_model/
model_config=$job_dir/model/final_model/model_config.json
sample_path=$job_dir/outputs/

python generate.py \
    --device 0 \
    --model_path $model_path \
    --model_config $model_config \
    --tokenizer_path $tokenizer_path \
    --temperature 0.8 \
    --length 50 \
    --topk 50 \
    --nsamples 50 \
    --save_samples \
    --save_samples_path $sample_path
