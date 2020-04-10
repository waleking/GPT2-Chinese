job_dir="dataaugmentation"

cd ..

if [ ! -e $job_dir/outputs_finetuned ]; then
    mkdir $job_dir/outputs_finetuned
fi
tokenizer_path=$job_dir/pretrain_model/vocab.txt
model_path=$job_dir/model_finetuned/final_model/
model_config=$model_path/model_config.json
sample_path=$job_dir/outputs_finetuned/

python generate.py \
    --device 0 \
    --model_path $model_path \
    --model_config $model_config \
    --tokenizer_path $tokenizer_path \
    --temperature 0.8 \
    --prefix 月亮 \
    --length 50 \
    --topk 50 \
    --nsamples 50 \
    --save_samples \
    --save_samples_path $sample_path
