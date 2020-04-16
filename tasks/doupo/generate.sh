job_dir="tasks/doupo"

cd ../..

if [ ! -e $job_dir/outputs ]; then
    mkdir $job_dir/outputs
fi

python generate.py \
    --device 0 \
    --model_path $job_dir/model/final_model/ \
    --model_config $job_dir/model/final_model/model_config.json \
    --tokenizer_path $job_dir/config/vocab.txt \
    --temperature 0.8 \
    --prefix [SEP][CLS]萧炎 \
    --length 100 \
    --topk 50 \
    --nsamples 10 \
    --save_samples \
    --save_samples_path $job_dir/outputs/
