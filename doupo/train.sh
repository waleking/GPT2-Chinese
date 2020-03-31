job_dir="doupo"

cd ..

if [ ! -e $job_dir/rawdata ]; then
    mkdir $job_dir/rawdata
fi

if [ ! -e $job_dir/divide ]; then
    mkdir $job_dir/divide
fi

if [ ! -e $job_dir/tokenized ]; then
    mkdir $job_dir/tokenized
fi

if [ ! -e $job_dir/model ]; then
    mkdir $job_dir/model
fi

if [ ! -e $job_dir/rawdata/train.txt ]; then
    wget -c -O $job_dir/rawdata/train.txt https://github.com/GaoPeng97/transformer-xl-chinese/blob/master/data/doupo/train.txt?raw=true
fi

if [ ! -e $job_dir/vocab.txt ]; then
    wget -c -O $job_dir/vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
fi

if [ ! -e $job_dir/tokenized/tokenized_train_0.txt ]; then
    # tokenization then run the training
    python train_single.py \
        --raw_data_path $job_dir/rawdata/train.txt \
        --tokenizer_path $job_dir/vocab.txt \
        --tokenized_data_path $job_dir/tokenized/ \
        --divide_path $job_dir/divide/ \
        --model_config config/model_config.json \
        --epochs 30 \
        --batch_size 4 \
        --log_step 20 \
        --output_dir $job_dir/model/ \
        --stride 1024 \
        --num_pieces 1 \
        --raw
else
    # run the training on the tokenized files
    python train_single.py \
        --raw_data_path $job_dir/rawdata/train.txt \
        --tokenizer_path $job_dir/vocab.txt \
        --tokenized_data_path $job_dir/tokenized/ \
        --divide_path $job_dir/divide/ \
        --model_config config/model_config.json \
        --epochs 30 \
        --batch_size 8 \
        --stride 1024 \
        --log_step 20 \
        --output_dir $job_dir/model/ \
        --num_pieces 1 \
        --device 0,1
fi
