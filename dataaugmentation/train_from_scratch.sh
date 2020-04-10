job_dir="dataaugmentation"

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

if [ ! -e $job_dir/model_without_fintuning ]; then
    mkdir $job_dir/model_without_fintuning
fi

if [ ! -e $job_dir/config ]; then
    mkdir $job_dir/config
fi

if [ ! -e $job_dir/rawdata/train.txt ]; then
    echo "downloading data ..."
    python $job_dir/download.py $job_dir/rawdata/
    python $job_dir/format_raw_txt.py $job_dir/rawdata/
    echo "data is downloaded at "$job_dir/rawdata/train.txt
fi

echo 'setting config/vocab.txt and config/model_config.json'
if [ ! -e $job_dir/config/vocab.txt ]; then
    wget -c -O $job_dir/config/vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
fi

if [ ! -e $job_dir/config/model_config.json ]; then
    cp config/model_config.json $job_dir/config/model_config.json
    # change the number of layers from 12 to 10
    perl -pi -e 's/"n_layer": 12/"n_layer": 10/g' $job_dir/config/model_config.json
fi

raw_data_path=$job_dir/rawdata/train.txt
tokenizer_path=$job_dir/config/vocab.txt
tokenized_data_path=$job_dir/tokenized/
divide_path=$job_dir/divide/
model_config=$job_dir/config/model_config.json
epochs=300
batch_size=8
stride=1024
log_step=1
output_dir=$job_dir/model_from_scratch/
num_pieces=1

if [ ! -e $job_dir/tokenized/tokenized_train_0.txt ]; then
    # tokenization then run the training
    python train_single.py \
        --raw_data_path $raw_data_path \
        --tokenizer_path $tokenizer_path \
        --tokenized_data_path $tokenized_data_path \
        --divide_path $divide_path \
        --model_config $model_config \
        --epochs $epochs \
        --batch_size $batch_size \
        --stride $stride \
        --log_step $log_step \
        --output_dir $output_dir \
        --num_pieces $num_pieces \
        --raw \
        --ignore_intermediate_epoch_model
else
    # run the training on the tokenized files
    python train_single.py \
        --raw_data_path $raw_data_path \
        --tokenizer_path $tokenizer_path \
        --tokenized_data_path $tokenized_data_path \
        --divide_path $divide_path \
        --model_config $model_config \
        --epochs $epochs \
        --batch_size $batch_size \
        --stride $stride \
        --log_step $log_step \
        --output_dir $output_dir \
        --num_pieces $num_pieces \
        --device 0,1 \
        --ignore_intermediate_epoch_model
fi
