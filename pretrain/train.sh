job_dir="pretrain"

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

if [ ! -e $job_dir/config ]; then
    mkdir $job_dir/config
fi

vocab_size=21128
new_vocab_size=$(($vocab_size+26))
echo 'setting config/vocab.txt and config/model_config.json'
if [ ! -e $job_dir/config/vocab.txt ]; then
    wget -c -O $job_dir/config/vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
    # append new characters to the vocabulary
    for letter in {A..Z} ; do
        echo $letter >> $job_dir/config/vocab.txt
    done
fi

if [ ! -e $job_dir/config/model_config.json ]; then
    cp config/model_config.json $job_dir/config/model_config.json
    perl -pi -e 's/'$vocab_size'/'$new_vocab_size'/g' $job_dir/config/model_config.json
fi

raw_data_path=$job_dir/rawdata/sogou_utf8_title_content.txt
tokenizer_path=$job_dir/config/vocab.txt
tokenized_data_path=$job_dir/tokenized/
divide_path=$job_dir/divide/
model_config=$job_dir/config/model_config.json
epochs=30
batch_size=4
stride=1024
log_step=20
output_dir=$job_dir/model/
num_pieces=500

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
        --raw
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
        --device 0,1
fi
