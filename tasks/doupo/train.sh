job_dir="tasks/doupo"

cd ../..

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

if [ ! -e $job_dir/rawdata/train.txt ]; then
    wget -c -O $job_dir/rawdata/train_raw.txt https://github.com/GaoPeng97/transformer-xl-chinese/blob/master/data/doupo/train.txt?raw=true
    # remove empty lines
    python $job_dir/format_raw_txt.py $job_dir/rawdata/train_raw.txt $job_dir/rawdata/train.txt 
fi

vocab_size=21128
declare -a additional_chars=("“" "”" "…" "’" "‘" "—" " " "\t" "\`")
new_vocab_size=$(($vocab_size+${#additional_chars[@]}+26))
echo 'setting config/vocab.txt and config/model_config.json'
if [ ! -e $job_dir/config/vocab.txt ]; then
    wget -c -O $job_dir/config/vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
    # append new characters to the vocabulary
    for letter in "${additional_chars[@]}"; do
        echo -e "$letter" >> $job_dir/config/vocab.txt
    done
    for letter in {A..Z} ; do
        echo $letter >> $job_dir/config/vocab.txt
    done
fi

stride=256
n_layers=10
n_ctx=512
if [ ! -e $job_dir/config/model_config.json ]; then
    cp config/model_config.json $job_dir/config/model_config.json
    # change vocabulary size
    perl -pi -e 's/'$vocab_size'/'$new_vocab_size'/g' $job_dir/config/model_config.json
    # change the number of layers from 12 to $n_layers
    perl -pi -e 's/"n_layer": 12/"n_layer": '$n_layers'/g' $job_dir/config/model_config.json
    # change the model input length from 1024 to $n_ctx
    perl -pi -e 's/"n_ctx": 1024/"n_ctx": '$n_ctx'/g' $job_dir/config/model_config.json
    perl -pi -e 's/"n_positions": 1024/"n_positions": '$n_ctx'/g' $job_dir/config/model_config.json
fi

raw_data_path=$job_dir/rawdata/train.txt
tokenizer_path=$job_dir/config/vocab.txt
tokenized_data_path=$job_dir/tokenized/
divide_path=$job_dir/divide/
model_config=$job_dir/config/model_config.json
epochs=100
batch_size=32
log_step=100
output_dir=$job_dir/model/
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
