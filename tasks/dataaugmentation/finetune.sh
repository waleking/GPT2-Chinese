job_dir="dataaugmentation"

cd ..

if [ ! -e $job_dir/rawdata ]; then
    mkdir $job_dir/rawdata
fi

if [ ! -e $job_dir/model_finetuned ]; then
    mkdir $job_dir/model_finetuned
fi

if [ ! -e $job_dir/pretrained_model ]; then
    mkdir $job_dir/pretrained_model
    wget -c -O $job_dir/pretrained_model/pytorch_model.bin https://www.dropbox.com/s/yqxuu6fszqto4od/pytorch_model.bin?dl=0
    wget -c -O $job_dir/pretrained_model/config.json https://www.dropbox.com/s/7z599ixdzyghkth/config.json?dl=0 
    wget -c -O $job_dir/pretrained_model/vocab.txt https://www.dropbox.com/s/9obd0qadtst347l/vocab.txt?dl=0 
fi


if [ ! -e $job_dir/rawdata/train.txt ]; then
    echo "downloading data ..."
    python $job_dir/download.py $job_dir/rawdata/
    python $job_dir/format_raw_txt.py $job_dir/rawdata/
    echo "data is downloaded at "$job_dir/rawdata/train.txt
fi

pretrained_model=$job_dir/pretrained_model
raw_data_path=$job_dir/rawdata/train.txt
tokenizer_path=$job_dir/pretrained_model/vocab.txt
model_config=$job_dir/pretrained_model/config.json
epochs=80
batch_size=8
stride=1024
log_step=1
output_dir=$job_dir/model_finetuned/
num_pieces=1

python train_on_small_file.py \
    --pretrained_model $pretrained_model \
    --raw_data_path $raw_data_path \
    --tokenizer_path $tokenizer_path \
    --model_config $model_config \
    --epochs $epochs \
    --batch_size $batch_size \
    --stride $stride \
    --log_step $log_step \
    --output_dir $output_dir \
    --num_pieces $num_pieces \
    --device 0,1 \
    --ignore_intermediate_epoch_model
