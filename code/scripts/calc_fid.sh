# configs of different datasets
cfg=$1
batch_size_per_gpu=64
rewriteFID_model_path=./saved_models/bird/
# DDP settings
multi_gpus=True
master_port=11122

# You can set CUDA_VISIBLE_DEVICES=0,1,2..., node=number_of_GPUs to accelerate the evaluation process if you have multiple GPUs
nodes=1
#python -m torch.distributed.launch --nproc_per_node=$nodes --master_port $master_port
CUDA_VISIBLE_DEVICES=0 python src/test_FID.py \
                    --cfg $cfg \
                    --batch_size $batch_size_per_gpu \
                    --rewriteFID_model_path $rewriteFID_model_path \
                    #--multi_gpus $multi_gpus \