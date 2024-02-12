#! /usr/bin/bash
# nproc_per_node : num of gpu used
# divid the batchsize equally on to each gpu

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=9 \
relpose/trainer_ddp.py --dataset=finetune \
                       --batch_size=45 \
                       --num_images=8 \
                       --random_num_images=true \
                       --gpu_ids=0,1,2,3,4,5,6,7,8 \
                       --lr=1e-5 \
                       --num_iterations=800000 \
                       --output_dir=ckpts_retrain \
                       --use_tf32 \
                       --use_amp \
                       --resume=ckpts_retrain/1224_1536_LR1e-05_N8_RandomNTrue_B144_AMP_TROURS_DDP

# torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 \
# relpose/trainer_ddp.py --dataset=finetune \
                     #   --batch_size=1 \
                     #   --num_images=8 \
                     #   --random_num_images=true \
                     #   --gpu_ids=0 \
                     #   --lr=1e-5 \
                     #   --num_iterations=800000 \
                     #   --output_dir=ckpts_retrain \
                     #   --use_tf32 \
                     #   --use_amp
                    #    --resume=ckpts_finetune/1026_0506_LR1e-05_N8_RandomNTrue_B36_Pretrainedckpt_back_AMP_TROURS_DDP \
