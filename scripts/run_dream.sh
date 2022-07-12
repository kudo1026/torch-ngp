#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_flowers --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "bouquet of flowers sitting in a clear glass vase" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_flowers --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "bouquet of flowers sitting in a clear glass vase" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_vase --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a small green vase displays some small yellow blooms" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_vase --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a small green vase displays some small yellow blooms" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_slug --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a slug crawling on the ground around flower petals" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_slug --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a slug crawling on the ground around flower petals" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_armchair --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "armchair in the shape of an avocado" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_armchair --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "armchair in the shape of an avocado" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_bird --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a bird that has many colors on it" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_bird --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a bird that has many colors on it" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_hash_hw_224_tr_flowers --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a blue jug in a garden filled with mud" --lr 1e-3  --h 224 --w 224 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network hash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_dream.py --workspace trial_dream_mlp_hw_128_tr_flowers --bound 1.0 --scale 0.67 --dt_gamma 0 --rand_pose 0 --clip_text "a blue jug in a garden filled with mud" --lr 1e-3  --h 128 --w 128 --cuda_ray --fp16 --batch_size 4 --iters 10000 --network mlp
