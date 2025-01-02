#!/bin/bash

index=$1

sparsity_weights=(0.01 0.05)
reg_weights=(0.5)
cls_weights=(0.5 1.0)
focal_alphas=(1)
focal_gammas=(1)


NUM_SPARSE=${#sparsity_weights[@]}
NUM_REG=${#reg_weights[@]}
NUM_CLS=${#cls_weights[@]}
NUM_ALPHA=${#focal_alphas[@]}
NUM_GAMMA=${#focal_gammas[@]}


TOTAL_COMBINATIONS=$((NUM_SPARSE * NUM_REG * NUM_CLS * NUM_ALPHA * NUM_GAMMA))



sparsity_idx=$((index / (NUM_REG * NUM_CLS * NUM_ALPHA * NUM_GAMMA) % NUM_SPARSE))
reg_idx=$((index / (NUM_CLS * NUM_ALPHA * NUM_GAMMA) % NUM_REG))
cls_idx=$((index / (NUM_ALPHA * NUM_GAMMA) % NUM_CLS))
alpha_idx=$((index / NUM_GAMMA % NUM_ALPHA))
gamma_idx=$((index % NUM_GAMMA))

# Assign parameters to variables
sparsity_weight=${sparsity_weights[$sparsity_idx]}
reg_weight=${reg_weights[$reg_idx]}
cls_weight=${cls_weights[$cls_idx]}
focal_alpha=${focal_alphas[$alpha_idx]}
focal_gamma=${focal_gammas[$gamma_idx]}



python /scratch/zc1592/small_data/main.py \
    --n_unfrozen 12 \
    --loss_type focal \
    --sparsity_weight $sparsity_weight \
    --reg_weight $reg_weight \
    --cls_weight $cls_weight \
    --focal_alpha $focal_alpha \
    --focal_gamma $focal_gamma
 
