#!/bin/bash

COMMON_ARGS=(
    # --- Preprocessing ---
    --input_path /workdir/allairec/ttbar_50k
    --input_format h5
    --input_tensor_path /scratch/allairec-806408
    --tensor_format h5
    --dataset_name seeding_data
    --events_per_file 100
    --max_events 50000
    --orphan_hit_fraction 0.0
    --binning_strategy neighbor
    --bin_width 0.05
    --binning_margin 0.05
    --max_hit_input 1200
    --eta_range -3.0 3.0
    --vertex_cuts 10.0 200.0
    --hit_range 500.0 1000.0
    --hit_features x y z
    --particle_features eta phi pT
    --pv_pair_weight 10
    --z0_weight_bin 0
    --num_workers 10
    # --- Transformer architecture ---
    --nb_layers_t 5
    --nb_heads 4
    --dim_embedding 256
    --feed_forward_ratio 2
    --dropout 0.1
    --embedding_feature 0 1 2 3
    --high_level_features 0 1 2 3 4 5
    --cosine_processing 4
    --fourier_num_frequencies 25 25 25 25
    --dim_max 500.0 500.0 1000.0 500.0
    --shift 0.0 0.0 0.0 0.0
    --dtype float32
    # --regression
    # --- Training ---
    --epoch_nb 2
    --num_warmup_steps 5
    --num_training_steps 50
    --min_lr_ratio 0.01
    --val_fraction 0.05
    --test_fraction 0.05
    --learning_rate 5e-05
    --weight_decay 0.01
    --batch_size 25
    --device cuda:0
    # --recompute_tensor
    --model_path transformer.pt
    --loss_components attention_next
    --loss_weights 1.0
    --reconstruction_method beam_search
    # --no_test
    --resume_training
    --timing_enabled
)

# Usage: python -m GUNTAM.Seed.Train "${COMMON_ARGS[@]}"

