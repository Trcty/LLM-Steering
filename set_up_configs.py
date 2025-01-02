import argparse

def set_up_initial_exp_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='bert-base-uncased', 
        help="Model to use for training"
    )
    parser.add_argument(
        '--n_unfrozen', 
        type=int, 
        default=12, 
        help="Number of layers to unfreeze"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0005, 
        help="initial learning rate"
    )
    parser.add_argument(
        '--l1_lambda', 
        type=float, 
        default=0.001, 
        help="sparsity penalty"
    )
    parser.add_argument(
        '--n_features', 
        type=int, 
        default=768, 
        help="Number of features considered."
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help="Batch size."
    )
    parser.add_argument(
        '--mse_weight', 
        type=int, 
        default=10, 
        help="scale up mse loss in msewithl1loss"
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='train'
    )
    parser.add_argument(
        '--ckpt_path', 
        type=str, 
        default='/scratch/zc1592/small_data/experiments/focal/bert-base-uncased_0.01_0.5_0.5_1.0_1/model_epoch_28.pt'
    )
    parser.add_argument(
        "--loss_type", 
        type=str, 
        default='focal',
        help="loss function to use (focal, mseWithL1Loss, maskedMseWithL1Loss)"
    )
    parser.add_argument(
        '--sparsity_weight', 
        type=float, 
        default=0.001, 
        help="sparsity penalty"
    )
    parser.add_argument(
        '--reg_weight', 
        type=float, 
        default=1.0, 
        help="regression loss weight"
    )
    parser.add_argument(
        '--cls_weight', 
        type=float, 
        default=1.0, 
        help="classification loss weight"
    )
    parser.add_argument(
        '--focal_alpha', 
        type=float, 
        default=0.9, 
    )
    parser.add_argument(
        '--focal_gamma', 
        type=int, 
        default=2, 
    )
    parser.add_argument(
        '--huber_delta', 
        type=float, 
        default=0.1, 
    )


    configs = parser.parse_args()
    return configs


def set_up_end2end_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='bert-base-uncased', 
        help="Model to use for training"
    )

    parser.add_argument(
        '--sae_model', 
        type=str, 
        default='gemma-2b', 
    )
    parser.add_argument(
        '--release', 
        type=str, 
        default='gemma-2b-res-jb', 
    )
    parser.add_argument(
        '--hook_id', 
        type=str, 
        default='blocks.0.hook_resid_post', 
    )
    parser.add_argument(
        '--n_unfrozen', 
        type=int, 
        default=12, 
        help="Number of layers to unfreeze"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001, 
        help="initial learning rate"
    )
    parser.add_argument(
        '--n_features', 
        type=int, 
        default=768, 
        help="Number of features considered."
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help="Batch size."
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='train'
    )
    parser.add_argument(
        '--ckpt_path', 
        type=str, 
        default=None
    )
    parser.add_argument(
        '--tmp', 
        type=float, 
        default=0.9, 
    )
    parser.add_argument(
        '--prompt_len', 
        type=int, 
        default=5, 
    )

    parser.add_argument(
        '--max_new_token', 
        type=int, 
        default=50, 
    )
    parser.add_argument(
        '--loss', 
        type=str, 
        default='fasg', 
    )


    configs = parser.parse_args()
    return configs
