import argparse
import numpy as np
import torch
import deepspeed 

def parse_ds_args(parser):
    parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="local rank passed from distributed launcher",
        )
    
    parser.add_argument(
            "--log-interval",
            type=int,
            default=2000,
            help="output logging information at a given interval",
        )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu ": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr":1e-5,
                "weight_decay": 1e-6,
            },
        },
        # "scheduler": {
        #     "type": "WarmupLR",
        #     "params": {
        #         "warmup_min_lr": 0,
        #         "warmup_max_lr": 0.001,
        #         "warmup_num_steps": 100,
        #     },
        # },
        #"gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config
