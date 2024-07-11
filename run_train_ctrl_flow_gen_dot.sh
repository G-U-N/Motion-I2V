python -u  -m torch.distributed.launch --nproc_per_node=8  train_ctrl_flow_gen_dot.py --config configs/configs_flowgen/training/ctrl_training.yaml --task=ctrl_test
