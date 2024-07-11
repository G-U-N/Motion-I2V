srun -x SH-IDC1-10-142-4-24 --cpus-per-task=5 --ntasks-per-node=1 --ntasks=1 -p ISPCodec --gres=gpu:1 python -u inference.py --visualization_modes overlay --video_path cartwheel.mp4
