srun --cpus-per-task=5 --ntasks=1 --ntasks-per-node=1 -p basemodel --gres=gpu:1 python setup.py install --user
