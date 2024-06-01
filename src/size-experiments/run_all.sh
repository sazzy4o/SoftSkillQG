#!/bin/bash
for seed in 1 2 3
do
sbatch --time=72:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py None $seed
sbatch --time=72:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 50000 $seed
sbatch --time=48:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 20000 $seed
sbatch --time=48:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 10000 $seed
sbatch --time=12:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 5000 $seed
sbatch --time=12:00:00 --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 2000 $seed
sbatch --time=6:00:00  --mem-per-cpu=52000M --mincpus=1 --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh train_squad_qgen.py 1000 $seed
done