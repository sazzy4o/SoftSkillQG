#!/bin/bash
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-10000/epoch_6
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-None/epoch_7
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-1000/epoch_9
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-50000/epoch_9
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-5000/epoch_9
sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-2000/epoch_9
# sbatch --time=1:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh predict.py ./models/squad-20000/epoch_4
