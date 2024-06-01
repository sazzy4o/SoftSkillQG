#!/bin/bash
# find ./models/squad-2000 -name epoch* | xargs -I% sbatch --time=2:00:00 --mem-per-cpu=100000M --mincpus=1 --exclude=ng10502 --account def-afyshe-ab --gres=gpu:a100:2 run_python.sh predict_xxl.py %

for model in `find ./models -name epoch*`
do
if [ ! -f $model/f1.json ]; then
    sbatch --time=1:40:00 --mem-per-cpu=100000M --mincpus=1 --exclude=ng10502 --account def-afyshe-ab --gres=gpu:a100:2 run_python.sh predict_xxl.py $model
fi
done