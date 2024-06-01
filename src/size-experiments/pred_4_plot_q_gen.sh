#!/bin/bash
# find ./models/squad-2000 -name epoch* | xargs -I% sbatch --time=2:00:00 --mem-per-cpu=100000M --mincpus=1 --exclude=ng10502 --account def-afyshe-ab --gres=gpu:a100:2 run_python.sh predict_xxl.py %

for model in `find ./models/squad-q-gen* -name epoch*`
do
if [ ! -f $model/multi_meteor.json ]; then
    sbatch --time=30:00 --mem-per-cpu=20000M --mincpus=1 --exclude=ng10502 --account def-afyshe-ab --gres=gpu:a100:2 run_python.sh predict_q_gen.py $model
fi
done