#!/bin/bash
dataset="dev" # test set is not public for squad

mkdir -p ./logs

for seed in 1 # 2 3 # 
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for take in None 50000 20000 10000 5000 2000 1000 
        do
            for architecture in single_model_with_token_random_token_init ghanem_et_al_2022_true
            do
                for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" # "1e-6" 
                do
                    file=./models-squad-$seed/$architecture-$take/$learning_rate/Unset/"predictions_nucleus_"$dataset"_$top_p.json"
                    if [ ! -f "$file" ]; then
                        echo $file
                        sbatch --time=4:00:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486,cdr2591 \
                            --gpus=1 run_python.sh ./B_predict_squad.py $architecture $learning_rate $take $top_p "$dataset" squad large $seed
                    fi
                done
            done
        done
    done
done