#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi

mkdir -p ./faithfulness_results

# for size in "large" # "3b" "base" "small" # "3b" 
for seed in 1 2 3 # "3b" "large" "small" "base"
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
        do
            for architecture in single_model_with_token_random_token_init ghanem_et_al_2022_true # single_model_with_token_random_token_init_with_sep # ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
            do
                file=./faithfulness_results_seed/$architecture-$learning_rate-$top_p-$1-large-$seed.json
                if [ ! -f "$file" ]; then
                    echo $file
                    sbatch --time=0:15:00 --mem-per-cpu=8000M --mincpus=1 --account def-afyshe-ab \
                        --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486 \
                        --gpus=1 run_python.sh ./H_faith_eval.py $architecture $learning_rate $top_p "$1" large $seed
                    # -d afterok:6815515:6815516:6815571:6815572
                fi
            done
        done
    done
done