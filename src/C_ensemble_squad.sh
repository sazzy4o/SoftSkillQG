#!/bin/bash
dataset='dev'

mkdir -p results_all_metrics_seed_squad

for seed in 1 # 2 3
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for metric in meteor # fbd rouge_l bleu_4 bleu_3 bleu_2 bleu_1
        do
            for take in None 50000 20000 10000 5000 2000 1000 
            do
                for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
                do
                    for architecture in single_model_with_token_random_token_init ghanem_et_al_2022_true
                    do
                        file=./results_all_metrics_seed_squad/ensemble_"$metric"_"$architecture"_probs_"$top_p"_none_nan_"$learning_rate"_"$dataset"_large_"$take"_"$seed".json
                        if [ ! -f $file ]; then
                            echo $file
                            sbatch --time=0:15:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab \
                                --output ./logs/slurm-%j.out \
                                run_python.sh ./C_squad_ensemble.py $architecture $learning_rate $top_p $metric $dataset squad $take $seed
                        fi
                    done
                done
            done
        done
    done
done
