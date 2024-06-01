#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi

mkdir -p results_all_metrics_seed_fairytaleqa

for seed in 1 2 3 
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for metric in meteor # fbd rouge_l bleu_4 bleu_3 bleu_2 bleu_1
        do
            for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" # "1e-6"
            do
                for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm separate_models_t5_lm t5_wta_control_init_striped
                do
                    file=./results_all_metrics_seed_fairytaleqa/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1"_large_"$seed".json
                    if [ ! -f $file ]; then
                        echo $file
                        sbatch --time=0:10:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out \
                            run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric $1 fairytaleqa large $seed
                    fi
                done
            done
        for learning_rate in "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1" # "1e-2" 
        do
            for architecture in soft_attempt
                do
                    file=./results_all_metrics_seed_fairytaleqa/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1"_large_"$seed".json
                    if [ ! -f $file ]; then
                        echo $file
                        sbatch --time=0:10:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out \
                            run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric $1 fairytaleqa large $seed
                    fi
                done
            done
        done
    done
done