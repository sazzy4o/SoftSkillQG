#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi
mkdir -p perplexity_loss_results_seed

# for size in "11b" # "3b" "large" "base" "small"
for seed in 1 2 3 # "3b" "large" "small" "base"
do
    # "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1" #
    for learning_rate in  "5e-6" "1e-5" "5e-5" "1e-4" "5e-4" # "1e-6" 
    do
        for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm separate_models_t5_lm t5_wta_control_length t5_wta_control_length_and_init t5_wta_control_init_striped t5_wta_control_init_start # soft_attempt
        do
            file="./perplexity_loss_results_seed/"$architecture"_large_quail_"$1"_"$learning_rate"_"$seed".json"
            if [ ! -f "$file" ]; then
                sbatch --time=1:00:00 --mem-per-cpu=12000M --mincpus=1 \
                --account def-afyshe-ab --gpus=1 run_python.sh \
                ./W_perplexity_tuning.py $architecture $learning_rate "$1" quail large $seed
            fi
        done
    done
done
# -d afterok:6815515:6815516:6815571:6815572