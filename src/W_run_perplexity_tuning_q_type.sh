#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi
mkdir -p perplexity_loss_results_seed_q_type

# for size in "11b" # "3b" "large" "base" "small"
for seed in 1 2 3 # "3b" "large" "small" "base"
do
    for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
    do
        for learning_rate in "1e-2" "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1"
        do
            for architecture in soft_attempt_t5_lm # single_model soft_attempt # separate_models t5_wta_patch single_model_soft_prompt_patch ghanem_et_al_2022_true single_model_with_token_random_token_init single_model # separate_models # single_model_soft_prompt_patch
            do
                file="./perplexity_loss_results_seed_q_type/"$architecture"_large_silver_squad_"$1"_"$learning_rate"_"$seed"_"$question_type".json"
                if [ ! -f "$file" ]; then
                    sbatch --time=1:00:00 --mem-per-cpu=12000M --mincpus=1 \
                    --account def-afyshe-ab --gpus=1 run_python.sh \
                    ./W_perplexity_tuning_q_type.py $architecture $learning_rate "$1" silver_squad large $seed $question_type
                fi
            done
        done
    done
done
# -d afterok:6815515:6815516:6815571:6815572