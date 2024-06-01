#!/bin/bash
for seed in 1 2 3
do
    for learning_rate in "5e-6" "1e-5" "5e-5" "1e-4" "5e-4" #"1e-6" 
    do
        echo '' > /dev/null
        for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm t5_wta_control_length t5_wta_control_length_and_init t5_wta_control_init_striped t5_wta_control_init_start # single_model_with_hard_prompt_t5_lm # t5_wta_control_init_striped_t5_lm t5_wta_control_init_start_t5_lm # ghanem_et_al_2022_t5_lm # control_t5_lm # t5_wta_control_init_start # t5_wta_control_init_striped # t5_wta_control_length t5_wta_control_length_and_init ghanem_et_al_2022_true # t5_wta_control_init_striped
        do
            file=./models-large-$seed-control/$architecture/$learning_rate/model_files/config.json
            if [ ! -f "$file" ]; then
                echo $file
                sbatch --time=72:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                ./A_train_question_generator_control.py $architecture $learning_rate quail large $seed
            fi
        done
    done

    for learning_rate in "5e-6" "1e-5" "5e-5" "1e-4" "5e-4" # "1e-6"
    do
        for architecture in separate_models_t5_lm 
        do
            for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
            do
                file=./models-large-$seed-control/$architecture/$learning_rate/$question_type/"model_files/config.json"
                if [ ! -f "$file" ]; then
                    sbatch --time=12:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                    --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                    ./A_train_question_generator_control.py $architecture $learning_rate quail large $seed $question_type
                fi
            done
        done
    done

    for learning_rate in "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1"
    do
        for architecture in soft_attempt
        do
            for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
            do
                file=./models-large-$seed-control/$architecture/$learning_rate/$question_type/"prefix_embeddings.pt"
                if [ ! -f "$file" ]; then
                    sbatch --time=12:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                    --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                    ./A_train_question_generator_attempt.py $architecture $learning_rate quail large $seed $question_type
                fi
            done
        done
    done
done