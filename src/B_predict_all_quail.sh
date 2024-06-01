#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi

mkdir -p ./logs

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models-context-minibatch/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1 quail models-context-minibatch
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in 1_group 2_group 3_group 7_group
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models-context-minibatch-dreamscape/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1 dreamscape models-context-minibatch-dreamscape
#                 fi
#             done
#         done
#     done
# done

for seed in 1 2 3 # "3b" "large" "small" "base"
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
        do
            for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm separate_models_t5_lm t5_wta_control_length t5_wta_control_length_and_init t5_wta_control_init_striped t5_wta_control_init_start
            do
                for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" # "1e-6" 
                do
                    file=./models-large-$seed-control/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
                    if [ ! -f "$file" ]; then
                        echo $file
                        sbatch --time=0:20:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486,cdr2591 \
                            --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p "$1" quail large $seed
                    fi
                done
            done
            # for architecture in soft_skill_attempt
            # do
            #     for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6" 
            #     do
            #         file=./models-large-$seed-v2/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
            #         if [ ! -f "$file" ]; then
            #             echo $file
            #             # sbatch --time=3:00:00 --mem-per-cpu=16G --mincpus=2 --account def-afyshe-ab \
            #             #     --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486 \
            #             #     --gres=gpu:v100l:1 B_run_prediction_group.sh $architecture $learning_rate $top_p "$1" quail $size
            #             # break
            #             sbatch --time=0:30:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
            #                 --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486 \
            #                 --gpus=1 run_python.sh ./B_predict_attempt.py $architecture $learning_rate $question_type $top_p "$1" quail large $seed
            #         fi
            #     done
            # done
            for architecture in soft_attempt
            do
                for learning_rate in "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1"
                do
                    file=./models-large-$seed-control/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
                    if [ ! -f "$file" ]; then
                        echo $file
                        sbatch --time=0:40:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486 \
                            --gpus=1 run_python.sh ./B_predict_attempt.py $architecture $learning_rate $question_type $top_p "$1" quail large $seed
                    fi
                done
            done
        done
    done
done
# for top_p in "0.95" "0.9" "0.85" "0.8" "0.75" "0.7" "0.65" "0.6" "0.55" "0.5" "0.45" "0.4" "0.35" "0.3" "0.25" "0.2" "0.15" "0.1"
# do
#     for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614,cdr2486 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1

#                     # -d afterok:6815515:6815516:6815571:6815572
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in 1_group 2_group 3_group 7_group
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models-dreamscape/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:20:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614,cdr2486 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1 dreamscape
#                     # sleep 30
#                     # -d afterok:6815515:6815516:6815571:6815572
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./models/$architecture-pt-$pretrain_learning_rate/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             --exclude=cdr2614 \
#                             --gpus=1 run_python.sh ./B_predict.py $architecture-pt-$pretrain_learning_rate $learning_rate $question_type $top_p $1
#                     fi
#                 done
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models-answer/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1 quail models-answer
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./models-answer/$architecture-pt-$pretrain_learning_rate/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             --exclude=cdr2614 \
#                             --gpus=1 run_python.sh ./B_predict.py $architecture-pt-$pretrain_learning_rate $learning_rate $question_type $top_p $1 quail models-answer
#                     fi
#                 done
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in 1_group 2_group 3_group 7_group
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./models-answer-dreamscape/$architecture/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         --exclude=cdr2614 \
#                         --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p $1 dreamscape models-answer-dreamscape
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for question_type in 1_group 2_group 3_group 7_group
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./models-answer-dreamscape/$architecture-pt-$pretrain_learning_rate/$learning_rate/$question_type/"predictions_nucleus_$1_$top_p.json"
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:10:00 --mem-per-cpu=6G --mincpus=4 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             --exclude=cdr2614 \
#                             --gpus=1 run_python.sh ./B_predict.py $architecture-pt-$pretrain_learning_rate $learning_rate $question_type $top_p $1 dreamscape models-answer-dreamscape
#                     fi
#                 done
#             done
#         done
#     done
# done