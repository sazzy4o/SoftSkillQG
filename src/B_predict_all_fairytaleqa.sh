#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi

mkdir -p ./logs

for seed in 1 2 3
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for question_type in  "character" "feeling" "action" "setting" "prediction" "outcome_resolution" "causal_relationship"
        do
            question_type_space=`echo $question_type| tr "_" " "`
            for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm separate_models_t5_lm t5_wta_control_init_striped
            do
                for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" # "1e-6" 
                do
                    file=./models-large-$seed-control-fairytaleqa/$architecture/$learning_rate/$question_type_space/"predictions_nucleus_$1_$top_p.json"
                    if [ ! -f "$file" ]; then
                        echo $file
                        sbatch --time=0:30:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486,cdr2591 \
                            --gpus=1 run_python.sh ./B_predict.py $architecture $learning_rate $question_type $top_p "$1" fairytaleqa large $seed
                    fi
                done
            done
            for architecture in soft_attempt
            do
                for learning_rate in "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1"
                do
                    file=./models-large-$seed-control-fairytaleqa/$architecture/$learning_rate/$question_type_space/"predictions_nucleus_$1_$top_p.json"
                    if [ ! -f "$file" ]; then
                        echo $file
                        sbatch --time=0:30:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out --exclude=cdr2614,cdr2486 \
                            --gpus=1 run_python.sh ./B_predict_attempt.py $architecture $learning_rate $question_type $top_p "$1" fairytaleqa large $seed
                    fi
                done
            done
        done
    done
done
