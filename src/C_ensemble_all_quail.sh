#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Need to specify dataset'
    exit 1
fi

mkdir -p results_all_metrics
mkdir -p results_all_metrics_seed
mkdir -p results_all_metrics_seed_control
# mkdir -p results_all_metrics_answer
# mkdir -p results_all_metrics_answer_dreamscape
# mkdir -p results_all_metrics_dups
# mkdir -p results_all_metrics_context_minibatch
# mkdir -p results_all_metrics_context_minibatch_dreamscape

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./results_all_metrics_context_minibatch_dreamscape/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 dreamscape models-context-minibatch-dreamscape results_all_metrics_context_minibatch_dreamscape
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./results_all_metrics_context_minibatch/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 quail models-context-minibatch results_all_metrics_context_minibatch
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./results_all_metrics_dups/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:10:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 quail models results_all_metrics_dups
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for d_cutoff in "0.60" "0.62" "0.64" "0.66" "0.68" "0.70" "0.72" "0.74" "0.76" "0.78" "0.80" "0.82" "0.84" "0.86" "0.88" "0.90" "0.92" "0.94" "0.96" "0.98" "1.00"
#     do
#         for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     for dot_cos in dot cos
#                     do
#                     file=./results_all_metrics/ensemble_"$metric"_"$architecture"_probs_"$top_p"_semantic_cluster_"$dot_cos"_"$d_cutoff"_"$learning_rate"_"$1".json
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:40:00 --mem-per-cpu=1000M --mincpus=4 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric semantic_cluster_$dot_cos $d_cutoff $1 quail models results_all_metrics
#                     fi
#                     done
#                 done
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./results_all_metrics/ensemble_"$metric"_"$architecture-pt-$pretrain_learning_rate"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:40:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             run_python.sh ./C_ensemble.py $architecture-pt-$pretrain_learning_rate $learning_rate $top_p $metric whitespace nan $1 quail
#                     fi
#                 done
#             done
#         done
#     done
# done

# for size in "11b" # "3b" "large" "base" "small"
for seed in 1 2 3 # "3b" "large" "small" "base"
do
    for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
    do
        for metric in meteor # fbd rouge_l bleu_4 bleu_3 bleu_2 bleu_1
        do
            for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" # "1e-6"
            do
                for architecture in single_model_t5_lm control_t5_lm ghanem_et_al_2022_t5_lm single_model_with_soft_prompt_t5_lm separate_models_t5_lm t5_wta_control_length t5_wta_control_length_and_init t5_wta_control_init_striped t5_wta_control_init_start
                do
                    file=./results_all_metrics_seed_control/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1"_large_"$seed".json
                    if [ ! -f $file ]; then
                        echo $file
                        sbatch --time=0:20:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out \
                            run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric $1 quail large $seed
                    fi
                done
            done
        for learning_rate in "2.5e-2" "5e-2" "1e-1" "2.5e-1" "5e-1" # "1e-2" 
        do
            for architecture in soft_attempt
                do
                    file=./results_all_metrics_seed_control/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1"_large_"$seed".json
                    if [ ! -f $file ]; then
                        echo $file
                        sbatch --time=0:30:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab \
                            --output ./logs/slurm-%j.out \
                            run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric $1 quail large $seed
                    fi
                done
            done
        done
    done
done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./results_all_metrics_answer/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:05:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 quail models-answer results_all_metrics_answer
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./results_all_metrics_answer/ensemble_"$metric"_"$architecture-pt-$pretrain_learning_rate"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:05:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             run_python.sh ./C_ensemble.py $architecture-pt-$pretrain_learning_rate $learning_rate $top_p $metric whitespace nan $1 quail models-answer results_all_metrics_answer
#                     fi
#                 done
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#             do
#                 file=./results_all_metrics_answer_dreamscape/ensemble_"$metric"_"$architecture"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                 if [ ! -f $file ]; then
#                     sbatch --time=0:05:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                         --output ./logs/slurm-%j.out \
#                         run_python.sh ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 dreamscape models-answer-dreamscape results_all_metrics_answer_dreamscape
#                 fi
#             done
#         done
#     done
# done

# for top_p in "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1"
# do
#     for metric in meteor # rouge_l bleu_4 bleu_3 bleu_2 bleu_1
#     do
#         for pretrain_learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#         do
#             for learning_rate in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"
#             do
#                 for architecture in ghanem_et_al_2022 single_model_with_token_random_token_init separate_models single_model
#                 do
#                     file=./results_all_metrics_answer_dreamscape/ensemble_"$metric"_"$architecture-pt-$pretrain_learning_rate"_probs_"$top_p"_whitespace_nan_"$learning_rate"_"$1".json
#                     if [ ! -f $file ]; then
#                         sbatch --time=0:05:00 --mem-per-cpu=4000M --mincpus=2 --account def-afyshe-ab \
#                             --output ./logs/slurm-%j.out \
#                             run_python.sh ./C_ensemble.py $architecture-pt-$pretrain_learning_rate $learning_rate $top_p $metric whitespace nan $1 dreamscape models-answer-dreamscape results_all_metrics_answer_dreamscape
#                     fi
#                 done
#             done
#         done
#     done
# done
