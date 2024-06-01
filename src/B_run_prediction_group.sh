#!/bin/bash
echo "Updated..."
shopt -s expand_aliases
export REPO_DIR=/scratch/vonderoh/EG-uni-cedar
source $REPO_DIR/activate.sh
# $architecture $learning_rate $question_type $top_p "$dataset"
for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
do
    file=./models-$6/$1/$2/$question_type/"predictions_nucleus_"$4"_"$3".json"
    if [ ! -f "$file" ]; then
        echo "Running $question_type"
        time python3 ./B_predict.py "$1" "$2" "$question_type" "$3" "$4" "$5" "$6"
    fi
done