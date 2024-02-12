model_name="gemini"

for iter in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
do
python3 code/compute_bias_commongen.py -llm_score_file "model_outputs/${model_name}/self_refine/commongen/${model_name}-scores/commongen_eval_100_one-shot_${model_name}_new_${iter}_rerun.txt" \
 -llm_out_file "model_outputs/${model_name}/self_refine/commongen/${model_name}-outputs/commongen_refinement_100_${model_name}_new_${iter}_rerun.txt"
done