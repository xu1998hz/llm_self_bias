# SYS_NAME="gpt-4"
# OPENAI_API_KEY="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" python3 code/refinement.py -lang_dir yor-en \
# -out "model_outputs/${SYS_NAME}/yor-en_self_refinement_100_${SYS_NAME}_new_11_rerun.txt" \
# -eval "model_outputs/${SYS_NAME}/yor-en_eval_100_one-shot_${SYS_NAME}_new_11_rerun.txt" \
# -suffix "${SYS_NAME}_new_12_rerun" -eval_suffix "${SYS_NAME}_new_12_rerun" -api_source openai -model_type "${SYS_NAME}" -task_type mt

MODEL_TYPE="gemini"
SYS_NAME="gpt-4"
# for SYS_NAME in "gpt-3.5-turbo" "gpt-4" "gemini"
# do
OPENAI_API_KEY="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" python3 code/generate_feedback.py \
-task_type mt \
-lang_dir yor-en \
-suffix "${SYS_NAME}_new_3_rerun" \
-api_source google \
-model_type "${MODEL_TYPE}" \
-base_name "model_outputs/${SYS_NAME}/yor-en_self_refinement_100_${SYS_NAME}_new_3_rerun.txt" \
-last_feedback "model_outputs/${MODEL_TYPE}/yor-en_eval_100_one-shot_${SYS_NAME}_new_2_rerun.txt"
# done

# for SYS_NAME in "gpt-3.5-turbo" "gpt-4" "gemini"
# do
#     MODEL_TYPE="gpt-4"
#     OPENAI_API_KEY="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" python3 code/generate_feedback.py \
#     -task_type mt \
#     -lang_dir yor-en \
#     -suffix "refined_2nd_${SYS_NAME}_rerun" \
#     -api_source openai \
#     -base_name "model_outputs/${SYS_NAME}/yor-en_self_refinement_100_${SYS_NAME}_2nd.txt" \
#     -model_type "${MODEL_TYPE}"
# done
# -last_feedback "model_outputs/${MODEL_TYPE}/yor-en_eval_100_one-shot_refined_2nd_${SYS_NAME}.txt"   