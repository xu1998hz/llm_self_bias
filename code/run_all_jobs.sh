# for MODEL_TYPE in "gemini" "gpt-4" "gpt-turbo-3.5" 
# do  
#     for LANG in "yor-en" "zh-en" "en-de" 
#     do  
#         echo "model_outputs/${MODEL_TYPE}/${LANG}_base_outputs_${MODEL_TYPE}.txt"
#         CUDA_VISIBLE_DEVICES=1 python3 code/run_mistral.py -lang_dir "${LANG}" -task_type feedback -batch_size 16 -suffix "${MODEL_TYPE}" -src_file "srcs/${LANG}_src_100.txt" -out_file "model_outputs/${MODEL_TYPE}/${LANG}_base_outputs_${MODEL_TYPE}.txt"
#     done
# done

# CUDA_VISIBLE_DEVICES=0 nohup python3 code/run_mistral.py -lang_dir zh-en -task_type feedback -batch_size 16 -suffix refined_2nd_gpt-4 -src_file srcs/yor-en_src_100.txt -out_file model_outputs/gpt-4/yor-en_self_refinement_100_gpt-4_2nd.txt > yor-en_1.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 code/run_mistral.py -lang_dir en-de -task_type feedback -batch_size 16 -suffix refined_3rd_gpt-4 -src_file srcs/yor-en_src_100.txt -out_file model_outputs/gpt-4/yor-en_self_refinement_100_gpt-4_3rd.txt > yor-en_2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python3 code/run_mistral.py -lang_dir yor-en -task_type feedback -batch_size 16 -suffix refined_1st_gpt-4 -src_file srcs/yor-en_src_100.txt -out_file model_outputs/gpt-4/yor-en_self_refinement_100_gpt-4_1st.txt > yor-en.out 2>&1 &

# for MODEL_TYPE in "gemini" # "gpt-4" "gpt-3.5-turbo" # "gemini" # "gpt-4"  # "gemini"  # "mistreal"
# do  
#     for LANG in "yor-en" #"zh-en" "en-de" 
#     do  
#         # echo "model_outputs/${MODEL_TYPE}/${LANG}_base_outputs_${MODEL_TYPE}.txt"
#         OPENAI_API_KEY="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" python3 code/generate_feedback.py -lang_dir "${LANG}" -suffix "refined_3rd_gpt4" -api_source google -base_name "model_outputs/gpt-4/yor-en_self_refinement_100_gpt-4_3rd.txt" -model_type "${MODEL_TYPE}" # "model_outputs/${MODEL_TYPE}/${LANG}_base_outputs_${MODEL_TYPE}.txt"
#     done
# done

# for MODEL_TYPE in "gpt-4" # "gemini" "gpt-3.5-turbo" "mistreal" 
# do  
#     for LANG in "yor-en" # "zh-en" "en-de" 
#     do  
#         OPENAI_API_KEY="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" python3 code/refinement.py -lang_dir "${LANG}" -out "model_outputs/${MODEL_TYPE}/${LANG}_self_refinement_100_${MODEL_TYPE}_2nd.txt" -suffix "${MODEL_TYPE}_3rd" -eval "model_outputs/${MODEL_TYPE}/${LANG}_eval_100_one-shot_refined_2nd_${MODEL_TYPE}.txt" -api_source openai -model_type "${MODEL_TYPE}"
#     done
# done