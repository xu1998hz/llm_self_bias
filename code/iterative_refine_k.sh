# sh code/iterative_refine.sh -m gemini -a transformers -l yor-en -d 0 -b 1 -t mt  -s 10 -i 0 -e True
while getopts "m:a:l:d:t:b:s:i:e:k:f:" flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        a) api_source=${OPTARG};;
        l) lang=${OPTARG};;
        d) device_id=${OPTARG};;
        t) task_type=${OPTARG};;
        b) batch_size=${OPTARG};;
        s) step=${OPTARG};;
        i) init_step=${OPTARG};;
        e) inst_enable=${OPTARG};;
        k) k=${OPTARG};;
        f) feedback_bs=${OPTARG};;
    esac
done

mkdir model_outputs
mkdir model_outputs/${model_name}
mkdir model_outputs/${model_name}/self_refine/
mkdir model_outputs/${model_name}/self_refine/${lang}
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs-${k}/
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores-${k}/
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs-1of${k}/
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores-1of${k}/
mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-raw/
mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-nor/

# we start with initial translation 
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_k.py -task_type "${task_type}" -lang_dir "${lang}" -api_source "${api_source}" -model_type "${model_name}" \
    -save_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs-${k}/${lang}_refinement_100_${model_name}_new_0_rerun.txt" -batch_size "${batch_size}" -k "${k}"

echo "Initial generation is done"

# we generate the corresponding feedback for initial translation
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/select_from_k.py -task_type mt -lang_dir "${lang}" \
    -api_source "${api_source}" -base_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs-${k}/${lang}_refinement_100_${model_name}_new_0_rerun.txt" \
    -model_type "${model_name}" -savename "model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores-${k}/${lang}_eval_100_one-shot_${model_name}_new_0_rerun.txt"  -batch_size "${feedback_bs}" -k "${k}" -instructscore_enable "${inst_enable}" \
    -final_out_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs-1of${k}/${lang}_refinement_100_${model_name}_new_0_rerun.txt" \
    -final_score_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores-1of${k}/${lang}_eval_100_one-shot_${model_name}_new_0_rerun.txt" 

echo "Initial feedback is done"

