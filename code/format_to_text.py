for sys_name in ["mistral_moe", "deepseek_moe", "llama2-7b", "gpt-4", "gpt-3.5-turbo", "gemini"]:
    for i in [0, 5]:
        lines = open(f'model_outputs/{sys_name}/self_refine/yor-en/{sys_name}-scores/yor-en_eval_100_one-shot_{sys_name}_new_5_rerun.txt', 'r').readlines()
        final_str = ''.join(lines)

        savefile = open(f"txt_human_annos/{sys_name}_{i}.txt", 'w')
        savefile.write(f"{sys_name} {i}"+'\n\n')

        for index, ele in enumerate(final_str.split('[SEP_TOKEN_WENDA]')[:-1]):
            savefile.write('-'*20+str(index)+'-'*20+'\n')
            savefile.write(ele+'\n')

        savefile.close()