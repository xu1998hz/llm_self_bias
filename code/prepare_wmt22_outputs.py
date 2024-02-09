from mt_metrics_eval import data

for lang in ["zh-en", "en-de"]:
    evs = data.EvalSet('wmt22', 'en-de')
    mqm_scores = evs.Scores('seg', 'mqm')
    all_scores, all_outputs, all_srcs = [], [], []
    for sys_name, score_ls in mqm_scores.items():
        assert(len(score_ls) == len(evs.sys_outputs[sys_name]))
        assert(len(score_ls) == len(evs.src))
        for src, score, out in zip(evs.src, score_ls, evs.sys_outputs[sys_name]):
            if score != None:
                all_scores += [str(score)+'\n']
                all_outputs += [out+'\n']
                all_srcs += [src+'\n']

    with open(f"model_outputs/wmt_sys/{lang}_base_outputs_wmt_sys.txt", 'w') as f:
        f.writelines(all_outputs)
    
    with open(f"model_outputs/wmt_sys/{lang}_src_wmt_sys.txt", 'w') as f:
        f.writelines(all_srcs)
    
    with open(f"model_outputs/wmt_sys/{lang}_scores_wmt_sys.txt", 'w') as f:
        f.writelines(all_scores)