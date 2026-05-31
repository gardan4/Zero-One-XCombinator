# Track eval — sft-instruct-3b-v2

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260531_025032_sft_leonardo-sft-fab-instruct-3b_88c885/artifacts`
- **Eval set:** `local`
- **Tags:** `version:sft-instruct-3b-v2, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260531_025032_sft_leonardo-sft-fab-instruct-3b_88c885--artifacts, eval-set:local, train-run:20260531_025032_sft_leonardo-sft-fab-instruct-3b_88c885, real-run, reportable, role:finetuned, split:id, family:MOSFET, model-size:3b`

## nextstep
- **by_family/MOSFET:** top1=0.435, top3=0.435, top5=0.435, mrr=0.435
- **by_family/overall:** top1=0.435, top3=0.435, top5=0.435, mrr=0.435

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.9883, token_acc=0.445, block_acc=0.555
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9883, token_acc=0.445, block_acc=0.555

## anomaly
- **by_family/MOSFET:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
