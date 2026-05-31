# Track eval — sft-scale-instruct-300-v2

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_234550_sft_leonardo-sft-scale-instruct-300_c72d26/artifacts`
- **Eval set:** `local`
- **Tags:** `version:sft-scale-instruct-300-v2, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_234550_sft_leonardo-sft-scale-instruct-300_c72d26--artifacts, eval-set:local, train-run:20260530_234550_sft_leonardo-sft-scale-instruct-300_c72d26, real-run, reportable, role:finetuned, split:id, family:MOSFET, scale, data-size:300`

## nextstep
- **by_family/MOSFET:** top1=0.435, top3=0.435, top5=0.435, mrr=0.435
- **by_family/overall:** top1=0.435, top3=0.435, top5=0.435, mrr=0.435

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.9899, token_acc=0.36, block_acc=0.5
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9899, token_acc=0.36, block_acc=0.5

## anomaly
- **by_family/MOSFET:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
