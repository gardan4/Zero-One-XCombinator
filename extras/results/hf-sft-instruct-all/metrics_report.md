# Track eval — sft-instruct-all-v2

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a/artifacts/checkpoint-3050`
- **Eval set:** `local`
- **Tags:** `version:sft-instruct-all-v2, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a--artifacts--checkpoint-3050, eval-set:local, train-run:20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a, real-run, reportable, role:finetuned, split:id, family:MOSFET, model-size:1.5b`

## nextstep
- **by_family/MOSFET:** top1=0.475, top3=0.475, top5=0.475, mrr=0.475
- **by_family/overall:** top1=0.475, top3=0.475, top5=0.475, mrr=0.475

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.9881, token_acc=0.415, block_acc=0.555
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9881, token_acc=0.415, block_acc=0.555

## anomaly
- **by_family/MOSFET:** binary_acc=0.495, precision=0.4962, recall=0.66, f1=0.5665, roc_auc=0.495, rule_attribution_acc=0.0, confusion=tp66/fp67/tn33/fn34, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=1.0, RULE_CMP_NO_DEP=0.375, RULE_DEP_NO_CLEAN=0.4, RULE_ETCH_NO_MASK=0.737, RULE_IMPLANT_NO_MASK=1.0]
- **by_family/overall:** binary_acc=0.495, precision=0.4962, recall=0.66, f1=0.5665, roc_auc=0.495, rule_attribution_acc=0.0, confusion=tp66/fp67/tn33/fn34, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=1.0, RULE_CMP_NO_DEP=0.375, RULE_DEP_NO_CLEAN=0.4, RULE_ETCH_NO_MASK=0.737, RULE_IMPLANT_NO_MASK=1.0]
