# Track eval — sft-instruct-all-IC-v2

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a/artifacts/checkpoint-3050`
- **Eval set:** `local`
- **Tags:** `version:sft-instruct-all-IC-v2, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a--artifacts--checkpoint-3050, eval-set:local, train-run:20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a, real-run, reportable, role:finetuned, split:id, family:IC`

## nextstep
- **by_family/IC:** top1=0.43, top3=0.43, top5=0.43, mrr=0.43
- **by_family/overall:** top1=0.43, top3=0.43, top5=0.43, mrr=0.43

## completion
- **by_family/IC:** exact_match=0.0, norm_edit_dist=0.9904, token_acc=0.39, block_acc=0.5
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9904, token_acc=0.39, block_acc=0.5

## anomaly
- **by_family/IC:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
