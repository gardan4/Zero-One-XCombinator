# Track eval — sft-lofo-ic-v1

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_192559_sft_leonardo-sft-fab-lofo-ic_ce7ff7/artifacts`
- **Eval set:** `local`
- **Tags:** `version:sft-lofo-ic-v1, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_192559_sft_leonardo-sft-fab-lofo-ic_ce7ff7--artifacts, eval-set:local, real-run, reportable, role:finetuned, split:ood, family:IC`

## nextstep
- **by_family/IC:** top1=0.25, top3=0.34, top5=0.355, mrr=0.2971
- **by_family/overall:** top1=0.25, top3=0.34, top5=0.355, mrr=0.2971

## completion
- **by_family/IC:** exact_match=0.0, norm_edit_dist=0.5698, token_acc=0.054, block_acc=0.1729
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.5698, token_acc=0.054, block_acc=0.1729

## anomaly
- **by_family/IC:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
