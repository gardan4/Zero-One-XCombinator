# Track eval — sft-lofo-igbt-v1

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_192551_sft_leonardo-sft-fab-lofo-igbt_7c1851/artifacts`
- **Eval set:** `local`
- **Tags:** `version:sft-lofo-igbt-v1, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_192551_sft_leonardo-sft-fab-lofo-igbt_7c1851--artifacts, eval-set:local, real-run, reportable, role:finetuned, split:ood, family:IGBT`

## nextstep
- **by_family/IGBT:** top1=0.25, top3=0.315, top5=0.315, mrr=0.28
- **by_family/overall:** top1=0.25, top3=0.315, top5=0.315, mrr=0.28

## completion
- **by_family/IGBT:** exact_match=0.0, norm_edit_dist=0.5996, token_acc=0.0045, block_acc=0.0954
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.5996, token_acc=0.0045, block_acc=0.0954

## anomaly
- **by_family/IGBT:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
