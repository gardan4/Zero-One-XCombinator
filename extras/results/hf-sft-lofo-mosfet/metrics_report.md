# Track eval — sft-lofo-mosfet-v1

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_192543_sft_leonardo-sft-fab-lofo-mosfet_912e9f/artifacts`
- **Eval set:** `local`
- **Tags:** `version:sft-lofo-mosfet-v1, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_192543_sft_leonardo-sft-fab-lofo-mosfet_912e9f--artifacts, eval-set:local, real-run, reportable, role:finetuned, split:ood, family:MOSFET`

## nextstep
- **by_family/MOSFET:** top1=0.06, top3=0.085, top5=0.085, mrr=0.0717
- **by_family/overall:** top1=0.06, top3=0.085, top5=0.085, mrr=0.0717

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.6732, token_acc=0.0403, block_acc=0.2009
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.6732, token_acc=0.0403, block_acc=0.2009

## anomaly
- **by_family/MOSFET:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
