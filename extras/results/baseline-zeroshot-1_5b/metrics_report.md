# Track eval — baseline-zeroshot-1_5b-v1

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/hf-local/Qwen2.5-1.5B-Instruct`
- **Eval set:** `local`
- **Tags:** `version:baseline-zeroshot-1_5b-v1, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--hf-local--Qwen2.5-1.5B-Instruct, eval-set:local, real-run, reportable, role:baseline, baseline:zeroshot, split:id, family:MOSFET, model-size:1.5b-base`

## nextstep
- **by_family/MOSFET:** top1=0.0, top3=0.0, top5=0.0, mrr=0.0
- **by_family/overall:** top1=0.0, top3=0.0, top5=0.0, mrr=0.0

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.9984, token_acc=0.0, block_acc=0.0052
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9984, token_acc=0.0, block_acc=0.0052

## anomaly
- **by_family/MOSFET:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.5, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, confusion=tp0/fp0/tn100/fn100, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
