# Track eval — sft-scale-instruct-2000-local-v3

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_234606_sft_leonardo-sft-scale-instruct-2000_ee052b/artifacts/checkpoint-2900`
- **Eval set:** `local`
- **Tags:** `version:sft-scale-instruct-2000-local-v3, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_234606_sft_leonardo-sft-scale-instruct-2000_ee052b--artifacts--checkpoint-2900, eval-set:local, train-run:20260530_234606_sft_leonardo-sft-scale-instruct-2000_ee052b, real-run, reportable, role:finetuned, split:id, family:MOSFET, scale, data-size:2000`

## nextstep
- **by_family/MOSFET:** top1=0.525, top3=0.525, top5=0.525, mrr=0.525
- **by_family/overall:** top1=0.525, top3=0.525, top5=0.525, mrr=0.525
- **by_cut/frac60:** top1=0.75, top3=0.75, top5=0.75, mrr=0.75
- **by_cut/frac80:** top1=0.3, top3=0.3, top5=0.3, mrr=0.3
- **by_cut/overall:** top1=0.525, top3=0.525, top5=0.525, mrr=0.525

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.988, token_acc=0.46, block_acc=0.735
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.988, token_acc=0.46, block_acc=0.735
- **by_cut/frac60:** exact_match=0.0, norm_edit_dist=0.9846, token_acc=0.7, block_acc=0.76
- **by_cut/frac80:** exact_match=0.0, norm_edit_dist=0.9913, token_acc=0.22, block_acc=0.71
- **by_cut/overall:** exact_match=0.0, norm_edit_dist=0.988, token_acc=0.46, block_acc=0.735

## anomaly
- **by_family/MOSFET:** binary_acc=0.505, precision=0.5455, recall=0.06, f1=0.1081, roc_auc=0.505, rule_attribution_acc=0.0, confusion=tp6/fp5/tn95/fn94, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.505, precision=0.5455, recall=0.06, f1=0.1081, roc_auc=0.505, rule_attribution_acc=0.0, confusion=tp6/fp5/tn95/fn94, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
