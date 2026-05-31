# Track eval — zeroshot-deepseek-v4-flash-mosfet40-v1

- **Predictor:** `llm-zeroshot`
- **Model:** `deepseek-ai/DeepSeek-V4-Flash`
- **Eval set:** `local`
- **Tags:** `version:zeroshot-deepseek-v4-flash-mosfet40-v1, predictor:llm-zeroshot, model-ref:deepseek-ai--DeepSeek-V4-Flash, eval-set:local, real-run, reportable, family:MOSFET, eval-set:local-mosfet40, compare:baseline, split:id, role:baseline, method:rules-in-context, baseline:zeroshot, provider:featherless`

## nextstep
- **by_family/MOSFET:** top1=0.2375, top3=0.2625, top5=0.2625, mrr=0.25
- **by_family/overall:** top1=0.2375, top3=0.2625, top5=0.2625, mrr=0.25
- **by_cut/frac60:** top1=0.25, top3=0.3, top5=0.3, mrr=0.275
- **by_cut/frac80:** top1=0.225, top3=0.225, top5=0.225, mrr=0.225
- **by_cut/overall:** top1=0.2375, top3=0.2625, top5=0.2625, mrr=0.25

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.8591, token_acc=0.0445, block_acc=0.1768
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.8591, token_acc=0.0445, block_acc=0.1768
- **by_cut/frac60:** exact_match=0.0, norm_edit_dist=0.7781, token_acc=0.028, block_acc=0.1077
- **by_cut/frac80:** exact_match=0.0, norm_edit_dist=0.9401, token_acc=0.0609, block_acc=0.2458
- **by_cut/overall:** exact_match=0.0, norm_edit_dist=0.8591, token_acc=0.0445, block_acc=0.1768

## anomaly
- **by_family/MOSFET:** binary_acc=0.525, precision=0.75, recall=0.075, f1=0.1364, roc_auc=0.525, rule_attribution_acc=0.6667, confusion=tp3/fp1/tn39/fn37, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
- **by_family/overall:** binary_acc=0.525, precision=0.75, recall=0.075, f1=0.1364, roc_auc=0.525, rule_attribution_acc=0.6667, confusion=tp3/fp1/tn39/fn37, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_IMPLANT_NO_MASK=0.0]
