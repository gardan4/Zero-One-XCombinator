# Track eval — zeroshot-deepseek-local-n25-v1

- **Predictor:** `llm-zeroshot`
- **Model:** `deepseek-ai/DeepSeek-V4-Flash`
- **Eval set:** `local`
- **Tags:** `version:zeroshot-deepseek-local-n25-v1, predictor:llm-zeroshot, model-ref:deepseek-ai--DeepSeek-V4-Flash, eval-set:local, split:id, role:baseline, method:rules-in-context, baseline:zeroshot, provider:featherless, reportable`

## nextstep
- **by_family/MOSFET:** top1=0.24, top3=0.26, top5=0.26, mrr=0.2467
- **by_family/overall:** top1=0.24, top3=0.26, top5=0.26, mrr=0.2467
- **by_cut/frac60:** top1=0.32, top3=0.36, top5=0.36, mrr=0.3333
- **by_cut/frac80:** top1=0.16, top3=0.16, top5=0.16, mrr=0.16
- **by_cut/overall:** top1=0.24, top3=0.26, top5=0.26, mrr=0.2467

## completion
- **by_family/MOSFET:** exact_match=0.0, norm_edit_dist=0.8496, token_acc=0.0372, block_acc=0.135
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.8496, token_acc=0.0372, block_acc=0.135
- **by_cut/frac60:** exact_match=0.0, norm_edit_dist=0.7874, token_acc=0.0241, block_acc=0.1156
- **by_cut/frac80:** exact_match=0.0, norm_edit_dist=0.9117, token_acc=0.0504, block_acc=0.1543
- **by_cut/overall:** exact_match=0.0, norm_edit_dist=0.8496, token_acc=0.0372, block_acc=0.135

## anomaly
- **by_family/MOSFET:** binary_acc=0.52, precision=0.6667, recall=0.08, f1=0.1429, roc_auc=0.524, rule_attribution_acc=0.0, confusion=tp2/fp1/tn24/fn23, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
- **by_family/overall:** binary_acc=0.52, precision=0.6667, recall=0.08, f1=0.1429, roc_auc=0.524, rule_attribution_acc=0.0, confusion=tp2/fp1/tn24/fn23, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.0, RULE_CMP_NO_DEP=0.0, RULE_DEP_NO_CLEAN=0.0, RULE_ETCH_NO_MASK=0.0, RULE_LITHO_LEVEL_SKIP=0.0]
