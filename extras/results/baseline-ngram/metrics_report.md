# Track eval — ngram-baseline-v1

- **Predictor:** `ngram`
- **Eval set:** `local`
- **Tags:** `version:ngram-baseline-v1, predictor:ngram, eval-set:local, split:id, role:compare-suite, role:baseline, suite:local_compare, suite-run:ngram-baseline`

## nextstep
- **by_family/MOSFET:** top1=0.69, top3=0.995, top5=1.0, mrr=0.8429
- **by_family/overall:** top1=0.69, top3=0.995, top5=1.0, mrr=0.8429

## completion
- **by_family/MOSFET:** exact_match=0.015, norm_edit_dist=0.4971, token_acc=0.3947, block_acc=0.6369
- **by_family/overall:** exact_match=0.015, norm_edit_dist=0.4971, token_acc=0.3947, block_acc=0.6369

## anomaly
- **by_family/MOSFET:** binary_acc=0.9, precision=1.0, recall=0.8, f1=0.8889, roc_auc=0.9999, rule_attribution_acc=0.0, confusion=tp80/fp0/tn100/fn20, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=1.0, RULE_CMP_NO_DEP=0.375, RULE_DEP_NO_CLEAN=0.6, RULE_ETCH_NO_MASK=0.526, RULE_IMPLANT_NO_MASK=1.0]
- **by_family/overall:** binary_acc=0.9, precision=1.0, recall=0.8, f1=0.8889, roc_auc=0.9999, rule_attribution_acc=0.0, confusion=tp80/fp0/tn100/fn20, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=1.0, RULE_CMP_NO_DEP=0.375, RULE_DEP_NO_CLEAN=0.6, RULE_ETCH_NO_MASK=0.526, RULE_IMPLANT_NO_MASK=1.0]
