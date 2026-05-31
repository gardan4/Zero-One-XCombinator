# Track eval — sft-instruct-all-IGBT-v2

- **Predictor:** `hf`
- **Model:** `/leonardo_scratch/large/usertrain/a08trd0f/zo-experiments/20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a/artifacts/checkpoint-3050`
- **Eval set:** `local`
- **Tags:** `version:sft-instruct-all-IGBT-v2, predictor:hf, model-ref:--leonardo_scratch--large--usertrain--a08trd0f--zo-experiments--20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a--artifacts--checkpoint-3050, eval-set:local, train-run:20260530_234537_sft_leonardo-sft-fab-instruct-all_869c8a, real-run, reportable, role:finetuned, split:id, family:IGBT`

## nextstep
- **by_family/IGBT:** top1=0.315, top3=0.315, top5=0.315, mrr=0.315
- **by_family/overall:** top1=0.315, top3=0.315, top5=0.315, mrr=0.315

## completion
- **by_family/IGBT:** exact_match=0.0, norm_edit_dist=0.9943, token_acc=0.215, block_acc=0.365
- **by_family/overall:** exact_match=0.0, norm_edit_dist=0.9943, token_acc=0.215, block_acc=0.365

## anomaly
- **by_family/IGBT:** binary_acc=0.54, precision=0.5426, recall=0.51, f1=0.5258, roc_auc=0.54, rule_attribution_acc=0.0588, confusion=tp51/fp43/tn57/fn49, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.9, RULE_CMP_NO_DEP=0.727, RULE_DEP_NO_CLEAN=0.75, RULE_ETCH_NO_MASK=0.476, RULE_IMPLANT_NO_MASK=0.6]
- **by_family/overall:** binary_acc=0.54, precision=0.5426, recall=0.51, f1=0.5258, roc_auc=0.54, rule_attribution_acc=0.0588, confusion=tp51/fp43/tn57/fn49, per_rule=[RULE_BACKSIDE_BEFORE_PASSIVATION=0.9, RULE_CMP_NO_DEP=0.727, RULE_DEP_NO_CLEAN=0.75, RULE_ETCH_NO_MASK=0.476, RULE_IMPLANT_NO_MASK=0.6]
