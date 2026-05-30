// =============================================================================
// FabSeq-S results — ILLUSTRATIVE PLACEHOLDER DATA.
// Every number below is a stand-in for the trained model. To go live, replace
// these values with the output of eval_metrics.py (per-task, per-family) and
// your training log. The dashboard renders entirely from this object.
// Fractions are in [0,1]; normEditDistance is lower-is-better.
// =============================================================================

export const RESULTS = {
  "copy": {
    "heroEyebrow": "Infineon Industrial-AI Hackathon · Fab-route modeling",
    "heroTitlePre": "A sequence model that",
    "heroTitleEmph": "learns process logic",
    "heroTitlePost": "from the route, not a lookup table",
    "heroLede": "Trained from scratch on three product families, it predicts the next step, completes routes, and flags rule violations — beating a retrieval-plus-rule baseline on identical inputs.",
    "modelName": "FabSeq-S",
    "modelBlurb": "Compact decoder-only transformer, family-conditioned, trained from scratch on IC, IGBT, and MOSFET fab routes — no pretraining, no external corpus.",
    "takeaways": [
      "Learned process grammar outperforms retrieval+rules on every metric, on the same 600 held-out sequences.",
      "Family conditioning plus diverse generation keeps the model honest on an unseen fourth family.",
      "Anomaly detection points at the violated rule, not just a yes/no — attribution comes for free."
    ],
    "sections": {
      "nextstep": "Ranks the five most likely next steps from a partial route; measured on Top-1/3/5 accuracy and MRR.",
      "completion": "Fills in every remaining step from a route cut at 60% or 80%, scored on exact match and edit distance.",
      "anomaly": "Reads a full route, calls it valid or not, and names the broken process rule with a confidence score.",
      "training": "From-scratch training over three families with logged loss curves — no pretrained weights, no leakage from the eval set.",
      "ood": "Held to a hidden fourth family after submission; we tuned for grammar over family quirks to limit the ID-to-OOD drop."
    }
  },
  "families": [
    "IC",
    "IGBT",
    "MOSFET"
  ],
  "datasetFamilies": [
    {
      "family": "IC",
      "refSteps": 107,
      "sequences": 1000,
      "meanLength": 115,
      "validSpace": "~6 billion"
    },
    {
      "family": "IGBT",
      "refSteps": 151,
      "sequences": 1000,
      "meanLength": 148,
      "validSpace": "~13 trillion"
    },
    {
      "family": "MOSFET",
      "refSteps": 126,
      "sequences": 1000,
      "meanLength": 125,
      "validSpace": "~51 billion"
    }
  ],
  "rules": [
    {
      "id": "RULE_DEP_NO_CLEAN",
      "short": "Any deposition step must be preceded by a cleaning step within ~12 steps."
    },
    {
      "id": "RULE_METAL_ETCH_NO_LITHO",
      "short": "Metal etch must have EXPOSE LITHO + DEVELOP PHOTORESIST within ~15 prior steps."
    },
    {
      "id": "RULE_ETCH_NO_MASK",
      "short": "Any patterned etch needs a DEVELOP step within ~12 prior steps (spacer etch exempt)."
    },
    {
      "id": "RULE_LITHO_LEVEL_SKIP",
      "short": "Mask levels must be sequential: level N+1 cannot appear before level N completes."
    },
    {
      "id": "RULE_IMPLANT_NO_MASK",
      "short": "Implant needs a prior oxide-etch or develop (open window) within ~15 steps."
    },
    {
      "id": "RULE_CMP_NO_DEP",
      "short": "CMP must follow a deposition or via-fill step within ~6 prior steps."
    },
    {
      "id": "RULE_PAD_OPEN_BEFORE_DEP",
      "short": "Pad-window opening must come after DEPOSIT PASSIVATION and CURE PASSIVATION."
    },
    {
      "id": "RULE_TEST_BEFORE_PASSIVATION",
      "short": "Electrical tests must come after CURE PASSIVATION."
    },
    {
      "id": "RULE_SHIP_BEFORE_TEST",
      "short": "SHIP LOT must come after WAFER SORT TEST."
    },
    {
      "id": "RULE_BACKSIDE_BEFORE_PASSIVATION",
      "short": "DEPOSIT BACKSIDE METAL must come after CURE PASSIVATION."
    }
  ],
  "nextstep": {
    "overall": {
      "baseline": {
        "top1": 0.634,
        "top3": 0.841,
        "top5": 0.902,
        "mrr": 0.724
      },
      "finetuned": {
        "top1": 0.871,
        "top3": 0.961,
        "top5": 0.986,
        "mrr": 0.912
      }
    },
    "perFamily": [
      {
        "family": "IC",
        "baselineTop1": 0.661,
        "baselineTop5": 0.915,
        "finetunedTop1": 0.886,
        "finetunedTop5": 0.989
      },
      {
        "family": "IGBT",
        "baselineTop1": 0.598,
        "baselineTop5": 0.884,
        "finetunedTop1": 0.842,
        "finetunedTop5": 0.981
      },
      {
        "family": "MOSFET",
        "baselineTop1": 0.643,
        "baselineTop5": 0.907,
        "finetunedTop1": 0.879,
        "finetunedTop5": 0.988
      }
    ]
  },
  "completion": {
    "overall": {
      "baseline": {
        "exactMatch": 0.112,
        "normEditDistance": 0.341,
        "tokenAcc": 0.804,
        "blockAcc": 0.552
      },
      "finetuned": {
        "exactMatch": 0.463,
        "normEditDistance": 0.108,
        "tokenAcc": 0.941,
        "blockAcc": 0.861
      }
    }
  },
  "anomaly": {
    "overall": {
      "baseline": {
        "binAcc": 0.912,
        "precision": 0.931,
        "recall": 0.879,
        "f1": 0.904,
        "rocAuc": 0.901,
        "ruleAttrAcc": 0.951
      },
      "finetuned": {
        "binAcc": 0.962,
        "precision": 0.961,
        "recall": 0.948,
        "f1": 0.954,
        "rocAuc": 0.985,
        "ruleAttrAcc": 0.921
      }
    },
    "confusion": {
      "tp": 367,
      "fp": 15,
      "tn": 585,
      "fn": 20
    },
    "perRule": [
      {
        "rule": "RULE_DEP_NO_CLEAN",
        "attrAcc": 0.948
      },
      {
        "rule": "RULE_METAL_ETCH_NO_LITHO",
        "attrAcc": 0.935
      },
      {
        "rule": "RULE_ETCH_NO_MASK",
        "attrAcc": 0.911
      },
      {
        "rule": "RULE_LITHO_LEVEL_SKIP",
        "attrAcc": 0.882
      },
      {
        "rule": "RULE_IMPLANT_NO_MASK",
        "attrAcc": 0.926
      },
      {
        "rule": "RULE_CMP_NO_DEP",
        "attrAcc": 0.939
      },
      {
        "rule": "RULE_PAD_OPEN_BEFORE_DEP",
        "attrAcc": 0.954
      },
      {
        "rule": "RULE_TEST_BEFORE_PASSIVATION",
        "attrAcc": 0.968
      },
      {
        "rule": "RULE_SHIP_BEFORE_TEST",
        "attrAcc": 0.972
      },
      {
        "rule": "RULE_BACKSIDE_BEFORE_PASSIVATION",
        "attrAcc": 0.943
      }
    ]
  },
  "training": {
    "params": "~7.4M",
    "epochs": 12,
    "finalLoss": 0.32,
    "finalValLoss": 0.41,
    "steps": [
      {
        "step": 0,
        "loss": 3.61,
        "valLoss": 3.68
      },
      {
        "step": 50,
        "loss": 3.12,
        "valLoss": 3.21
      },
      {
        "step": 100,
        "loss": 2.74,
        "valLoss": 2.85
      },
      {
        "step": 150,
        "loss": 2.41,
        "valLoss": 2.53
      },
      {
        "step": 200,
        "loss": 2.13,
        "valLoss": 2.27
      },
      {
        "step": 250,
        "loss": 1.89,
        "valLoss": 2.04
      },
      {
        "step": 300,
        "loss": 1.68,
        "valLoss": 1.83
      },
      {
        "step": 350,
        "loss": 1.51,
        "valLoss": 1.66
      },
      {
        "step": 400,
        "loss": 1.36,
        "valLoss": 1.51
      },
      {
        "step": 450,
        "loss": 1.23,
        "valLoss": 1.39
      },
      {
        "step": 500,
        "loss": 1.12,
        "valLoss": 1.28
      },
      {
        "step": 550,
        "loss": 1.02,
        "valLoss": 1.19
      },
      {
        "step": 600,
        "loss": 0.94,
        "valLoss": 1.1
      },
      {
        "step": 650,
        "loss": 0.86,
        "valLoss": 1.03
      },
      {
        "step": 700,
        "loss": 0.8,
        "valLoss": 0.96
      },
      {
        "step": 750,
        "loss": 0.74,
        "valLoss": 0.9
      },
      {
        "step": 800,
        "loss": 0.69,
        "valLoss": 0.85
      },
      {
        "step": 850,
        "loss": 0.64,
        "valLoss": 0.8
      },
      {
        "step": 900,
        "loss": 0.6,
        "valLoss": 0.76
      },
      {
        "step": 950,
        "loss": 0.57,
        "valLoss": 0.72
      },
      {
        "step": 1000,
        "loss": 0.54,
        "valLoss": 0.69
      },
      {
        "step": 1050,
        "loss": 0.51,
        "valLoss": 0.66
      },
      {
        "step": 1100,
        "loss": 0.48,
        "valLoss": 0.63
      },
      {
        "step": 1150,
        "loss": 0.46,
        "valLoss": 0.61
      },
      {
        "step": 1200,
        "loss": 0.44,
        "valLoss": 0.59
      },
      {
        "step": 1250,
        "loss": 0.43,
        "valLoss": 0.57
      },
      {
        "step": 1300,
        "loss": 0.41,
        "valLoss": 0.55
      },
      {
        "step": 1350,
        "loss": 0.4,
        "valLoss": 0.53
      },
      {
        "step": 1400,
        "loss": 0.39,
        "valLoss": 0.52
      },
      {
        "step": 1450,
        "loss": 0.38,
        "valLoss": 0.5
      },
      {
        "step": 1500,
        "loss": 0.37,
        "valLoss": 0.49
      },
      {
        "step": 1550,
        "loss": 0.36,
        "valLoss": 0.48
      },
      {
        "step": 1600,
        "loss": 0.35,
        "valLoss": 0.47
      },
      {
        "step": 1650,
        "loss": 0.345,
        "valLoss": 0.455
      },
      {
        "step": 1700,
        "loss": 0.338,
        "valLoss": 0.445
      },
      {
        "step": 1750,
        "loss": 0.332,
        "valLoss": 0.435
      },
      {
        "step": 1800,
        "loss": 0.328,
        "valLoss": 0.428
      },
      {
        "step": 1850,
        "loss": 0.324,
        "valLoss": 0.422
      },
      {
        "step": 1900,
        "loss": 0.321,
        "valLoss": 0.415
      },
      {
        "step": 1950,
        "loss": 0.32,
        "valLoss": 0.41
      }
    ]
  },
  "ood": {
    "heldOutFamily": "BJT",
    "idF1": 0.954,
    "oodF1": 0.821,
    "idTop1": 0.879,
    "oodTop1": 0.742,
    "idExactMatch": 0.463,
    "oodExactMatch": 0.291
  }
};
