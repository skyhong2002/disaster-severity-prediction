# 20260523 1650 Final Selection Matrix

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Stop public-chase: `True`
- Quota recommendation: `do_not_submit_more_today`

## Final Recommendation

Use `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` as the leaderboard-optimal submitted file if final selection is based on submitted Kaggle artifacts. Use `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` as the clean reportable model lineage. Do not present the public-chase blend as a reportable modeling improvement.

## Candidate Matrix

| Candidate | Label | Public MAE | SHA-12 | Reportable | Recommendation |
|---|---|---:|---|---|---|
| `public_chase_best_alpha_p035` | `public-chase` | `0.7991` | `d550c9cbc465` | `False` | final_selection_candidate_if_rule_allows; do_not_use_as_reportable_method_claim |
| `reportable_legal_anchor_cat35` | `reportable` | `0.8124` | `bee6f618828d` | `True` | primary_reportable_model_lineage; conservative_final_fallback |
| `historical_public_reference_v0` | `public-chase-source-reference-only` | `0.8094` | `2f1eb3575419` | `False` | reference_only_do_not_claim_as_current_method |
| `gru_constrained_stack` | `reportable-experiment` | `0.985` | `2bf9f97b852a` | `False` | do_not_select_final |
| `raw_gru` | `reportable-experiment` | `0.9916` | `5ca7f847eba1` | `False` | do_not_select_final |

## Risk Notes

- GRU stack/raw were submitted and returned public `0.9850`/`0.9916`; do not use them as final/private anchors.
- `restored_20260522_*` and `restored_unverified_*` remain blocked as submit sources.
- No additional public-chase quota should be spent after the `0.7991` crossing.

## Artifacts

- `final_selection_matrix.json`
- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
