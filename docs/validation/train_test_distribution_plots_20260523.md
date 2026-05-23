# Train/Test Distribution Plot Gallery

Each PNG has density or share bars on the top row and ECDF curves on the bottom row.
The train panels compare test against full train, recent tail train, and season-matched previous-year train.

- [Multi-page PDF](train_test_distribution_plots_20260523.pdf)
- PNG directory: `train_test_distribution_plots_20260523`

## Raw weather features

| feature | plot |
|---|---|
| `prec` | [daily precipitation](train_test_distribution_plots_20260523/01_prec.png) |
| `humidity` | [humidity](train_test_distribution_plots_20260523/02_humidity.png) |
| `tmp` | [mean temperature](train_test_distribution_plots_20260523/03_tmp.png) |
| `dp_tmp` | [dew point temperature](train_test_distribution_plots_20260523/04_dp_tmp.png) |
| `wb_tmp` | [wet bulb temperature](train_test_distribution_plots_20260523/05_wb_tmp.png) |
| `tmp_max` | [max temperature](train_test_distribution_plots_20260523/06_tmp_max.png) |
| `tmp_min` | [min temperature](train_test_distribution_plots_20260523/07_tmp_min.png) |
| `tmp_range` | [daily temperature range](train_test_distribution_plots_20260523/08_tmp_range.png) |
| `surf_tmp` | [surface temperature](train_test_distribution_plots_20260523/09_surf_tmp.png) |
| `wind` | [wind](train_test_distribution_plots_20260523/10_wind.png) |

## Daily dryness proxies

| feature | plot |
|---|---|
| `dry_day` | [dry-day share, prec < 0.1](train_test_distribution_plots_20260523/11_dry_day.png) |
| `rain_day_ge_1` | [rain-day share, prec >= 1](train_test_distribution_plots_20260523/12_rain_day_ge_1.png) |
| `heavy_precip_ge_10` | [heavy-rain share, prec >= 10](train_test_distribution_plots_20260523/13_heavy_precip_ge_10.png) |
| `dew_point_depression` | [tmp - dew point](train_test_distribution_plots_20260523/14_dew_point_depression.png) |
| `wet_bulb_depression` | [tmp - wet bulb](train_test_distribution_plots_20260523/15_wet_bulb_depression.png) |
| `surf_air_temp_diff` | [surface temp - air temp](train_test_distribution_plots_20260523/16_surf_air_temp_diff.png) |
| `dryness_idx_daily` | [tmp_range * tmp_max](train_test_distribution_plots_20260523/17_dryness_idx_daily.png) |
| `heat_humidity_idx_daily` | [tmp_max - humidity](train_test_distribution_plots_20260523/18_heat_humidity_idx_daily.png) |

## Shifted rolling context

| feature | plot |
|---|---|
| `prec_sum_30d` | [30-day precipitation sum](train_test_distribution_plots_20260523/19_prec_sum_30d.png) |
| `prec_sum_90d` | [90-day precipitation sum](train_test_distribution_plots_20260523/20_prec_sum_90d.png) |
| `prec_sum_180d` | [180-day precipitation sum](train_test_distribution_plots_20260523/21_prec_sum_180d.png) |
| `dry_days_30d` | [30-day dry days](train_test_distribution_plots_20260523/22_dry_days_30d.png) |
| `dry_days_90d` | [90-day dry days](train_test_distribution_plots_20260523/23_dry_days_90d.png) |
| `dry_share_90d` | [90-day dry-day share](train_test_distribution_plots_20260523/24_dry_share_90d.png) |
| `rain_days_90d` | [90-day rain days, prec >= 1](train_test_distribution_plots_20260523/25_rain_days_90d.png) |
| `hot_days_90d` | [90-day hot days, tmp_max >= 30](train_test_distribution_plots_20260523/26_hot_days_90d.png) |
| `tmp_mean_90d` | [90-day mean temperature](train_test_distribution_plots_20260523/27_tmp_mean_90d.png) |
| `humidity_mean_90d` | [90-day mean humidity](train_test_distribution_plots_20260523/28_humidity_mean_90d.png) |
| `dew_depression_mean_90d` | [90-day mean tmp-dew point spread](train_test_distribution_plots_20260523/29_dew_depression_mean_90d.png) |
| `wet_bulb_depression_mean_90d` | [90-day mean tmp-wet bulb spread](train_test_distribution_plots_20260523/30_wet_bulb_depression_mean_90d.png) |
| `max_dry_streak_90d` | [max dry streak in prior 90d](train_test_distribution_plots_20260523/31_max_dry_streak_90d.png) |
| `rolling_drought_idx_90d` | [90-day temp / precipitation proxy](train_test_distribution_plots_20260523/32_rolling_drought_idx_90d.png) |
