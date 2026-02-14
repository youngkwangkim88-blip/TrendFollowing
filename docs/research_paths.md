# Research paths & conventions (v2)

본 리포는 **핵심 전략/엔진 코드는 `src/`**, 실험·백테스트·콤보 러너는 **`scripts/`** 에만 둡니다.  
(즉, `src/` 안에는 “실험을 돌리는 스크립트”가 들어가지 않습니다.)

## 1) Research scripts

- 콤보 러너(005930, 3-part report)
  - `scripts/toy_005930_combo_compare.py`

## 2) Output layout (고정)

모든 실험 결과는 아래 구조로 저장합니다.

- `outputs/<experiment_name>/...`

예: `--experiment combo_005930_v4` 인 경우

- `outputs/combo_005930_v4/`
  - `results_long_only.csv`
  - `results_short_only.csv`
  - `report_three_parts.html`
  - `runs_long/<run_id>/...`
    - `config.json`
    - `plot_data.csv`
    - `equity_curve.csv`
    - `trades.csv`
    - `summary.json`
    - `abc.html`
  - `runs_short/<run_id>/...`
    - (동일)

## 3) Filter rule conventions

- Filter A (PL filter): **항상 ON (기본 탑재)**
  - 직전 거래가 이익이면, “바로 다음” 반대 포지션 진입 1회 금지
- 콤보 차원에서 테스트하는 필터는 B/C만:
  - `NONE, B, C, BC`

