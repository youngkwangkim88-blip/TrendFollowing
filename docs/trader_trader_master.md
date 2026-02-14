# Trader / TraderMaster 아키텍처

이 문서는 v2 시스템을 **트레이더(종목/전략 단위) + 트레이더 마스터(계좌/자본 배분 단위)**로 확장하기 위한 코드 구조와 동작을 정리합니다.

## 목표

1. **종목/섹터별 최적 전략을 독립적으로 튜닝**할 수 있도록, “포지션 상태(pos_*) + 전략 규칙”을 종목 단위 객체(Trader)로 캡슐화한다.
2. **자본 배분 문제(최적화)의 외부화를 위한 스캐폴딩**을 제공한다.
   - 예: 전자 업종 트레이더, 내수 서비스 트레이더, 레버리지/인버스 ETF 트레이더 등
3. 보고/시각화/디버깅 편의성 개선
   - “고스트 타점” (피라미딩으로 인해 entry_price가 평균단가로 덮이는 문제) 제거
   - 트레이더에게 “어떻게 거래했니?”를 물었을 때, **Fill(체결) 로그 + Trade(요약)**를 재현 가능하게 출력

## 개념

### Trader

- **하나의 종목(symbol)**을 특정 전략/규칙으로 운영하는 개별 운영자
- 책임 범위
  - 포지션 상태 보유: `pos_side, pos_shares, pos_units, pos_avg_price, pos_h_max, pos_l_min, pos_ts_active, pos_even_armed, ...`
  - T일 종가 기준 신호 평가 → T+1 시가 시장가 체결(스케줄링)
  - TS/손절/even-stop/emergency stop/피라미딩 등 **포지션 관리 로직**
  - 체결(Fill) 로그 기록 및 트레이드(Trade) 요약 생성

### TraderMaster

- **계좌(또는 계좌 내 슬리브)**를 관리하고, 트레이더들에게 자본/제약을 배분하는 상위 관리자
- 현재 구현에서는 단순화를 위해 **트레이더 1개 = 슬리브 계좌 1개(Account)** 구조
- 책임 범위
  - 계좌 현금흐름 처리: long buy/sell, short sell/cover, CMA↔trading cash 이동
  - 포트폴리오-wide 제약
    - 시스템 총 unit 제한 (기본 10)
    - 동일 종목 동시 포지션 금지(기본: symbol exclusivity)

## 핵심 변경점

### 1) Trade 레코드의 “첫 진입가”와 “평단가” 분리

피라미딩이 들어가는 추세추종에서는 다음 두 값이 다릅니다.

- 첫 진입가: 차트 상에서 의미 있는 최초 체결가
- 평균단가: 피라미딩 포함 최종 평단가(실제 PnL 계산에 사용)

따라서 `Trade`는 아래 필드를 가집니다.

- `entry_date`, `entry_price`: **첫 진입**
- `avg_entry_price`, `entry_notional_gross`, `num_entries`: **피라미딩 포함 집계**

이로 인해 보고서(abc.html)에서 과거 “차트 허공에 찍히던 진입 마커” 문제가 제거됩니다.

### 2) fills.csv 도입

모든 ENTRY/PYRAMID/EXIT를 체결 단위로 기록합니다.

- 차트는 **trades.csv 기반이 아니라 fills.csv 기반**으로 마커를 찍을 수 있습니다.
- 피라미딩을 했을 때도, 각 피라미딩 체결 지점이 그대로 표시됩니다.

## 실행 방법

### 단일 종목(기존 호환)

기존과 동일하게 `SingleSymbolBacktester`로 실행합니다.

- 출력에 `fills.csv`가 추가됩니다.
- abc.html은 `fills.csv`를 자동 감지하여 시각화를 개선합니다.

### 다중 트레이더(포트폴리오)

트레이더 설정을 CSV/XLSX로 정의하고, 한 번에 실행합니다.

```bash
python scripts/run_trader_master_from_config.py \
  --csv data/krx100_adj_5000.csv \
  --trader-config data/traders.xlsx \
  --sheet Sheet1 \
  --out-dir runs/portfolio_test
```

#### trader-config 스키마(최소)

필수 컬럼:
- `trader_id`
- `symbol`

선택 컬럼:
- `trade_mode` (LONG_ONLY / SHORT_ONLY / LONG_SHORT)
- `entry_rule`, `entry_rule_long`, `entry_rule_short`
- `ts_type`, `pyramiding_type`
- `initial_capital`, `max_units_per_symbol`, `one_trading_risk`, `short_notional_limit` 등

## 향후 확장 방향

- **섹터별 최적화**: 섹터마다 트레이더를 따로 두고 `entry_rule/TS/피라미딩` 등을 다르게 튜닝
- **상품별 운용 제약**: 예) 레버리지/인버스 ETF는 최대 1 unit 등
- **TraderMaster 최적화 문제**
  - 목적함수: 전체 NAV CAGR 최대화, MDD/리스크 제약 포함
  - 의사결정변수: 트레이더별 배정 자본, max units, 우선순위(priority), 동시 보유 허용/금지 규칙 등

