# Strategy C — 시계열 모멘텀 + 이동평균선 대순환(6단계) + 시장(지수) 필터

본 문서는 “전략 C”의 원리 시험(005930 등) 목적의 **진입/청산 로직**을 정의한다.  
구현은 `EntryRuleType.C_TSMOM_CYCLE`로 제공된다.

> 거래/체결/비용/유닛/손절/TS/피라미딩/숏 제약은 기존 규칙 문서를 따른다.  
> (특히: T일 종가 신호 → T+1 09:00 시장가 체결 가정)

---

## 1) 구성 요소

### 1.1 모멘텀 (Time-Series Momentum)
- 정의: `MOM_L(T) = Close_T / Close_{T-L} - 1`
- 기본: `L = 63` (약 3개월)
- 코드 컬럼: `mom_63` (일반형: `mom_{L}`)

### 1.2 이동평균선 대순환(6단계) — EMA 5/20/40
EMA 3개를 사용한다.
- 단기 EMA: 5
- 중기 EMA: 20
- 장기 EMA: 40

단계는 세 EMA의 “위→아래 순서”로 결정한다.

| Phase | 순서(위→아래) |
|---:|---|
| 1 | 단기 > 중기 > 장기 |
| 2 | 중기 > 단기 > 장기 |
| 3 | 중기 > 장기 > 단기 |
| 4 | 장기 > 중기 > 단기 |
| 5 | 장기 > 단기 > 중기 |
| 6 | 단기 > 장기 > 중기 |

- 코드 컬럼: `cycle_phase_5_20_40` (일반형: `cycle_phase_{s}_{m}_{l}`)

---

## 2) 필터 구조 (시장 > 종목)

전략 C는 **시장(지수) 필터 + 종목 필터**를 동시에 적용한다.

- 종목 phase: `cycle_phase_5_20_40`
- 시장 phase: `mkt_cycle_phase_5_20_40`
  - 시장 데이터는 **KOSPI200 선물 지수 CSV**로부터 계산 (CSV는 외부에서 제공된다고 가정)

---

## 3) 진입 규칙 (롱만)

### 3.1 롱 허용 구간
- 롱은 phase 6→1→2 구간에서만 허용  
  - 즉 `{6, 1, 2}`에서만 LONG 가능

### 3.2 레짐 엔트리(첫 전환일 진입)
T일 종가 기준으로 아래 조건을 만족하면 “레짐 ON”:

- `MOM_63(T) > +5%`
- `stock_phase(T) ∈ {6,1,2}`
- `market_phase(T) ∈ {6,1,2}`

진입은 **레짐이 OFF → ON 으로 바뀌는 첫날**에만 발생한다.

---

## 4) 레짐 청산 (기본 ON)

포지션 보유 중 T일 종가 기준으로 아래 중 하나라도 만족하면,
T+1 시가에 전량 청산한다.

- `MOM_63(T) ≤ 0%`
- `stock_phase(T) ∉ {6,1,2}`
- `market_phase(T) ∉ {6,1,2}`

---

## 5) 실행 예시

```bash
python scripts/toy_005930_backtest.py \
  --csv data/krx100_adj_5000.csv \
  --entry-rule C_TSMOM_CYCLE \
  --market-csv data/kospi200_futures.csv \
  --outdir outputs
```

시장 CSV 없이 실행하려면(시장 필터 비활성):
```bash
python scripts/toy_005930_backtest.py \
  --csv data/krx100_adj_5000.csv \
  --entry-rule C_TSMOM_CYCLE \
  --c-no-market-filter \
  --outdir outputs
```
