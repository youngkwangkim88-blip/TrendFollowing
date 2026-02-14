# 진입 규칙 문서 (Entry Rules & Signal Options)
> 본 문서는 `spec_updated_v2.md`, `capital_management_rules_v2.md`와 함께 사용되는 **진입(Entry) 신호 규칙**을 정의한다.  
> 진입 규칙은 단일 고정이 아니라 **옵션 파라미터(열거형)로 관리**하며, 백테스트/전진분석을 통해 선택/검증한다.  
> 업데이트: 2026-02-10 (Asia/Seoul)

---

## 1) 적용 범위 및 공통 전제
### 1.1 시장/세션/집행
- 거래소: **KRX만 사용**
- 모든 주문 체결 시간대: **09:00 ~ 15:20** (세션 내 체결 가정)
- 진입 주문 방식: **익일 09:00 시장가**  
  - 신호 산출 시점: `T일 종가(C)`  
  - 진입 체결 시점(근사): `T+1일 시가(O)` (09:00 시장가)

### 1.2 데이터
- 가격 데이터는 **항상 수정주가(Adjusted OHLC)** 를 사용한다.
- 지표 계산도 수정주가 기준으로 수행한다.

### 1.3 포지션/한도 게이트(진입 공통 필터)
진입 신호가 발생해도 아래를 만족하지 않으면 **주문을 생성하지 않는다.**

- (G1) 동일 종목 롱/숏 동시 보유 금지
  - 해당 종목에 이미 포지션이 있으면(롱/숏), **반대 방향 신규 진입 금지**
- (G2) 유닛/익스포저 한도
  - 종목별 최대 4 units, 전체 최대 10 units (`capital_management_rules_v2.md` 참조)
- (G3) 숏 유니버스 제약
  - 숏 진입은 **허용된 숏 유니버스(예: 코스피 시총 Top10)** 에서만 가능
- (G4) 숏 명목 한도
  - 신규 숏(대주매도) 전, **대주매도 체결가격 기준 숏 명목 합**이 한도 이내인지 확인  
  - (백테스트 가정) 5.7억, (실운영 상위 정책) 6억 (`capital_management_rules_v2.md` 참조)

> 진입 후 손절/TS/피라미딩/강제청산(90거래일)은 별도 문서의 규칙을 따른다.

---

## 2) 진입 규칙은 “옵션 파라미터”로 관리
- 진입 규칙은 `entry_rule_type`이라는 옵션 파라미터로 관리한다.
- 후보군:
  - **Type A: Turtle(돈치안 브레이크아웃) 계열**
  - **Type B: EMA cross + Donchian breakout 계열 (신규)**
  - **Type C: Time-series momentum(모멘텀) 계열 (신규)**

---

## 3) 성능 검증 원칙(승률 35% 기준)
- 진입 규칙 옵션은 **KOSPI 시장에 대해** 백테스트 후 성능을 검토한다.
- 목표 기준:
  - **승률(Trade Win Rate) ≥ 35%** 달성 여부를 확인한다.
- 평가 구간은 `spec_updated_v2.md`에 정의된 기간을 따른다.
  - In-sample(백테스트): 2017~2020
  - OOS(전진분석): 2021~2024

> 실무 적용 권장: “In-sample 통과 + OOS에서 붕괴하지 않는지”를 함께 확인.

---

## 4) Type A — Turtle Trading(돈치안 브레이크아웃)
Type A는 돈치안 채널 돌파를 사용한다.  
돈치안 채널은 **과거 N일의 고가/저가 범위**를 이용한다.

> 구현 주의: 본 프로젝트에서의 “Strategy A”는 단일 윈도우가 아니라,
> **A.1(20) + A.2(PL 필터) + A.3(55 override)** 를 합친 **`A_TURTLE`** 를 의미한다.
> `A_20_PL`, `A_55`는 과거 호환/디버깅을 위해 남겨둔 *legacy* 옵션이며,
> 기본 스윕/리포트(LONG_ONLY)에서는 제거한다.

### 4.1 돈치안 채널 정의(룩어헤드 방지)
- 상단(Upper):
  - `DonchianHigh_N(T) = max( H_{T-N} ... H_{T-1} )`
- 하단(Lower):
  - `DonchianLow_N(T)  = min( L_{T-N} ... L_{T-1} )`

> 신호는 `T일 종가(C_T)` 기준으로 판단한다.

---

### 4.2 Type A.1 — 돈치안 브레이크아웃(20)
- 파라미터: `N = 20`

#### Long 진입 신호(롱)
- 조건:
  - `C_T >= DonchianHigh_20(T)`
- 실행:
  - `T+1 09:00 시장가`로 1 unit 진입

#### Short 진입 신호(숏, 대주)
- 조건:
  - `C_T <= DonchianLow_20(T)`
- 실행:
  - `T+1 09:00 시장가`로 1 unit 숏 진입(대주매도)

---

### 4.3 Type A.2 — PL 필터(Profit/Loss Filter)
- 목적: 직전 거래가 “수익”이었을 경우, **다음 진입에서 반대 방향 시도**를 억제한다.
- 정의:
  - 각 종목별로 “직전 종료된 트레이드”의 방향(`last_dir ∈ {long, short}`)과 손익(`last_pnl`)을 저장한다.
  - `last_pnl > 0`인 경우에 한해 필터가 활성화된다.

#### 필터 규칙
- 만약 `last_pnl > 0`이고, A.1에서 발생한 이번 진입 방향이 `last_dir`의 **반대 방향**이라면:
  - 해당 A.1 신호는 **무시(진입하지 않음)**

예시:
- 직전 거래가 **long이고 수익**이었다면 → 이번 A.1의 **short 진입 신호**는 무시
- 직전 거래가 **short이고 수익**이었다면 → 이번 A.1의 **long 진입 신호**는 무시

> 구현 옵션화: 본 필터는 이제 “Filter A”로 승격되어, A/B/C 등 **여러 진입 전략에 대해 ON/OFF** 할 수 있다.
> 단, Strategy A에서는 규칙 우선순위를 유지한다:
> **Donchian55(override) > PL Filter > Donchian20**

---

### 4.4 Type A.3 — 돈치안 브레이크아웃(55), PL 필터 무시
- 파라미터: `N = 55`
- 특징:
  - 돈치안 55 규칙에서는 **PL 필터(A.2)를 무시**하고,
  - 신호가 발생하면 **상시 원하는 방향으로 진입**한다(=신호 항상 실행).

#### Long 진입 신호
- `C_T >= DonchianHigh_55(T)` → `T+1 09:00 시장가`로 1 unit 롱

#### Short 진입 신호
- `C_T <= DonchianLow_55(T)` → `T+1 09:00 시장가`로 1 unit 숏

---

## 5) Type B — EMA(5/20) cross 이후 Donchian(10) breakout
Type B는 “추세 전환의 초기 신호(EMA cross)”를 먼저 확인한 뒤,
실제 추세 가속이 발생했을 때(Donchian10 breakout)만 진입한다.

### 5.1 EMA(5/20) 교차 정의
- GOLDEN cross: `EMA5_{T-1} <= EMA20_{T-1}` 그리고 `EMA5_T > EMA20_T`
- DEAD cross: `EMA5_{T-1} >= EMA20_{T-1}` 그리고 `EMA5_T < EMA20_T`

### 5.2 Donchian(10) breakout
- `DonchianHigh_10(T) = max( H_{T-10} ... H_{T-1} )`
- `DonchianLow_10(T)  = min( L_{T-10} ... L_{T-1} )`

### 5.3 진입 규칙
- Long:
  - 직전에 GOLDEN cross가 발생했고(=레짐이 long),
  - 그 이후의 어느 날 `C_T >= DonchianHigh_10(T)`이면 → `T+1 09:00 시장가` 진입
- Short:
  - 직전에 DEAD cross가 발생했고(=레짐이 short),
  - 그 이후의 어느 날 `C_T <= DonchianLow_10(T)`이면 → `T+1 09:00 시장가` 진입

> “이후”의 의미: 교차 발생일 당일에는 breakout이 동시에 발생해도 진입하지 않고,
> 다음 날부터 breakout을 관찰한다(lookahead 방지 + 과최적화 억제).

## 6) Type C — Time-series momentum(모멘텀) 진입
Type C는 63거래일 모멘텀을 사용해, 임계치 돌파 시에만 진입한다.

- 모멘텀 정의:
  - `MOM63_T = Close_T / Close_{T-63} - 1`
- 기본 진입:
  - Long: `MOM63`가 +5%를 상향 돌파
  - Short: `MOM63`가 -5%를 하향 돌파

> 실제 운영에서는 Filter B/C(대순환)를 함께 조합해 “시장>업종>종목” 흐름을 반영한다.

---

## Appendix) (Deprecated) 회귀 기반 Strategy B
아래 “정규화 기울기 + R²” 기반 Type B 규칙은 **이제 Deprecated**이며,
본 세션에서의 “Strategy B”는 *EMA cross + Donchian10 breakout*을 의미한다.

과거 실험 재현이 필요하면, 회귀 기반 전략B는 별도 브랜치/모듈로 분리해서 유지한다(본 세션 기본 코드에는 미포함).

### Appendix.B.1 왜 ‘기울기 정규화’가 필요한가(요약)
일반적인 선형 회귀 기울기(slope)는 단위가 `ΔPrice/ΔTime`이므로,
가격 레벨이 큰 종목이 기울기가 더 크게 보이는 왜곡이 발생한다.

- A 종목(1,000원): 하루 10원 상승 → slope = 10 (≈ +1%/day)
- B 종목(100,000원): 하루 100원 상승 → slope = 100 (≈ +0.1%/day)

단순 slope만 보면 B가 더 강해 보이지만, 수익률 관점의 추세는 A가 더 강하다.  
따라서 slope를 “가격 대비 비율”로 정규화한다.

### Appendix.B.2 정규화 기울기(Percentage Slope) 정의
- 회귀 구간: 최근 `W = 20`일
- 종속변수: `y_i = Close_{t-W+1+i}` (i=0..W-1)
- 독립변수: `x_i = i` (0..W-1)
- 회귀식: `y = a + b x`
  - `b`가 회귀 기울기(slope, KRW/day)

정규화:
- `NormalizedSlope_t = ( b / C_t ) * 100`

> 단위: “%/day”에 가까운 해석이 가능하다.

### Appendix.B.3 R-square 조건
- 동일 회귀에서 계산된 `R²`가:
  - `R² > 0.6`일 때만 신호 유효

### Appendix.B.4 60일 이동평균 방향 필터
- `MA60_t = SMA(Close, 60)` (기본은 단순이평; 필요 시 EMA로 변경 가능)
- 필터:
  - `C_t < MA60_t`이면 **long 금지**
  - `C_t > MA60_t`이면 **short 금지**

---

### Appendix.B.5 Type B.1 — 정규화 기울기(20) 0선 돌파 + R²
- 파라미터:
  - `W = 20`
  - `R2_threshold = 0.6`

#### Long 진입 신호
- 조건(동시에 만족):
  1) `NormalizedSlope_{T-1} <= 0` 그리고 `NormalizedSlope_T > 0`  (0 상향 돌파)
  2) `R²_T > 0.6`
  3) `C_T >= MA60_T` (Type B.2 필터)

- 실행:
  - `T+1 09:00 시장가`로 1 unit 롱

#### Short 진입 신호
- 조건(동시에 만족):
  1) `NormalizedSlope_{T-1} >= 0` 그리고 `NormalizedSlope_T < 0`  (0 하향 돌파)
  2) `R²_T > 0.6`
  3) `C_T <= MA60_T` (Type B.2 필터)

- 실행:
  - `T+1 09:00 시장가`로 1 unit 숏(대주매도)

---

## 6) 설정 예시(구현 관점)
아래는 YAML/JSON 스타일로 진입 규칙 옵션을 고정하는 예시이다(실제 키 이름은 구현에서 확정).

```yaml
entry:
  rule_type: "A_TURTLE"     # ["A_TURTLE", "B_EMA_CROSS_DC10", "C_TSMOM63"]
  # Filters (independent toggles)
  filters:
    A_PL: true            # Filter A
    B_CYCLE: true         # Filter B (standard: always ON)
    C_MARKET_CYCLE: false # Filter C

  # Rule-specific params are currently fixed in code for the toy run
  # (Donchian windows, EMA windows, momentum window/threshold). For full optimization,
  # expose them as config/CLI knobs later.
```

---

## 7) 구현 메모(중요)
- 진입 신호는 “후보 신호”이고, 실제 주문 생성은 섹션 1.3의 **게이트를 모두 통과**해야 한다.
- 진입 후 청산/피라미딩은 `capital_management_rules_v2.md`의 규칙을 따른다.
- Type A.2(PL 필터)의 “직전 거래”는 **종목 단위**로 관리하는 것을 기본으로 한다.


---

## Filters (A/B/C)

See `docs/filters.md`.

- Filter A: PL filter (block immediate flip after profitable trade)
- Filter B: Ticker EMA(5/20/40) cycle filter (Long 6-1-2, Short 3-4-5)
- Filter C: Market EMA(5/20/40) cycle filter (Long 6-1-2, Short 3-4-5)
