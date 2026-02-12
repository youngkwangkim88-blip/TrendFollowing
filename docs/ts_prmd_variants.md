# TS / 피라미딩 확장 (TS.A/B/C, PRMD.A/B)

이 문서는 백테스트(일봉) 기준으로 **Trailing Stop(=TS)**와 **피라미딩(PRMD)** 정책을 확장한 내용을 정리합니다.

## 공통 실행 가정

- 의사결정: T일 **종가** 기준 신호 평가 → T+1 09:00 **시장가** 진입/청산
- 장중 스탑/TS: OHLC 기반 근사
  - 오버나잇 갭 트리거: 익일 시가(O) 체결
  - 장중 터치: 해당 레벨(틱 단위) 체결
- 초기 손절: `± stop_atr_mult * ATR10` (기본 2.0배) **항상 적용**

---

## TS 타입

### TS.A (Percent trailing stop)
- 기존 % 기준 TS를 **TS.A**로 명명
- 활성화 조건(롱): `Hmax >= (1+ts_activate_gain)*X` (기본 20% 상승)
- TS 레벨(롱): `max((1+ts_floor_gain)*X, (1-ts_trail_frac)*Hmax)`
- 숏은 대칭적으로 `Lmin` 기반
- 피라미딩 발생 시: Hmax/Lmin 리셋 + TS 비활성화 후 재활성화 조건을 다시 충족해야 함

### TS.B (EMA5/20 Cross exit)
- **롱:** EMA5/EMA20 **Dead-cross** 발생 시(전일 EMA5>=EMA20 && 당일 EMA5<EMA20) → 다음날 시가 청산
- **숏:** EMA5/EMA20 **Golden-cross** 발생 시 → 다음날 시가 청산
- 가격 레벨 기반이 아니라 **close-based exit** 이므로, 스탑(ATR 기반)과 함께 동작

### TS.C (Darvas/Box S/R trailing stop)
- Donchian 기반 S/R을 Darvas Box의 간단 근사로 사용
- 박스 최소 크기: 기본 20일 (설정: `ts_c_box_window`)
- **롱:** trailing support = Donchian Low(window)
- **숏:** trailing resistance = Donchian High(window)
- trailing 성질을 유지하기 위해(기본값):
  - 롱 support는 **단조 증가**(max 누적)
  - 숏 resistance는 **단조 감소**(min 누적)
- 장중 트리거는 스탑과 동일한 방식(GAP/TOUCH)으로 체결 근사

---

## 피라미딩(PRMD) 타입

### PRMD.A (Percent pyramiding)
- 기존 % 기준 피라미딩을 **PRMD.A**로 명명
- **롱:** `Close >= (1+pyramid_trigger)*avg_price` (기본 15%) → 다음날 시가 1 unit 추가
- **숏:** `Close <= (1-pyramid_trigger)*avg_price` → 다음날 시가 1 unit 추가

### PRMD.B (Darvas/Box breakout pyramiding + cooldown)
- Donchian(20) 박스 상단/하단 돌파로 피라미딩
- **롱:** `Close >= DonchianHigh(window)`
- **숏:** `Close <= DonchianLow(window)`
- **쿨다운:** 피라미딩 1회 실행 후 `prmd_b_cooldown_days` (기본 5일) 동안 추가 피라미딩 금지

---

## 구현 상의 Exit Reason 표기

- ATR 기반 스탑: `STOP_LOSS_GAP`, `STOP_LOSS_TOUCH`
- TS.A: `TS_A_GAP`, `TS_A_TOUCH`
- TS.C: `TS_C_GAP`, `TS_C_TOUCH`
- TS.B: `TS_B_DEAD_CROSS`, `TS_B_GOLDEN_CROSS`

---

## 데이터 결측/이상치 처리(중요)

- 일부 패널 CSV에서 O/H/L이 0으로 들어오는 케이스가 확인됨(예: 005930 2018-04-30 등)
- 0 값은 실제 가격이 아니므로, 다음 규칙으로 **sanitize** 수행:
  - OHLC의 0/음수는 결측으로 간주
  - close가 결측인 행은 제거
  - open/high/low 결측은 close로 대체
  - high/low는 open/close와의 일관성 강제

