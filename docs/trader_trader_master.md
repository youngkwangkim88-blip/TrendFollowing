# Trader / TraderMaster 설계 (v2 디버깅 & 가시성 강화)

## 1. 배경: "고스트 타점"(ghost entry) 문제

피라미딩(불타기)을 사용하는 추세추종 전략에서는 **최초 진입 가격**과 **포지션의 평균단가(수량가중 평균)** 가 서로 다른 것이 정상입니다.

그런데 백테스트/리포트가 다음처럼 기록되면 문제가 발생합니다.

- `entry_date`는 **최초 진입일**로 유지
- `entry_price`는 피라미딩이 끝난 뒤 **최종 평균단가**로 덮어쓰기

이 경우, 차트에서는 `entry_date` 시점의 캔들(OHLC) 고가보다 높은 곳에 진입 마커가 찍히는 **"허공 진입"** 이 발생합니다.

중요: 이는 보통 **데이터 오류가 아니라 리포팅(로그 스키마) 불일치**입니다.

---

## 2. 목표

1. 포지션을 합쳐서 관리(평균단가/총수량/유닛 수)하면서도,
2. 주문/체결 단위(각 피라미딩 체결)의 가격/날짜를 잃지 않고,
3. 사람이 디버깅 가능한 형태로 리포트/플롯(abc.html)에서 정확히 보이도록 한다.

---

## 3. 핵심 컨셉

### 3.1 Trader (심볼 단위 포지션 매니저)

`Trader`는 **단일 종목(symbol)** 의 포지션 라이프사이클을 관리합니다.

- ENTRY / PYRAMID / EXIT 체결을 **lot(체결 단위)로 누적**
- `first_entry_date`, `first_entry_price`는 절대 변경하지 않음
- `avg_entry_price`는 lot 기반으로 계산 (피라미딩 시 변동)
- `fills`(체결 이벤트 로그) + `trades`(포지션 단위 요약 로그) 생성

또한 디버깅을 위해 `Trader.how_did_you_trade()` 를 제공하여 사람이 읽을 수 있는 텍스트 로그를 뽑을 수 있습니다.

### 3.2 TraderMaster (계좌/트레이더 오케스트레이션)

`TraderMaster`는 계좌(원화/달러/CMA 등)와 여러 Trader를 소유하는 상위 컨테이너입니다.

현재 v2 toy 백테스트에서는:

- 단일 KRW 계좌(`Account`)를 사용
- 심볼별 `Trader`를 생성/제공

향후 라이브 트레이딩에서 **브로커 계좌 조회값과 내부 장부의 reconcile** 기능을 이 계층에 확장하는 것이 자연스럽습니다.

---

## 4. 로그/산출물 스키마

### 4.1 `trades.csv` (포지션 단위)

한 포지션(진입~전량청산)을 1행으로 기록합니다.

필드(핵심):

- `entry_date`, `entry_price`: **최초 진입일/가격** (차트 마커 기준)
- `avg_entry_price`: 피라미딩 포함 **최종 평균단가**
- `num_entries`: ENTRY + PYRAMID 체결 횟수
- `shares`, `exit_date`, `exit_price`, `realized_pnl`, `exit_reason`

### 4.2 `fills.csv` (체결 이벤트 단위)

각 체결(ENTRY/PYRAMID/EXIT)을 날짜/가격/수량으로 기록합니다.

- `action ∈ {ENTRY, PYRAMID, EXIT}`
- `date`, `price`, `shares`, `reason`
- `avg_entry_price_after`, `pos_units_after` 등 상태 스냅샷 포함

### 4.3 `trader_report.txt`

`Trader.how_did_you_trade()` 결과를 저장한 텍스트 로그입니다.

---

## 5. abc.html(Plotly) 가시성 정책

`abc.html`은 다음 우선순위를 사용합니다.

1. `fills.csv`가 있으면 **fills 기반으로 entry/pyramiding/exit 마커를 표시** (가장 정확)
2. 없으면 `trades.csv`/`equity_curve.csv` 기반의 레거시 방식(유닛 diff)으로 표시

이로써 피라미딩 평균단가가 최초 진입일에 찍히는 문제를 구조적으로 방지합니다.
