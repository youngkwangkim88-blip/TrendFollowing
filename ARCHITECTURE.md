# Python 모듈/클래스 아키텍처(초안)

본 토이 리포지토리는 “백테스트 코어”와 “라이브 브로커 연동”을 분리하는 구조를 취합니다.

## Backtest Layer
- `invest_v2.prep`
  - 지표(ATR/Donchian/Regression/MA) 일괄 계산
- `invest_v2.strategy.entry_rules`
  - 진입 규칙 옵션(Type A/B)
- `invest_v2.backtest.engine.SingleSymbolBacktester`
  - 1종목 toy 백테스트 실행 엔진
  - 이벤트 순서: 월초 이자 차감 → 오픈 체결 → 스탑/TS → EOD 신호 → 스냅샷
- `invest_v2.backtest.accounting.Account`
  - CMA/트레이딩/담보 현금 bucket
  - 매수 0% / 매도 0.3% 비용
  - 월 1회 대주이자 차감

## Live Layer (Stub)
- `invest_v2.live.broker_kis_stub`
  - 향후 KIS API 구현을 위한 인터페이스 스텁
  - 라이브에서는 브로커 스냅샷을 소스 오브 트루스로 두고
    내부 계산치와 reconcile + 알림(델타 임계치)을 붙이는 구조가 적합

## 문서 기반 구현 포인트
- 실행/체결 가정(익일 09:00 시장가) 및 평가 구간: `docs/spec_updated_v2.md`
- 유닛/손절/TS/피라미딩/틱 규칙: `docs/capital_management_rules_v2.md`
- 진입 옵션(Type A/B): `docs/entry_rules.md`
- 백테스트 회계 단순화 및 라이브 reconcile 방향: `docs/accounting_rules_backtest.md`
