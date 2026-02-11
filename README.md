# 투자 자동화 시스템 구축 v2 — Toy Backtest (005930)

이 압축 파일은 **백테스트(토이) + 향후 한국투자(KIS) API 연계**를 염두에 둔 최소 아키텍처 스캐폴딩입니다.

## 포함 기능(토이)
- 1종목(기본: **005930**)에 대해
  - 진입 규칙(Type A/B 옵션)
  - 유닛 사이징(ATR10 기반)
  - 손절/트레일링 스탑(틱 반영, 갭/장중 터치 체결가 규칙)
  - 피라미딩
  - 숏 명목 한도(대주매도 체결가 기준)
  - 대주 이자(연 4.5%, **월 1회 다음달 첫 거래일 차감**)
  - NAV(Equity curve) 스냅샷
을 포함한 “한 번에 실행되는” toy backtest 스크립트를 제공합니다.

## 실행
### 의존성
```bash
pip install -r requirements.txt
```

### 샘플 데이터로 실행
```bash
python scripts/toy_005930_backtest.py --csv data/sample_005930.csv --entry-rule A_20_PL --plot
```

결과는 `outputs/`에 생성됩니다.

## 문서
`docs/`에 spec/자금관리/진입규칙/회계(백테스트) 문서를 포함합니다.
