from pykrx import stock
import pandas as pd
from datetime import datetime

# 1. 설정값 (조회 기간 및 종목명)
start_date = "20060101"
end_date = datetime.now().strftime("%Y%m%d")
target_name = "코스피200" # 또는 "KOSPI 200"

# 2. KOSPI 200 지수/선물 관련 티커 확인
# 선물 데이터의 경우 stock.get_index_ohlcv_by_date를 활용하여 
# '코스피 200' 지수 혹은 관련 파생 상품 데이터를 접근할 수 있습니다.
# 실제 '선물' 종목 자체를 추출하려면 아래와 같이 진행합니다.

# KOSPI 200 지수 데이터 가져오기 (지수 기준)
df = stock.get_index_ohlcv_by_date(start_date, end_date, "1028") # 1028은 KOSPI 200 지수 코드

# 만약 '선물' 종목의 구체적인 OHLCV가 필요하다면 (연결선물 등)
# pykrx의 get_market_ohlcv는 주식/ETF 위주이므로 지수 API를 주로 활용합니다.
# 아래는 지수 데이터를 정리하여 CSV로 저장하는 과정입니다.

if not df.empty:
    # 컬럼명 영문 변환 (o, h, l, c, v)
    df = df.rename(columns={
        '시가': 'o',
        '고가': 'h',
        '저가': 'l',
        '종가': 'c',
        '거래량': 'v'
    })
    
    # 종목코드 컬럼 추가 (KOSPI200 지수 코드: 1028)
    df['ticker'] = '1028'
    
    # 인덱스(Date)를 컬럼으로 빼기
    df.index.name = 'date'
    df = df.reset_index()
    
    # 필요한 컬럼 순서 재배치
    df = df[['ticker', 'date', 'o', 'h', 'l', 'c', 'v']]
    
    # 3. CSV 저장
    file_name = f"kospi200_futures_{end_date}.csv"
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"성공적으로 저장되었습니다: {file_name}")
    print(df.head())
else:
    print("데이터를 불러오지 못했습니다.")