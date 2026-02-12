import pandas as pd
import matlab.engine
import os
from pathlib import Path

def plot_with_matlab():
    # 1. 경로 설정 (스크립트 위치 기준)
    # 현재 파일(scripts/...)의 부모의 부모 폴더(root) 밑의 outputs 폴더를 찾습니다.
    base_path = Path(__file__).resolve().parent.parent
    equity_file = base_path / "outputs" / "equity_curve.csv"
    trades_file = base_path / "outputs" / "trades.csv"

    # 파일 존재 여부 확인
    if not equity_file.exists():
        print(f"에러: 파일을 찾을 수 없습니다 -> {equity_file}")
        return

    # 2. 데이터 로드
    df = pd.read_csv(equity_file)
    nav_data = df['nav'].tolist()
    
    # 날짜 데이터도 MATLAB으로 보내기 위해 처리 (선택 사항)
    dates = df['date'].astype(str).tolist()

    # 3. MATLAB 엔진 시작
    print("MATLAB 2025b 엔진을 시작합니다...")
    eng = matlab.engine.start_matlab()

    # 4. 데이터 전송 및 시각화
    # 리스트를 MATLAB double형 배열로 변환
    nav_mat = matlab.double(nav_data)
    
    eng.workspace['nav'] = nav_mat
    eng.eval("figure('Name', 'TrendFollowing Analysis', 'Color', 'w');", nargout=0)
    
    # MATLAB 스타일의 고급 그래프 작성
    eng.eval("plot(nav, 'LineWidth', 2, 'Color', [0 0.447 0.741]);", nargout=0)
    eng.eval("grid on; hold on;", nargout=0)
    
    # 제목 및 라벨
    eng.eval("title('Backtest Result: Equity Curve', 'FontSize', 14);", nargout=0)
    eng.eval("xlabel('Trading Days');", nargout=0)
    eng.eval("ylabel('NAV (KRW)');", nargout=0)
    
    # 깔끔한 출력을 위해 Y축 지수 표기법 조정
    eng.eval("ax = gca; ax.YAxis.Exponent = 0; datetick('x', 'yyyy-mm', 'keeplimits');", nargout=0)

    print("MATLAB 창이 활성화되었습니다.")
    input("분석을 마친 후 Enter를 누르면 엔진이 종료됩니다...")
    eng.quit()

if __name__ == "__main__":
    plot_with_matlab()