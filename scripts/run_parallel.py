import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# 1. 실행할 모델과 반복 횟수 설정
models = ["CNN-LSTM", "Encoder", "Transformer"]
runs = 5

# 2. i7-14700K의 위력: 15개 작업을 동시에 실행!
# (만약 RAM 용량이 부족해서 뻗는다면 이 숫자를 8이나 10으로 살짝 내려주세요)
MAX_CONCURRENT_PROCESSES = 15  

def run_isolated_process(task_info):
    model, run_id = task_info
    
    # 텐서플로가 스레드를 적당히 나눠 쓰도록 설정 (코어 독식 방지)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["TF_NUM_INTRAOP_THREADS"] = "2"
    env["TF_NUM_INTEROP_THREADS"] = "2"
    
    # Codex가 수정한 stage4.py를 외부 프로세스로 호출
    # (참고: 이 방식이 작동하려면 stage4.py 내부가 단일 실행을 받도록 수정되어 있어야 합니다.
    # 만약 현재 stage4.py가 그냥 통째로 다 도는 구조라면, 이 스크립트 대신
    # stage4.py 내부의 for문을 다시 ThreadPoolExecutor로 감싸는 것이 낫습니다.)
    cmd = f"python AI_pivot_point_stage4.py --target-model {model} --run-id {run_id}"
    
    print(f"🚀 [시작] {model} - Run {run_id} (코어 할당됨)")
    subprocess.run(cmd, shell=True, env=env)
    print(f"✅ [완료] {model} - Run {run_id}")

def main():
    tasks = [(m, r+1) for m in models for r in range(runs)]
    
    print(f"🔥 i7-14700K 멀티코어 풀가동 모드 시작!")
    print(f"총 {len(tasks)}개의 작업을 독립 프로세스로 병렬 실행합니다.")
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
        executor.map(run_isolated_process, tasks)
        
    print("🎉 모든 병렬 학습이 빛의 속도로 종료되었습니다!")

if __name__ == "__main__":
    main()