import socket
import numpy as np
from stable_baselines3 import PPO

socketPort = 9999
modelName = [
    "ppo_300k_m1",
    "ppo_300k_m2"
]
modelName = modelName[1]

def run_ai_server():
    # 모델 로드 (학습이 완료되어 ppo_walk_the_stork.zip 파일이 있어야 합니다)
    try:
        model = PPO.load(modelName)
    except Exception as e:
        print(f"[에러] 학습된 모델 파일을 찾을 수 없습니다: {e}")
        return

    print(modelName)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(("127.0.0.1", socketPort))
    except OSError:
        print(f"[에러] {socketPort}번 포트가 이미 사용 중입니다. 기존 서버를 종료하거나 포트를 변경해 주세요.")
        return

    server.listen(5)
    
    # [핵심] 1초마다 타임아웃을 걸어주어 Ctrl + C가 언제든 즉시 먹히도록 설정
    server.settimeout(1.0)
    
    print(f"=== 황새 오래걷기 AI 제어 서버 가동 (Port: {socketPort}) ===")
    print("대기 중... 수정한 stork_ai.swf 파일을 Flash Player로 실행하세요!")
    print("(종료하려면 언제든 Ctrl + C를 누르세요)")

    try:
        while True:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue  # 1초마다 루프를 돌며 Ctrl+C 입력을 감지함
            
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"\n[연결됨] SWF 클라이언트가 접속했습니다: {addr}")

            buffer = ""
            step_count = 0
            conn.settimeout(2.0)
            
            try:
                while True:
                    try:
                        data = conn.recv(1024)
                    except socket.timeout:
                        # 2초 동안 게임에서 데이터가 안 오면 연결 끊긴 것으로 간주
                        break
                        
                    if not data:
                        break
                    
                    buffer += data.decode("utf-8", errors="ignore")
                    
                    while "\x00" in buffer:
                        msg, buffer = buffer.split("\x00", 1)
                        if "," in msg:
                            try:
                                vals = [float(x) for x in msg.split(",")]
                                obs = np.array(vals, dtype=np.float32)
                                
                                # AI 예측
                                action, _ = model.predict(obs, deterministic=True)
                                
                                # SWF로 행동 전송
                                conn.sendall(f"{int(action)}\x00".encode("utf-8"))
                                
                                step_count += 1
                                if step_count % 100 == 0:
                                    print(f"진행 프레임: {step_count:5d} | 기울기: {vals[0]:6.2f}° | 난이도(bi): {vals[3]:.3f}")
                            except ValueError:
                                pass
            except (ConnectionResetError, socket.timeout):
                pass
            finally:
                conn.close()
                print("[연결 종료] 플래시 게임 재접속 대기 중...")
                
    except KeyboardInterrupt:
        print("\n[서버 종료] 사용자에 의해 서버가 안전하게 중지되었습니다.")
    finally:
        server.close()

if __name__ == "__main__":
    run_ai_server()