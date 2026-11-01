from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd

modelName = ["ppo_300k"]
bin_size = 8192  # 평균을 낼 스텝 구간 크기 (8192 = PPO 1회 업데이트 주기)

for v in range(len(modelName)):
    combined_df = []
    for i in range(8):
        df = pd.read_csv(
            f"logs/{modelName[v]}/{i}.monitor.csv", skiprows=2, header=None
        ).values.tolist()
        combined_df.extend(df)

    modelName[v] = [modelName[v], combined_df]

plt.figure(figsize=(12, 6))

for i in modelName:
    # 1. 에피소드 종료 시간(t, 인덱스 2) 순서로 정렬
    i[1].sort(key=itemgetter(2))

    raw_steps = []
    raw_rewards = []
    
    avg_steps = []
    avg_rewards = []
    
    current_step = 0
    current_bin_limit = bin_size
    bin_rewards = []  # 현재 구간에 들어온 보상들을 임시로 모으는 리스트

    # 2. 누적 스텝을 계산하면서 동시에 bin_size 단위로 평균 계산
    for row in i[1]:
        r = row[0]
        l = row[1]
        current_step += l
        
        raw_steps.append(current_step)
        raw_rewards.append(r)
        
        bin_rewards.append(r)
        
        # 누적 스텝이 구간(bin_size)을 넘어설 때마다 평균을 구해 저장
        if current_step >= current_bin_limit:
            avg_steps.append(current_step)
            avg_rewards.append(sum(bin_rewards) / len(bin_rewards))
            bin_rewards = []  # 바구니 비우기
            current_bin_limit += bin_size

    # 마지막에 남은 자투리 구간 처리
    if bin_rewards:
        avg_steps.append(current_step)
        avg_rewards.append(sum(bin_rewards) / len(bin_rewards))

    # [배경] 원본 진동 데이터는 연하게(alpha=0.25) 뒤에 깔아줌
    plt.plot(raw_steps, raw_rewards, alpha=0.25, color="gray", linestyle="-")

    # [메인] 구간 평균 데이터를 진하게 그리기
    plt.plot(
        avg_steps,
        avg_rewards,
        alpha=0.9,
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label=f"{i[0]} ({bin_size} steps avg)",
    )

plt.title("Walk The Stork - Training Learning Curve (Step-Binned Average)", fontsize=14)
plt.xlabel("Total Cumulative Steps", fontsize=12)
plt.ylabel("Average Episode Reward", fontsize=12)

plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("logs/learningcurve_avg.png")
plt.show()