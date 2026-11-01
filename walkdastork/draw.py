from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd

modelName = ["ppo_300k_m1", "ppo_300k_m2"]

for v in range(len(modelName)):
    combined_df = []
    for i in range(8):
        df = pd.read_csv(
            f"logs/{modelName[v]}/{i}.monitor.csv", skiprows=2, header=None
        ).values.tolist()
        combined_df.extend(df)

    modelName[v] = [modelName[v], combined_df]

# Matplotlib 시각화
plt.figure(figsize=(12, 6))

for i in modelName:
    # 1. 반드시 에피소드 종료 시간(t, 인덱스 2) 순서로 정렬해야 시간 흐름대로 줄을 섭니다!
    i[1].sort(key=itemgetter(2))

    rewards = [row[0] for row in i[1]]

    # 2. 각 판의 스텝 수(row[1])를 누적해서 더해야 전체 학습 스텝(300k)이 됩니다!
    steps = []
    current_step = 0
    for row in i[1]:
        current_step += row[1]  # 이전 판 스텝 수에 이번 판 스텝 수를 계속 더함
        steps.append(current_step)

    # 그래프 그리기
    plt.plot(
        steps,
        rewards,
        alpha=0.8,
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label=i[0],
    )

plt.title("Walk The Stork - Training Learning Curve", fontsize=14)
plt.xlabel("Total Cumulative Steps", fontsize=12)  # 누적 스텝
plt.ylabel("Episode Reward (r)", fontsize=12)

plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("logs/learningcurve.png")
plt.show()