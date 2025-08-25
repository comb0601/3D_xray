import numpy as np

# 파일 경로
# txt_path = 'botleft(6679).txt'
# npy_save_path = '6679_2d_converted.npy'
txt_path = 'data/calibration/6681/topleft(6681).txt'
npy_save_path = 'data/calibration/6681/6681_2d.npy'

# 데이터 파싱
data = []
with open(txt_path, 'r') as file:
    for line in file:
        if line.strip() == "" or line.startswith("View#"):
            continue
        parts = line.strip().split()
        view, bead, u, v = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
        data.append([view, bead, u, v])

data = np.array(data)

# view 개수와 bead 개수 파악
view_count = int(np.max(data[:, 0]) + 1)
bead_count = int(np.max(data[:, 1]) + 1)

# (view, bead, 2) 형태로 변환
formatted = np.zeros((view_count, bead_count, 2), dtype=np.float32)
for row in data:
    v, b = int(row[0]), int(row[1])
    formatted[v, b] = row[2:]

# npy로 저장
np.save(npy_save_path, formatted)
print(f"✅ 저장 완료: {npy_save_path}")
