import numpy as np
import matplotlib.pyplot as plt

# Параметры
G = 4 * np.pi**2  # гравитационная постоянная в а.е.^3 / (солнечная масса * год^2)
m1 = 1.1   # масса Alpha Centauri A (в массах Солнца)
m2 = 0.123 # масса Alpha Centauri B (в массах Солнца)
M = m1 + m2

# Яркости (пропорциональны массам)
L1 = m1
L2 = m2

# Начальные условия (положение относительно центра масс)
r1_0 = np.array([-m2 / M, 0])  # начальная позиция A
r2_0 = np.array([m1 / M, 0])   # начальная позиция B
v1_0 = np.array([0, -np.sqrt(G * m2 / np.linalg.norm(r1_0 - r2_0))])
v2_0 = np.array([0, np.sqrt(G * m1 / np.linalg.norm(r1_0 - r2_0))])

# Параметры интеграции
dt = 0.001  # шаг по времени (лет)
t_max = 10  # всего лет
N = int(t_max / dt)

# Массивы
r1 = np.zeros((N, 2))
r2 = np.zeros((N, 2))
v1 = np.zeros((N, 2))
v2 = np.zeros((N, 2))
t = np.zeros(N)
brightness = np.zeros(N)

# Начальные значения
r1[0] = r1_0
r2[0] = r2_0
v1[0] = v1_0
v2[0] = v2_0
brightness[0] = L1 + L2

# Интегрирование
for i in range(N - 1):
    r = r2[i] - r1[i]
    dist = np.linalg.norm(r)
    F = G * m1 * m2 / dist**3 * r

    v1[i+1] = v1[i] + F / m1 * dt
    v2[i+1] = v2[i] - F / m2 * dt
    r1[i+1] = r1[i] + v1[i+1] * dt
    r2[i+1] = r2[i] + v2[i+1] * dt
    t[i+1] = t[i] + dt

    # Проверка затмения (вдоль оси Y → сравниваем координату X)
    if abs(r1[i+1][0] - r2[i+1][0]) < 0.01:
        if r1[i+1][1] > r2[i+1][1]:
            brightness[i+1] = L1  # A закрывает B
        else:
            brightness[i+1] = L2  # B закрывает A
    else:
        brightness[i+1] = L1 + L2  # обе видны

# Нормализация яркости
brightness /= (L1 + L2)

# Построение кривой блеска
plt.figure(figsize=(10, 4))
plt.plot(t, brightness, color='orange')
plt.title("Синтетическая кривая блеска системы Alpha Centauri A/B")
plt.xlabel("Время (лет)")
plt.ylabel("Относительная яркость")
plt.grid(True)
plt.tight_layout()
plt.show()
