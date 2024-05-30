import matplotlib.pyplot as plt
import numpy as np

# Константы
R = 10 ** 4
C = 10 ** (-6)
tau = R * C
t_mod = 10 ** (-2)
i = 2

# Кол-во отсчетов
N = 10
# Создадим отсчеты
T = t_mod / N
nT = np.arange(N) * T
# Создаем входной сигнал
input_signal = np.heaviside(nT, 0)
# Создаем выходной сигнал
output_signal = 1 - np.exp(-nT / tau)

# Графики входного / выходного сигнала
def graphs1():
    fig, ax = plt.subplots()
    ax.plot(nT, input_signal, label='Входной сигнал')
    ax.plot(nT, output_signal, '--', label='Выходной сигнал')
    ax.set(xlabel='t', ylabel='U')
    ax.legend()
    plt.show()

# Форматирования числа по кол-ву знаков после запятой (i)
def format_num(num):
    return float(f'{num:.{i}g}')

# Метод билинейного Z-преобразования
def bilinear_z_transformation(signal, T):
    a0 = a1 = format_num(T / (2 * tau + T))
    b1 = format_num((2 * tau - T) / (2 * tau + T))
    print(f'Коэффициенты билинейного Z-преобразования: {a0=} {a1=} {b1=}')
    bilinear_output_signal = np.zeros_like(signal)
    for n in range(1, len(signal)):
        bilinear_output_signal[n] = a0 * signal[n] + a1 * signal[n - 1] + b1 * bilinear_output_signal[n - 1]
    return bilinear_output_signal

# Метод инвариантности импульсной характеристики
def invariance_impulse_response(signal, T):
    a0 = format_num(T / tau)
    b1 = format_num(np.exp(-T / tau))
    print(f'Коэффициенты инвариантной импульсной характеристики: {a0=} {b1=}')
    invariance_output_signal = np.zeros_like(signal)
    for n in range(1, len(signal)):
        invariance_output_signal[n] = a0 * signal[n] + b1 * invariance_output_signal[n - 1]
    return invariance_output_signal

# Графики входного / выходного сигнала / билинейного z-преобразования / инвариантности ИХ
def graphs2():
    model_bilinear_output_signal = bilinear_z_transformation(input_signal, T)
    model_invariance_output_signal = invariance_impulse_response(input_signal, T)
    fig, ax = plt.subplots()
    ax.plot(nT, input_signal, label='Входной сигнал')
    ax.plot(nT, output_signal, '--', label='Выходной сигнал')
    ax.plot(nT, model_bilinear_output_signal, ':', label='Билинейное Z-преобразование')
    ax.plot(nT, model_invariance_output_signal, '-.', label='Инвариантная ИХ')
    ax.set(xlabel='t', ylabel='U')
    ax.legend()
    plt.show()

# Расчет СКО (билинейного Z-преобразования и ИИХ) для разного шага дискретизации
def standard_deviation():
    bilinear_std = []
    invariance_str = []
    for T in [tau / divider for divider in (10, 100, 1000, 10_000)]:
        N = t_mod / T
        print(f'Шаг: {T=}\nКол-во отсчетов:{round(N)}')
        nT = np.arange(N) * T
        # Создаем входной сигнал для нового T
        input_signal = np.heaviside(nT, 0)
        # Создаем выходной сигнал для нового T
        output_signal = 1 - np.exp(-nT / tau)
        # Создаем модели
        model_bilinear_output_signal = bilinear_z_transformation(input_signal, T)
        model_invariance_output_signal = invariance_impulse_response(input_signal, T)
        # Считаем СКО
        bilinear_std.append(np.std(output_signal - model_bilinear_output_signal))
        invariance_str.append(np.std(output_signal - model_invariance_output_signal))
    return bilinear_std, invariance_str

# Графики СКО
def graphs3():
    T_range = [tau / divider for divider in (10, 100, 1000, 10_000)]
    bilinear_std, invariance_std = standard_deviation()
    fig, ax = plt.subplots()
    ax.plot(T_range, bilinear_std, 'o-', label='Билинейное Z-преобразование')
    ax.plot(T_range, invariance_std, 'x-', label='Инвариантная ИХ')
    ax.set(xlabel='T', ylabel='σ(T)', yscale='log', xscale='log')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # Построение графиков
    graphs1()
    graphs2()
    graphs3()
