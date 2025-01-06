import time

import matplotlib.pyplot as plt
import numpy as np


def step_rod_simulation(rod_t, rod_result, imp_scratch, rod_k, dt, t):
    # Adapted from Cen, Hoppe, and Gu (2016)
    # Read more: https://pubs.aip.org/aip/adv/article/6/9/095305/882010/Fast-and-accurate-determination-of-3D-temperature

    # rod_t is the starting rod T(t=0)
    # rod_result is the buffer for the explicit phase and also stores the b values for the implicit phases,
    #   converted into the result written back to rod_t
    # imp_scratch stores the modified superdiagonal values for the matrix calculations in the implicit step
    # rod_k is a precomputed constants rod

    sz = len(rod_t)

    end_left = rod_t[0]
    end_right = rod_t[sz - 1]

    # explicit step
    for idx, temp in enumerate(rod_t):
        temp_left = rod_t[idx - 1] if idx > 0 else end_left
        temp_right = rod_t[idx + 1] if idx < sz - 1 else end_right
        rod_result[idx] = temp + (temp_left - 2 * temp + temp_right) * rod_k[idx]

    # implicit step - Thomas' algorithm
    # Adapted from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Method
    imp_scratch[0] = -rod_k[0] / (1 + 2 * rod_k[0])
    rod_result[0] = (rod_result[0] + rod_k[0] * end_left) / (1 + 2 * rod_k[0])

    for idx in range(1, sz):
        k = rod_k[idx]
        scr = imp_scratch[idx - 1]
        # a,c = -k
        # b = 1 + 2k
        recip = 1 / (1 + (2 + scr)*k)
        imp_scratch[idx] = -k * recip
        rod_result[idx] = (rod_result[idx] + k*rod_result[idx - 1]) * recip

    rod_result[sz - 1] -= imp_scratch[sz - 1] * end_right

    for idx in range(sz - 2, -1, -1):
        rod_result[idx] -= imp_scratch[idx] * rod_result[idx + 1]

    # write result to rod_t, adding sources as well
    absorb_scale = min(t / 20, 1)
    absorb = -2 * absorb_scale
    gen = 0.1
    for idx, r in enumerate(rod_t):
        s = 0
        # s += gen * r * dt
        # if idx % 4 == 0:
        #     s += absorb * r * dt
        rod_t[idx] = rod_result[idx] + s

def main():
    length = 10
    xes = range(length)
    dt = 1 / 20
    steps = 20

    diff_scale = 20
    #random.seed(42)

    initial_rod = [x * (length - 1 - x) + 100 for x in xes]
    #initial_rod = [100 for _ in xes]
    #rod_diff = [((x + 1) / length) * diff_scale for x in xes]
    rod_diff = [1 * diff_scale for _ in xes]
    rod_k = [diff * dt * 0.5 for diff in rod_diff] # diffusivity * dt / 2 / spacing^2 (spacing = 1)

    results = []

    rod_t = [e for e in initial_rod]
    rod_result = [0 for _ in initial_rod]
    imp_scratch = [0 for _ in initial_rod]

    results.append(initial_rod)
    t = 0

    start = time.perf_counter()
    for i in range(steps):
        t += dt
        step_rod_simulation(rod_t, rod_result, imp_scratch, rod_k, dt, t)
        results.append([x for x in rod_t])
    end = time.perf_counter()
    sim_time = end - start

    print()
    print(f"Simulation time       : {sim_time:0.8f} s")
    print(f"Simulated time        : {dt * steps:0.8f} s")
    print(f"Average time per tick : {sim_time / steps:0.8f} s")
    print(f"Time per tick to beat : {dt:0.8f} s")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Heat equation sim")
    ax.set_xlabel("Position")
    ax.set_ylabel("Time")
    ax.set_zlabel("Heat value")

    plot_times = [x for x in range(0, steps + 1)]
    if steps not in plot_times:
        plot_times.append(steps)

    X, Y = np.meshgrid(xes, plot_times)
    Z = np.array(results, dtype="float64")
    ax.plot_wireframe(X, Y, Z)
    plt.show()

if __name__ == "__main__":
    main()