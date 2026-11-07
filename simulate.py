"""Simulate the vertical motion of a table on a spring and damper."""

from dataclasses import dataclass

import numpy as np


MASS_KG = 1.0
NATURAL_FREQUENCY_HZ = 13.5
ANGULAR_FREQUENCY_RAD_S = 2 * np.pi * NATURAL_FREQUENCY_HZ
SPRING_N_M = MASS_KG * ANGULAR_FREQUENCY_RAD_S**2
DAMPING_NS_M = 3.0
DISTURBANCE_AMPLITUDE_N = 0.4
DISTURBANCE_END_S = 5.0
RMS_START_S = 3.0
DURATION_S = 8.0
STEP_S = 0.001


@dataclass
class Trace:
    time_s: np.ndarray
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    energy_j: np.ndarray


def disturbance_force(time_s):
    """Return the sinusoidal force, switched off after five seconds."""
    return np.where(
        time_s < DISTURBANCE_END_S,
        DISTURBANCE_AMPLITUDE_N * np.sin(ANGULAR_FREQUENCY_RAD_S * time_s),
        0.0,
    )


def acceleration(displacement_m, velocity_m_s, force_n):
    return (force_n - DAMPING_NS_M * velocity_m_s - SPRING_N_M * displacement_m) / MASS_KG


def advance(displacement_m, velocity_m_s, start_force_n, middle_force_n, end_force_n):
    """Advance displacement and velocity one fixed step using RK4."""
    v1 = velocity_m_s
    a1 = acceleration(displacement_m, v1, start_force_n)
    v2 = velocity_m_s + STEP_S * a1 / 2
    a2 = acceleration(displacement_m + STEP_S * v1 / 2, v2, middle_force_n)
    v3 = velocity_m_s + STEP_S * a2 / 2
    a3 = acceleration(displacement_m + STEP_S * v2 / 2, v3, middle_force_n)
    v4 = velocity_m_s + STEP_S * a3
    a4 = acceleration(displacement_m + STEP_S * v3, v4, end_force_n)
    next_displacement = displacement_m + STEP_S * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    next_velocity = velocity_m_s + STEP_S * (a1 + 2 * a2 + 2 * a3 + a4) / 6
    return next_displacement, next_velocity


def simulate():
    time_s = np.arange(round(DURATION_S / STEP_S) + 1) * STEP_S
    force_n = disturbance_force(time_s)
    midpoint_force_n = disturbance_force(time_s[:-1] + STEP_S / 2)
    displacement_m = np.zeros_like(time_s)
    velocity_m_s = np.zeros_like(time_s)

    for index in range(len(time_s) - 1):
        displacement_m[index + 1], velocity_m_s[index + 1] = advance(
            displacement_m[index],
            velocity_m_s[index],
            force_n[index],
            midpoint_force_n[index],
            force_n[index + 1],
        )

    energy_j = (MASS_KG * velocity_m_s**2 + SPRING_N_M * displacement_m**2) / 2
    return Trace(time_s, displacement_m, velocity_m_s, acceleration(displacement_m, velocity_m_s, force_n), energy_j)


def acceleration_rms(trace):
    window = (trace.time_s >= RMS_START_S) & (trace.time_s < DISTURBANCE_END_S)
    return float(np.sqrt(np.mean(trace.acceleration_m_s2[window] ** 2)))


if __name__ == "__main__":
    baseline = simulate()
    print(f"OFF acceleration RMS ({RMS_START_S:g}–{DISTURBANCE_END_S:g} s): {acceleration_rms(baseline):.6f} m/s²")
    print(f"OFF mechanical energy at 5 s and 8 s: {baseline.energy_j[5000]:.9g}, {baseline.energy_j[-1]:.9g} J")
