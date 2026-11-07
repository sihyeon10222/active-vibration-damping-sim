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
MODES = ("off", "pid", "damping")
FORCE_LIMIT_N = 0.8
PID_KP_N_M = 800.0
PID_KI_N_M_S = 120.0
PID_KD_NS_M = 7.0
ACTIVE_DAMPING_NS_M = 12.0
GRAVITY_M_S2 = 9.80665
SENSOR_NOISE_M_S2 = 0.05
SENSOR_SEED = 42


@dataclass
class Trace:
    time_s: np.ndarray
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    sensor_raw_m_s2: np.ndarray
    control_force_n: np.ndarray
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


def limit_force(force_n):
    return min(FORCE_LIMIT_N, max(-FORCE_LIMIT_N, force_n))


def control_force(mode, displacement_m, velocity_m_s, integral_m_s):
    if mode == "off":
        return 0.0, integral_m_s
    if mode == "damping":
        return limit_force(-ACTIVE_DAMPING_NS_M * velocity_m_s), integral_m_s

    error_m = -displacement_m
    proportional_derivative_n = PID_KP_N_M * error_m - PID_KD_NS_M * velocity_m_s
    next_integral_m_s = integral_m_s + error_m * STEP_S
    requested_n = proportional_derivative_n + PID_KI_N_M_S * next_integral_m_s
    applied_n = limit_force(requested_n)
    if (requested_n - applied_n) * error_m > 0:
        # Keep the old integral when the new error would deepen saturation.
        return limit_force(proportional_derivative_n + PID_KI_N_M_S * integral_m_s), integral_m_s
    return applied_n, next_integral_m_s


def simulate(mode="off"):
    if mode not in MODES:
        raise ValueError(f"Unknown control mode: {mode}")
    time_s = np.arange(round(DURATION_S / STEP_S) + 1) * STEP_S
    external_force_n = disturbance_force(time_s)
    external_midpoint_n = disturbance_force(time_s[:-1] + STEP_S / 2)
    displacement_m = np.zeros_like(time_s)
    velocity_m_s = np.zeros_like(time_s)
    applied_control_n = np.zeros_like(time_s)
    integral_m_s = 0.0

    for index in range(len(time_s) - 1):
        applied_n, integral_m_s = control_force(mode, displacement_m[index], velocity_m_s[index], integral_m_s)
        applied_control_n[index] = applied_n
        displacement_m[index + 1], velocity_m_s[index + 1] = advance(
            displacement_m[index],
            velocity_m_s[index],
            external_force_n[index] + applied_n,
            external_midpoint_n[index] + applied_n,
            external_force_n[index + 1] + applied_n,
        )

    applied_control_n[-1], _ = control_force(mode, displacement_m[-1], velocity_m_s[-1], integral_m_s)
    acceleration_m_s2 = acceleration(displacement_m, velocity_m_s, external_force_n + applied_control_n)
    rng = np.random.default_rng(SENSOR_SEED)
    sensor_raw_m_s2 = GRAVITY_M_S2 + acceleration_m_s2 + rng.normal(0, SENSOR_NOISE_M_S2, time_s.size)
    energy_j = (MASS_KG * velocity_m_s**2 + SPRING_N_M * displacement_m**2) / 2
    return Trace(time_s, displacement_m, velocity_m_s, acceleration_m_s2, sensor_raw_m_s2, applied_control_n, energy_j)


def acceleration_rms(trace):
    window = (trace.time_s >= RMS_START_S) & (trace.time_s < DISTURBANCE_END_S)
    return float(np.sqrt(np.mean(trace.acceleration_m_s2[window] ** 2)))


if __name__ == "__main__":
    traces = {mode: simulate(mode) for mode in MODES}
    baseline_rms = acceleration_rms(traces["off"])
    print(f"Acceleration RMS ({RMS_START_S:g}–{DISTURBANCE_END_S:g} s) and energy at 5 s and 8 s")
    for mode, trace in traces.items():
        rms = acceleration_rms(trace)
        reduction = 100 * (1 - rms / baseline_rms)
        cutoff = np.searchsorted(trace.time_s, DISTURBANCE_END_S)
        print(f"{mode}: {rms:.6f} m/s² ({reduction:.2f}% lower), {trace.energy_j[cutoff]:.9g} → {trace.energy_j[-1]:.9g} J")
