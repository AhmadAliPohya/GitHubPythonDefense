import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

from scipy.constants import value

from classes.events import FlightEvent, TurnaroundEvent, MaintenanceEvent

def find_last_jump_index(values, jump_threshold):
    """
    Finds the last index i where values[i] - values[i-1] > jump_threshold,
    scanning from the beginning to the end.
    If none is found, returns 0.

    :param values: 1D array-like of numeric data
    :param jump_threshold: float, threshold for a 'jump'
    :return: integer, the index in 'values' where the last jump occurs (+1 from np.diff index)
    """
    if len(values) < 2:
        return 0  # Not enough data to even compute diff

    diffs = np.diff(values)  # diffs[i] = values[i+1] - values[i]
    mask = diffs > jump_threshold  # boolean array
    indices = np.where(mask)[0]  # positions of True
    if len(indices) > 0:
        return indices[-1] + 1  # +1 to map from diff index to 'values' index
    else:
        return 0


def fit_and_extrapolate(x_data, y_data):
    """
    Fits a line y = a*x + b to (x_data, y_data). Then solves a*x + b = crossing_value.
    Returns (efc_fail, a, b).

    :param x_data: 1D array-like (e.g. EFCs)
    :param y_data: 1D array-like (e.g. EGTM, SOH, etc.)
    :return: (efc_fail, a, b)
    """
    if len(x_data) < 2:
        return None, None  # Not enough points to fit

    # Fit
    a, b = np.polyfit(x_data, y_data, 1)

    return a, b


def predict_rul(mng):
    for aircraft in mng.aircraft_fleet:

        engine = aircraft.Engines

        # Part 1) EGTM
        if engine.fc_counter <= 1050:
            # below ~1000 EFC, we skip the forecast because the slope is different (faster deterioration).
            continue

        past_flights = []
        one_week_ago = mng.SimTime - dt.timedelta(weeks=1)
        for flight in reversed(aircraft.past_flights):
            if flight.t_beg >= one_week_ago:
                past_flights.append(flight)

        avg_fc_per_week = len(past_flights)

        # Find the first index where EFC >= 1000
        first_index = next(
            (i for i, val in enumerate(engine.history['EFCs']) if val >= 1000),
            None
        )
        if first_index is None:
            continue  # No EFC >= 1000 found?

        # Slice from there
        efcs_steady = engine.history['EFCs'][first_index:]
        egtm_steady = engine.history['EGTM'][first_index:]

        # Identify the last big jump
        jump_threshold_egtm = 20
        relevant_index = find_last_jump_index(egtm_steady, jump_threshold_egtm)

        # Slice again for final linear portion
        x_egtm = efcs_steady[relevant_index:]
        y_egtm = egtm_steady[relevant_index:]

        # Fit line and solve for crossing at 5°C
        a_egtm, b_egtm = fit_and_extrapolate(x_egtm, y_egtm)
        efc_crit_egtm = (5 - b_egtm) / a_egtm
        efc_warn_egtm = (10 - b_egtm) / a_egtm
        time_crit_egtm = mng.SimTime + dt.timedelta(
            days=7*(efc_crit_egtm - engine.fc_counter)/avg_fc_per_week)
        time_warn_egtm = mng.SimTime + dt.timedelta(
            days=7 * (efc_warn_egtm - engine.fc_counter) / avg_fc_per_week)

        # Part 2) SOHs
        sohs = engine.history['SOH']
        efcs = engine.history['EFCs']

        jump_threshold_soh = 0.2
        relevant_index_soh = find_last_jump_index(sohs, jump_threshold_soh)

        x_soh = efcs[relevant_index_soh:]
        y_soh = sohs[relevant_index_soh:]

        # Fit line and solve for crossing at 0 (assuming 0 means 'failed'?)
        a_soh, b_soh = fit_and_extrapolate(x_soh, y_soh)
        efc_crit_soh = (0.0 - b_soh) / a_soh
        efc_warn_soh = (0.2 - b_soh) / a_soh
        time_crit_soh = mng.SimTime + dt.timedelta(
            days=7 * (efc_crit_soh - engine.fc_counter) / avg_fc_per_week)
        time_warn_soh = mng.SimTime + dt.timedelta(
            days=7 * (efc_warn_soh - engine.fc_counter) / avg_fc_per_week)

        # Part 3) LLPs
        efc_crit_llp = engine.llp_life - 500
        efc_warn_llp = engine.llp_life - 3000
        time_crit_llp = mng.SimTime + dt.timedelta(
            days=7 * (efc_crit_llp - engine.fc_counter) / avg_fc_per_week)
        time_warn_llp = mng.SimTime + dt.timedelta(
            days=7 * (efc_warn_llp - engine.fc_counter) / avg_fc_per_week)

        engine.rul = {
            'EGTM': {
                'EFC': [efc_warn_egtm, efc_crit_egtm],
                'TIME': [time_warn_egtm, time_crit_egtm]
            },
            'SOH': {
                'EFC': [efc_warn_soh, efc_crit_soh],
                'TIME': [time_warn_soh, time_crit_soh]
            },
            'LLP': {
                'EFC': [efc_warn_llp, efc_crit_llp],
                'TIME': [time_warn_llp, time_crit_llp]
            }
        }

        # next_esv_time, min_category = min(
        #     (time, category_key)
        #     for category_key, category in engine.rul.items()
        #     for time in category.get('TIME', [])
        # )
        #
        # mng.prepare_utilization_change()

    return mng


def debug_plot_egtm(engine, for_prediction, a, b):
    efcs_all = engine.history['EFCs']
    egtms_all = engine.history['EGTM']
    efcs_history = for_prediction['EGTM']['x']
    egtm_history = for_prediction['EGTM']['y']

    efc_fail = (5-b) / a  # crossing at y=0

    # 5) Create x-range for line
    x_min = efcs_history[0]
    x_max = efc_fail
    x_line = np.linspace(x_min, x_max, 5)
    y_line = a * x_line + b

    # ---- PLOT ----
    plt.figure(figsize=(8, 6))
    # Scatter all data (blue)
    plt.scatter(efcs_all, egtms_all, color='blue', alpha=0.1, s=0.5, label='All data')
    # Highlight portion used for linear fit (red)
    plt.scatter(efcs_history, egtm_history, color='red', alpha=0.8, s=0.5, label='Segment used for fit')
    # Fit line
    plt.plot(x_line, y_line, 'r--', label='Extrapolation')

    # Mark the EFC fail line
    plt.axvline(efc_fail, color='orange', linestyle=':', label=f'Fail EFC ~ {efc_fail:.1f}')
    plt.axhline(5, color='gray', linestyle=':', label='EGTM=5°C')

    plt.title(f"EGTM Debug Plot for Engine")
    plt.xlabel("EFC")
    plt.ylabel("EGTM")
    plt.grid(True)
    plt.legend()
    plt.show()
