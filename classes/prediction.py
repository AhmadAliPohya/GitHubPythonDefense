import logging
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt


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

    logging.info("(%s) | Predicting Remaining Useful Life"
                 % mng.SimTime.strftime("%Y-%m-%d %H:%M:%S"))

    for aircraft in mng.aircraft_fleet:

        engine = aircraft.Engines

        # Part 1) EGTM
        if engine.fc_counter <= 1050:
            # below ~1000 EFC, we skip the forecast because the slope is different (faster deterioration).
            continue

        past_flights = []
        one_week_ago = mng.SimTime - dt.timedelta(weeks=4)
        for flight in reversed(aircraft.past_flights):
            if flight.t_beg >= one_week_ago:
                past_flights.append(flight)

        avg_fc_per_week = len(past_flights) / 4

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
        efc_crit_egtm = ((5 - b_egtm) / a_egtm) - engine.fc_counter
        efc_warn_egtm = ((10 - b_egtm) / a_egtm) - engine.fc_counter
        time_crit_egtm = mng.SimTime + dt.timedelta(
            days=7*efc_crit_egtm/avg_fc_per_week)
        time_warn_egtm = mng.SimTime + dt.timedelta(
            days=7 * efc_warn_egtm / avg_fc_per_week)

        # Part 2) SOHs
        sohs = engine.history['SOH']
        efcs = engine.history['EFCs']

        jump_threshold_soh = 0.2
        relevant_index_soh = find_last_jump_index(sohs, jump_threshold_soh)

        x_soh = efcs[relevant_index_soh:]
        y_soh = sohs[relevant_index_soh:]

        # Fit line and solve for crossing at 0 (assuming 0 means 'failed'?)
        a_soh, b_soh = fit_and_extrapolate(x_soh, y_soh)
        efc_crit_soh = ((0.0 - b_soh) / a_soh) - engine.fc_counter
        efc_warn_soh = ((0.2 - b_soh) / a_soh) - engine.fc_counter
        time_crit_soh = mng.SimTime + dt.timedelta(
            days=7*efc_crit_soh / avg_fc_per_week)
        time_warn_soh = mng.SimTime + dt.timedelta(
            days=7*efc_warn_soh / avg_fc_per_week)

        # Part 3) LLPs
        efc_crit_llp = engine.llp_life - 500
        efc_warn_llp = engine.llp_life - 3000
        time_crit_llp = mng.SimTime + dt.timedelta(
            days=7*efc_crit_llp / avg_fc_per_week)
        time_warn_llp = mng.SimTime + dt.timedelta(
            days=7*efc_warn_llp / avg_fc_per_week)

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

        next_esv_time, min_category = min(
            (category['TIME'][-1], category_key)  # Only consider the last entry ("crit") in TIME
            for category_key, category in engine.rul.items()
            if category_key in ['LLP', 'EGTM']
        )

        next_esv_efcs = engine.rul[min_category]['EFC'][engine.rul[min_category]['TIME'].index(next_esv_time)]

        engine.next_esv = {'TIME': next_esv_time, 'EFCs': next_esv_efcs, 'DRIVER': min_category}

        engine.delete_esv_efc_abs = efc_crit_egtm + engine.fc_counter
        # if engine.uid == 0:

            # delete
            # print(next_esv_time, next_esv_efcs, min_category)

            # debug_plot_rul(
            #     engine=engine,
            #     x_egtm=x_egtm,
            #     y_egtm=y_egtm,
            #     a_egtm=a_egtm,
            #     b_egtm=b_egtm,
            #     x_soh=x_soh,
            #     y_soh=y_soh,
            #     a_soh=a_soh,
            #     b_soh=b_soh,
            #     efc_crit_llp=efc_crit_llp
            # )

            # Convert relevant data to time for EGTM
            # time_egtm = [mng.SimTime + dt.timedelta(days=7 * (efc - engine.fc_counter) / avg_fc_per_week) for efc in
            #              x_egtm]
            # time_soh = [mng.SimTime + dt.timedelta(days=7 * (efc - engine.fc_counter) / avg_fc_per_week) for efc in
            #             x_soh]
            #
            # # Call the function
            # debug_plot_rul_over_time(
            #     engine=engine,
            #     time_egtm=time_egtm,
            #     y_egtm=y_egtm,
            #     time_soh=time_soh,
            #     y_soh=y_soh,
            #     efc_crit_llp=efc_crit_llp,
            #     time_warn_llp=time_warn_llp,
            #     time_crit_llp=time_crit_llp,
            # )

    return mng


def debug_plot_rul(engine, x_egtm, y_egtm, a_egtm, b_egtm, x_soh, y_soh, a_soh, b_soh, efc_crit_llp):
    """
    Creates a three-part subplot visualizing EGTM, SOH, and LLP predictions.

    Parameters
    ----------
    engine : Engine object
        The engine being analyzed.
    x_egtm, y_egtm : list
        EFCs and EGTM values used for the linear fit.
    a_egtm, b_egtm : float
        Slope and intercept of the EGTM linear fit.
    x_soh, y_soh : list
        EFCs and SOH values used for the linear fit.
    a_soh, b_soh : float
        Slope and intercept of the SOH linear fit.
    efc_crit_llp : float
        Critical EFC value for LLP life.
    """
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"RUL Debug Plot for Engine UID {engine.uid}", fontsize=16)

    # EGTM subplot
    efc_fail_egtm = (5 - b_egtm) / a_egtm
    x_line_egtm = np.linspace(x_egtm[0], efc_fail_egtm, 100)
    y_line_egtm = a_egtm * x_line_egtm + b_egtm

    axes[0].scatter(engine.history['EFCs'], engine.history['EGTM'], color='blue', alpha=0.5, s=10, label='All EGTM data')
    axes[0].scatter(x_egtm, y_egtm, color='red', s=20, label='Segment used for fit')
    axes[0].plot(x_line_egtm, y_line_egtm, 'r--', label='Extrapolated EGTM trend')
    axes[0].axvline(efc_fail_egtm, color='orange', linestyle=':', label=f'Critical EFC ~ {efc_fail_egtm:.1f}')
    axes[0].axhline(5, color='gray', linestyle=':', label='EGTM=5°C Threshold')
    axes[0].set_title("EGTM Prediction")
    axes[0].set_xlabel("EFC (Engine Flight Cycles)")
    axes[0].set_ylabel("EGTM (°C)")
    axes[0].grid(True)
    axes[0].legend()

    # SOH subplot
    efc_fail_soh = (0.0 - b_soh) / a_soh
    x_line_soh = np.linspace(x_soh[0], efc_fail_soh, 100)
    y_line_soh = a_soh * x_line_soh + b_soh

    axes[1].scatter(engine.history['EFCs'], engine.history['SOH'], color='blue', alpha=0.5, s=10, label='All SOH data')
    axes[1].scatter(x_soh, y_soh, color='red', s=20, label='Segment used for fit')
    axes[1].plot(x_line_soh, y_line_soh, 'r--', label='Extrapolated SOH trend')
    axes[1].axvline(efc_fail_soh, color='orange', linestyle=':', label=f'Critical EFC ~ {efc_fail_soh:.1f}')
    axes[1].axhline(0.0, color='gray', linestyle=':', label='SOH=0 Threshold')
    axes[1].set_title("SOH Prediction")
    axes[1].set_xlabel("EFC (Engine Flight Cycles)")
    axes[1].set_ylabel("SOH")
    axes[1].grid(True)
    axes[1].legend()

    # LLP subplot
    axes[2].scatter(engine.history['EFCs'], engine.history['LLP'], color='blue', alpha=0.5, s=10, label='All LLP data')
    axes[2].axvline(efc_crit_llp, color='orange', linestyle=':', label=f'Critical EFC ~ {efc_crit_llp:.1f}')
    axes[2].set_title("LLP Life Prediction")
    axes[2].set_xlabel("EFC (Engine Flight Cycles)")
    axes[2].set_ylabel("LLP Remaining Life")
    axes[2].grid(True)
    axes[2].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def debug_plot_rul_over_time(engine, time_egtm, y_egtm, time_soh, y_soh, efc_crit_llp, time_warn_llp, time_crit_llp):
    """
    Creates a three-part subplot visualizing EGTM, SOH, and LLP predictions over time.

    Parameters
    ----------
    engine : Engine object
        The engine being analyzed.
    time_egtm, y_egtm : list
        Time and EGTM values used for the linear fit.
    time_soh, y_soh : list
        Time and SOH values used for the linear fit.
    efc_crit_llp : float
        Critical EFC value for LLP life.
    time_warn_llp, time_crit_llp : datetime
        Warning and critical times for LLP.
    """
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f"RUL Debug Plot for Engine UID {engine.uid} (Over Time)", fontsize=16)

    # EGTM subplot
    axes[0].scatter(engine.history['TIME'], engine.history['EGTM'], color='blue', alpha=0.5, s=10, label='All EGTM data')
    axes[0].scatter(time_egtm, y_egtm, color='red', s=20, label='Segment used for fit')
    axes[0].axvline(time_egtm[-1], color='orange', linestyle=':', label=f'Predicted EGTM Failure')
    axes[0].axhline(5, color='gray', linestyle=':', label='EGTM=5°C Threshold')
    axes[0].set_title("EGTM Prediction")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("EGTM (°C)")
    axes[0].grid(True)
    axes[0].legend()

    # SOH subplot
    axes[1].scatter(engine.history['TIME'], engine.history['SOH'], color='blue', alpha=0.5, s=10, label='All SOH data')
    axes[1].scatter(time_soh, y_soh, color='red', s=20, label='Segment used for fit')
    axes[1].axvline(time_soh[-1], color='orange', linestyle=':', label=f'Predicted SOH Failure')
    axes[1].axhline(0.0, color='gray', linestyle=':', label='SOH=0 Threshold')
    axes[1].set_title("SOH Prediction")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("SOH")
    axes[1].grid(True)
    axes[1].legend()

    # LLP subplot
    axes[2].scatter(engine.history['TIME'], engine.history['LLP'], color='blue', alpha=0.5, s=10, label='All LLP data')
    axes[2].axvline(time_warn_llp, color='yellow', linestyle='--', label=f'LLP Warning')
    axes[2].axvline(time_crit_llp, color='orange', linestyle=':', label=f'LLP Failure')
    axes[2].set_title("LLP Life Prediction")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("LLP Remaining Life")
    axes[2].grid(True)
    axes[2].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
