import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import t

class PrognosticManager():

    def __init__(self, sim_mng):

        self.sim_mng = sim_mng
        self.esv_schedule = pd.DataFrame(
            columns=['TP_MIN', 'TP_EST', 'TP_MAX', 'T_END', 'OBJ', 'DRIVER']
        ).set_index(pd.Index([], name="UID"))  # Empty index named UID

    def add_engine_esv_plan(self, engine):

        uid = engine.uid
        tp_min = engine.next_esv['TP_MIN']
        tp_est = engine.next_esv['TP_EST']
        tp_max = engine.next_esv['TP_MAX']
        t_end  = engine.next_esv['T_END']
        driver = engine.next_esv['DRIVER']

        self.esv_schedule.loc[uid] = \
            [tp_min, tp_est, tp_max, t_end, engine, driver]

    def update_engine_esv_plan(self, engine):

        uid = engine.uid
        tp_min = engine.next_esv['TP_MIN']
        tp_est = engine.next_esv['TP_EST']
        tp_max = engine.next_esv['TP_MAX']
        t_end = engine.next_esv['T_END']
        driver = engine.next_esv['DRIVER']

        self.esv_schedule.loc[uid] = \
            [tp_min, tp_est, tp_max, t_end, engine, driver]


    def predict_rul(self):

        logging.debug("(%s) | Predicting Remaining Useful Life"
                      % self.sim_mng.SimTime.strftime("%Y-%m-%d %H:%M:%S"))

        for aircraft in self.sim_mng.aircraft_active:

            engine = aircraft.Engines

            # Part 1) EGTM
            if engine.fc_since_esv <= 3000:
                # below ~3000 EFC, we skip the forecast
                continue

            time_relevant = engine.history['TIME'][-325:]
            egtm_relevant = engine.history['EGTM'][-325:]
            llps_relevant = engine.history['LLP'][-325:]

            # Identify the last big jump to catch engine shop visits
            jump_threshold_egtm = 20
            relevant_index = find_last_jump_index(egtm_relevant, jump_threshold_egtm)

            # Slice again for final linear portion
            x_time = time_relevant[relevant_index:]
            y_egtm = egtm_relevant[relevant_index:]
            y_llps = llps_relevant[relevant_index:]

            # Fit line and solve for crossing at 5°C
            failtime_egtm = find_time_to_failure(x_time, y_egtm, 5)

            # Fit line and solve for crossing at critical remaining LLPs
            failtime_llps = find_time_to_failure(x_time, y_llps, 150)

            if failtime_egtm[0] < failtime_llps[0]:
                engine.next_esv = {
                    'UID': engine.uid,
                    'TP_MIN': failtime_egtm[1],
                    'TP_EST': failtime_egtm[0],
                    'TP_MAX': failtime_egtm[2],
                    'T_END': failtime_egtm[2] + dt.timedelta(days=25),
                    'OBJ': engine,
                    'DRIVER': 'EGTM'}
            else:
                engine.next_esv = {
                    'UID': engine.uid,
                    'TP_MIN': failtime_llps[1],
                    'TP_EST': failtime_llps[0],
                    'TP_MAX': failtime_llps[2],
                    'T_END': failtime_llps[2] + dt.timedelta(days=25),
                    'OBJ': engine,
                    'DRIVER': 'LLPS'}

            self.update_engine_esv_plan(engine)


    def _find_opp_windows(self, scope_beg, scope_end, esv_plan_df):

        # Extract blocked periods and sort by TP_MIN
        blocked_periods = sorted(zip(esv_plan_df["TP_MIN"], esv_plan_df["T_END"]), key=lambda x: x[0])

        opp_windows = [(scope_beg, scope_end)]  # Start with full opportunity range

        for tp_min, t_end in blocked_periods:
            new_available = []
            for start, end in opp_windows:
                # Case 1: Blocked period is completely outside (no overlap)
                if t_end < start or tp_min > end:
                    new_available.append((start, end))

                # Case 2: Blocked period overlaps with current available period
                else:
                    if start < tp_min:  # Keep the left part
                        new_available.append((start, tp_min))
                    if t_end < end:  # Keep the right part
                        new_available.append((t_end, end))

            opp_windows = new_available  # Update list

        return opp_windows

    def _find_mro_windows(self, opp_windows, buffer=dt.timedelta(days=7)):
        mro_windows = []
        mro_duration = dt.timedelta(days=25)

        for scope_beg, scope_end in reversed(opp_windows):  # Loop from future to past
            current_end = scope_end  # Start from the latest possible date in the window

            while current_end - buffer - mro_duration >= scope_beg:
                mro_start = current_end - buffer - mro_duration
                mro_windows.append((mro_start, current_end - buffer))  # Store the window

                current_end = mro_start

        return mro_windows

    def fix_esv_planning(self):

        if self.sim_mng.predictive != 'phm_case1':
            return

        now = self.sim_mng.SimTime
        three_years_later = now + dt.timedelta(days=3*365)

        # Reduce esv_plan dataframe to those entries which have TP_MIN
        filtered_esv_schedule = self.esv_schedule[self.esv_schedule['TP_MIN'].notna()]

        # Reduce further with only those that have a TP_MIN that is in less than a year
        filtered_esv_schedule = filtered_esv_schedule[(filtered_esv_schedule["TP_MIN"] >= now)
                                        & (filtered_esv_schedule["TP_MIN"] <= three_years_later)]

        if len(filtered_esv_schedule) < 2:
            return

        # Now for the remaining ones, check the scope
        # scope should start from SimTime and end at latest ESV planning start
        scope_beg = now
        latest_tp_min_entry = filtered_esv_schedule.loc[filtered_esv_schedule["TP_MIN"].idxmax()]
        scope_end = latest_tp_min_entry["TP_MIN"]

        # Find opportunity windows
        #filtered_esv_schedule = filtered_esv_schedule.sort_values(by='TP_MIN', ascending=True)
        opp_windows = self._find_opp_windows(scope_beg, scope_end, filtered_esv_schedule)

        mro_windows = self._find_mro_windows(opp_windows)

        if len(mro_windows) == 0:
            return

        #print("Break")

        # Randomly set the engine shop visits to that time
        n_iter = 500
        list_deviations = []
        engines2assign = list(filtered_esv_schedule['OBJ'])
        n_engines2assign = len(engines2assign)
        list_chosen_mro_window_indices = []

        for iteration in range(n_iter):
            #print('\n')
            #print("Iteration %d" % iteration)

            mro_windows_indices = np.arange(len(mro_windows))
            shuffled = np.random.permutation(mro_windows_indices)
            chosen_mro_window_indices = [int(shuffled[i % len(mro_windows)]) for i in range(n_engines2assign)]
            list_chosen_mro_window_indices.append(chosen_mro_window_indices)

            time_devs = []

            # For evaluation
            for idx, engine2assign in enumerate(engines2assign):

                est_time = filtered_esv_schedule.loc[engine2assign.uid]['TP_EST']
                chosen_window = mro_windows[chosen_mro_window_indices[idx]]
                new_time = chosen_window[0] # + 0.5*(chosen_window[1] - chosen_window[0])
                time_dev = abs(est_time - new_time)
                time_devs.append(time_dev)

                #print("ENG %d | Old: %s | New: %s | Dev: %s" % (engine2assign.uid, str(est_time), str(new_time), str(time_dev)))


            deviation = np.sum(time_devs)
            #print("Total Deviation: %s" % deviation)
            list_deviations.append(deviation)

        lowest_dev_idx = np.argmin(list_deviations)
        for idx, engine2assign in enumerate(engines2assign):

            chosen_window_idx = list_chosen_mro_window_indices[lowest_dev_idx][idx]
            chosen_window = mro_windows[chosen_window_idx]
            new_time = chosen_window[0]

            self.esv_schedule.loc[engine2assign.uid, 'TP_MIN'] = new_time
            self.esv_schedule.loc[engine2assign.uid, 'TP_EST'] = new_time
            self.esv_schedule.loc[engine2assign.uid, 'TP_MAX'] = new_time + dt.timedelta(days=25)
            self.esv_schedule.loc[engine2assign.uid, 'T_END'] = new_time + dt.timedelta(days=25)

            engine2assign.next_scheduled_esv = new_time

    def fix_esv_planning_with_fhratio(self):

        if self.sim_mng.predictive != 'phm_case2':
            return

        # self.esv_schedule.loc[0, 'OBJ'].set_fh_fc_ratio(0.5)
        # self.esv_schedule.loc[1, 'OBJ'].set_fh_fc_ratio(1)
        # self.esv_schedule.loc[2, 'OBJ'].set_fh_fc_ratio(1.5)
        # self.esv_schedule.loc[3, 'OBJ'].set_fh_fc_ratio(2)
        # self.esv_schedule.loc[4, 'OBJ'].set_fh_fc_ratio(2.5)
        # self.esv_schedule.loc[5, 'OBJ'].set_fh_fc_ratio(3)
        # self.esv_schedule.loc[6, 'OBJ'].set_fh_fc_ratio(3.5)
        # self.esv_schedule.loc[7, 'OBJ'].set_fh_fc_ratio(4)
        # self.esv_schedule.loc[8, 'OBJ'].set_fh_fc_ratio(4.5)
        # self.esv_schedule.loc[9, 'OBJ'].set_fh_fc_ratio(5)
        # self.esv_schedule.loc[10, 'OBJ'].set_fh_fc_ratio(5.5)
        # self.esv_schedule.loc[11, 'OBJ'].set_fh_fc_ratio(6)
        # self.esv_schedule.loc[12, 'OBJ'].set_fh_fc_ratio(6.5)
        # self.esv_schedule.loc[13, 'OBJ'].set_fh_fc_ratio(7)
        # self.esv_schedule.loc[14, 'OBJ'].set_fh_fc_ratio(7.5)
        # self.esv_schedule.loc[15, 'OBJ'].set_fh_fc_ratio(8)
        # self.esv_schedule.loc[16, 'OBJ'].set_fh_fc_ratio(8.5)
        # self.esv_schedule.loc[17, 'OBJ'].set_fh_fc_ratio(9)
        # self.esv_schedule.loc[18, 'OBJ'].set_fh_fc_ratio(9.5)
        # self.esv_schedule.loc[19, 'OBJ'].set_fh_fc_ratio(10)
        # self.esv_schedule.loc[20, 'OBJ'].set_fh_fc_ratio(10.5)
        # self.esv_schedule.loc[21, 'OBJ'].set_fh_fc_ratio(11)
        # self.esv_schedule.loc[22, 'OBJ'].set_fh_fc_ratio(11.5)
        #
        # print("Pause")

        now = self.sim_mng.SimTime
        # three_years_later = now + dt.timedelta(days=3 * 365)
        #
        # # Reduce esv_plan dataframe to those entries which have TP_MIN
        # filtered_esv_schedule = self.esv_schedule[self.esv_schedule['TP_MIN'].notna()]
        #
        # # Reduce further with only those that have a TP_MIN that is in less than a year
        # filtered_esv_schedule = filtered_esv_schedule[(filtered_esv_schedule["TP_MIN"] >= now)
        #                                               & (filtered_esv_schedule["TP_MIN"] <= three_years_later)]
        #
        # if len(filtered_esv_schedule) < 2:
        #     return

        filtered_esv_schedule = self.esv_schedule

        # Now for the remaining ones, check the scope
        # scope should start from SimTime and end at latest ESV planning start
        earliest_estimation = filtered_esv_schedule.loc[filtered_esv_schedule['TP_EST'].idxmin()]
        latest_estimation = filtered_esv_schedule.loc[filtered_esv_schedule['TP_EST'].idxmax()]
        scope_beg = earliest_estimation['TP_EST'] - dt.timedelta(days=365)
        scope_end = latest_estimation['TP_EST'] + dt.timedelta(days=365)


        # Find opportunity windows
        # filtered_esv_schedule = filtered_esv_schedule.sort_values(by='TP_MIN', ascending=True)
        opp_windows = self._find_opp_windows(scope_beg, scope_end, filtered_esv_schedule)

        mro_windows = self._find_mro_windows(opp_windows)

        if len(mro_windows) == 0:
            print("NO MRO WINDOWS")
            return

        # Randomly set the engine shop visits to that time
        n_iter = 500
        list_deviations = []
        engines2assign = list(filtered_esv_schedule['OBJ'])
        n_engines2assign = len(engines2assign)
        list_chosen_mro_window_indices = []

        for iteration in range(n_iter):
            # print('\n')
            # print("Iteration %d" % iteration)

            mro_windows_indices = np.arange(len(mro_windows))
            shuffled = np.random.permutation(mro_windows_indices)
            chosen_mro_window_indices = [int(shuffled[i % len(mro_windows)]) for i in range(n_engines2assign)]
            list_chosen_mro_window_indices.append(chosen_mro_window_indices)

            time_devs = []

            # For evaluation
            for idx, engine2assign in enumerate(engines2assign):
                est_time = filtered_esv_schedule.loc[engine2assign.uid]['TP_EST']
                chosen_window = mro_windows[chosen_mro_window_indices[idx]]
                new_time = chosen_window[0]  # + 0.5*(chosen_window[1] - chosen_window[0])


                required_slope = (5 - engine2assign.egtm) / ((new_time - now).total_seconds() / (3600 * 24))
                required_fh_ratio = required_slope * 14548.9948029459 + 69.0960960240128
                time_dev = abs(est_time - new_time)

                if required_fh_ratio > 12 or required_slope < 0.5:
                    time_dev *= 1000
                time_devs.append(time_dev)

                # print("ENG %d | Old: %s | New: %s | Dev: %s" % (engine2assign.uid, str(est_time), str(new_time), str(time_dev)))

            deviation = np.sum(time_devs)
            # print("Total Deviation: %s" % deviation)
            list_deviations.append(deviation)

        lowest_dev_idx = np.argmin(list_deviations)
        for idx, engine2assign in enumerate(engines2assign):
            chosen_window_idx = list_chosen_mro_window_indices[lowest_dev_idx][idx]
            chosen_window = mro_windows[chosen_window_idx]
            new_time = chosen_window[0]

            # f(x) = a*x + b
            # x is in days, a is degrees per days, b is degrees
            # to "hit" the new time, we have to adjust the fh-ratio

            required_slope = (5 - engine2assign.egtm)/((new_time - now).total_seconds()/(3600*24))
            required_fh_ratio = required_slope * 14548.9948029459 + 69.0960960240128

            if required_fh_ratio < 0.5:
                engine2assign.next_scheduled_esv = new_time
            else:

                required_fh_ratio = min(12, required_fh_ratio)
                engine2assign.set_fh_fc_ratio(required_fh_ratio)

            self.esv_schedule.loc[engine2assign.uid, 'TP_MIN'] = new_time
            self.esv_schedule.loc[engine2assign.uid, 'TP_EST'] = new_time
            self.esv_schedule.loc[engine2assign.uid, 'TP_MAX'] = new_time + dt.timedelta(days=25)
            self.esv_schedule.loc[engine2assign.uid, 'T_END'] = new_time + dt.timedelta(days=25)







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


def find_time_to_failure(x_data, y_data, threshold):

    confidence = 0.95

    # Convert datetime to numerical values (seconds since the first timestamp)
    first_timestamp = x_data[0].timestamp()
    x_numeric = np.array([(date.timestamp() - first_timestamp) for date in x_data])

    # Fit
    a, b = np.polyfit(x_numeric, y_data, 1)

    x_numeric_target = (threshold - b) / a

    # Convert back to datetime
    crossing_datetime = dt.datetime.fromtimestamp(first_timestamp + x_numeric_target)

    # Calculate residuals to estimate uncertainty
    y_pred = a * x_numeric + b
    residuals = y_data - y_pred
    n = len(x_numeric)

    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))  # Standard error

    # Confidence interval calculation
    t_val = t.ppf((1 + confidence) / 2, df=n - 2) if n > 2 else 0
    x_uncertainty = t_val * s_err / np.abs(a)  # Propagated uncertainty

    # Convert to datetime bounds
    lower_bound = dt.datetime.fromtimestamp(first_timestamp + (x_numeric_target - x_uncertainty))
    upper_bound = dt.datetime.fromtimestamp(first_timestamp + (x_numeric_target + x_uncertainty))


    return crossing_datetime, lower_bound, upper_bound

