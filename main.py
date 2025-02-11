import logging
import configparser as cp
from classes.aircraft import Aircraft
from classes.engines import Engines
import classes.tailassignment as tap
from classes.prediction import PrognosticManager
import classes.prediction as pred
from classes.events import FlightEvent, TurnaroundEvent, MaintenanceEvent
import numpy as np
import datetime as dt
from post import post
from utils.debug_plots import *
import os, csv

# Configure logging to write debug and above to a file called log.txt
logging.basicConfig(
    filename='log.txt',  # file path
    filemode='w',  # <--- Overwrite the file each time
    level=logging.DEBUG,  # minimum severity level to capture
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Only display to the nearest second
)

# Add a StreamHandler to also print log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Adjust level if needed
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s]: %(message)s',  # Match file format
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Get the root logger and add the console handler
logging.getLogger().addHandler(console_handler)


class SimulationManager:
    def __init__(self, config_path: str):
        """
        Initializes the Manager class with configuration, aircraft fleet,
        and engine tracking.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        """
        self._time_bounds = []
        self.current_time = None
        self.num_needed_spares = 0
        self.aog_events = 0
        self.event_calendar = None
        self.config = None
        self._mro_base = None
        self.aircraft_active = []
        self.engines_in_shop = []
        self.engines_in_pool = []
        self.yemo = []

        # Read configuration
        self._read_config(config_path)

        # Initiate Simulation Time
        SimTime_str = self.config.get('Simulation', 'initial_time')
        self.SimTime = dt.datetime.strptime(SimTime_str, '%Y-%m-%d %H:%M:%S')
        sim_duration = self.config.getint('Simulation', 'sim_duration')
        self.EndTime = self.SimTime + dt.timedelta(days=365 * sim_duration)

        # Initialize settings
        self._mro_base = self.config.get('Other', 'mro_base')
        self.predictive = self.config.get('Simulation', 'maint_strategy')
        if self.predictive in ['conventional', 'phm_case1']:
            self.tap_threshold = 1.
        else:
            self.tap_threshold = 0.25

        # Create aircraft fleet
        num_aircraft = self.config.getint('Aircraft', 'num_aircraft')
        self.aircraft_active, self.aircraft_inactive = self.create_aircraft_fleet(num_aircraft)

        # Create and attach engine sets to aircraft
        self.create_and_attach_engines()

        # Store some constant values for easier access
        self._avg_tat_hrs = self.config.getfloat('Aircraft', 'avg_tat_hrs')
        self._avg_tat_td = dt.timedelta(hours=self._avg_tat_hrs)

        logging.info("Initialization complete. Ready to start operations.")

    def _read_config(self, file_path: str):
        """
        Reads and parses the configuration file.

        Parameters
        ----------
        file_path : str
            Path to the configuration file.
        """
        self.config = cp.ConfigParser()
        self.config.read(file_path)
        random_seed = self.config.getint('Simulation', 'random_seed')
        np.random.seed(random_seed)

    def create_aircraft_fleet(self, num_aircraft: int) -> list:
        """
        Creates a fleet of aircraft based on the configuration.

        Parameters
        ----------
        num_aircraft : int
            Number of aircraft to create.

        Returns
        -------
        list of Aircraft
            List of created aircraft objects.
        """
        aircraft_fleet = [Aircraft(uid=n, config=self.config) for n in range(num_aircraft)]
        logging.info("Created %d aircraft" % num_aircraft)

        active_aircraft = []
        inactive_aircraft = []
        for idx, aircraft in enumerate(aircraft_fleet):
            if idx == 0:
                aircraft.set_entries_into_service(first=True)
            if aircraft.eis <= self.SimTime:
                active_aircraft.append(aircraft)
            else:
                inactive_aircraft.append(aircraft)

        return active_aircraft, inactive_aircraft

    def create_and_attach_engines(self) -> list:
        """
        Creates engine sets, assigns them to aircraft, and returns the engine fleet.

        Parameters
        ----------
        num_aircraft : int
            Number of engines to create and assign.
        aircraft_fleet : list of Aircraft
            List of aircraft to which engines will be assigned.

        Returns
        -------
        list of Engines
            List of created engine objects.
        """

        n = 0
        for aircraft in self.aircraft_active:
            engines = Engines(uid=n, config=self.config, aircraft=aircraft)
            engines.aircraft = aircraft
            aircraft.Engines = engines
            n += 1

        for aircraft in self.aircraft_inactive:
            engines = Engines(uid=n, config=self.config, aircraft=aircraft)
            engines.aircraft = aircraft
            aircraft.Engines = engines
            n += 1


    def next_aircraft_in_line(self):

        idx_earliest = 0
        tstamp = self.aircraft_active[0].last_tstamp
        for idx_loop, aircraft_loop in enumerate(self.aircraft_active):
            tstamp_loop = aircraft_loop.last_tstamp
            if tstamp_loop < tstamp:
                tstamp = tstamp_loop
                idx_earliest = idx_loop

        return self.aircraft_active[idx_earliest], idx_earliest


    def _calculate_scope(self, efc):
        """
        Calculates the scope of possible ESV times based on EFCs and FH/FC ratios.

        Parameters
        ----------
        efc : float
            Remaining Engine Flight Cycles (EFCs) until the ESV.

        Returns
        -------
        tuple
            A tuple (lower_bound, upper_bound) representing the earliest and latest
            possible times for the ESV.
        """

        # scope is defined by FH/FC ratio
        # 6 FH/FC for delay (longer flights) and 2 FH/FC for early (shorter flights)
        # todo IN PREDICTION redefine the scope after a "flag" (change of goal fh/fc ratio)
        lower_bound = self.SimTime + efc * (dt.timedelta(hours=2) + self._avg_tat_td)
        upper_bound = self.SimTime + efc * (dt.timedelta(hours=6) + self._avg_tat_td)

        return lower_bound, upper_bound

    def _find_opportunity_windows(self, scope_start, scope_end):
        """
        Finds time windows within the given scope where no ESVs are scheduled,
        accounting for overlapping ESVs.

        Parameters
        ----------
        scope_start : datetime
            The start of the scope to search for opportunity windows.
        scope_end : datetime
            The end of the scope to search for opportunity windows.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple represents a (start, end) time window
            where no ESVs are scheduled, meeting the minimum duration requirement.
        """
        # Sort time bounds by their start time
        sorted_bounds = sorted(self._time_bounds, key=lambda x: x['START'])

        # List to store identified windows
        windows = []

        # Handle overlapping ESVs by merging them
        merged_bounds = []
        current = {**sorted_bounds[0]}  # Shallow copy of the first ESV

        for next_bound in sorted_bounds[1:]:
            if next_bound['START'] <= current['END']:  # Overlap detected
                current['END'] = max(current['END'], next_bound['END'])  # Extend the end time
            else:
                merged_bounds.append(current)  # No overlap, finalize the current bound
                current = {**next_bound}  # Shallow copy of the next dictionary

        merged_bounds.append(current)  # Add the last bound

        # Handle the gap before the first ESV
        if merged_bounds:
            first_bound_start = merged_bounds[0]['START']
            if scope_start < first_bound_start:
                pre_gap_start = scope_start
                pre_gap_end = first_bound_start
                if (pre_gap_end - pre_gap_start).days >= 35:  # Check if at least 5 weeks
                    windows.append((pre_gap_start, pre_gap_end))

        # Iterate through the merged bounds to find gaps
        for i in range(len(merged_bounds) - 1):
            gap_start = merged_bounds[i]['END']
            gap_end = merged_bounds[i + 1]['START']

            # Ignore gaps outside the scope
            if gap_start > scope_end or gap_end < scope_start:
                continue

            # Constrain the gap to fit within the scope
            gap_start = max(gap_start, scope_start)
            gap_end = min(gap_end, scope_end)

            # Add the gap if it meets the minimum duration
            if gap_start < gap_end and (gap_end - gap_start).days >= 77:  # Check if at least 5 weeks
                windows.append((gap_start, gap_end))

        # Handle the gap after the last ESV
        if merged_bounds:
            last_bound_end = merged_bounds[-1]['END']
            if scope_end > last_bound_end:
                post_gap_start = last_bound_end
                post_gap_end = scope_end
                if (post_gap_end - post_gap_start).days >= 77:  # Check if at least 5 weeks
                    windows.append((post_gap_start, post_gap_end))

        return windows

    def _find_overlaps(self):

        # Initialize variables for overlap detection
        overlapping_groups = []  # List to store groups of overlapping ESVs
        visited = set()  # Set to track UIDs already processed

        # Compare each ESV's time bounds with all others
        for i, current in enumerate(self._time_bounds):
            if current['UID'] in visited:  # Skip UIDs already grouped
                continue

            group = [current['UID']]  # Start a new group with the current UID

            # Check for overlaps with other ESVs
            for j, other in enumerate(self._time_bounds):
                if i != j:  # Skip self-comparison
                    # Check if current and other time bounds overlap
                    if not (current['END'] < other['START'] or current['START'] > other['END']):
                        group.append(other['UID'])  # Add overlapping UID to the group
                        visited.add(other['UID'])  # Mark as visited

            # Add the group to the list if it contains more than one UID
            if len(group) > 1:
                overlapping_groups.append(group)
                visited.update(group)  # Mark all in the group as visited

        return overlapping_groups

    def _identify_time_bounds(self, esv_plan):

        self._time_bounds = [
            {
                'UID': uid,
                'START': time - dt.timedelta(weeks=4),  # Start time with uncertainty
                'END': time + dt.timedelta(weeks=3 + 4)  # End time with uncertainty
            }
            for uid, time in zip(esv_plan['UID'], esv_plan['TIME'])
        ]

    def adjust_engine_shop_visits(self):

        if self.predictive != 'phm_case1':
            return

        # Plan structure to store ESV data
        esv_plan = {'UID': [], 'OBJs': [], 'TIME': [], 'TIMEmin': [], 'TIMEmax': []}

        overlap_potential = -1
        # Step 2: Populate the ESV plan for each aircraft with available RUL data
        for aircraft in self.aircraft_active:
            engine = aircraft.Engines

            has_prediction = hasattr(engine, 'next_esv')
            if has_prediction:
                is_close = (engine.next_esv['BOUNDS'][0] - self.SimTime) < dt.timedelta(days=6 * 30)
                if is_close:
                    # only add if the next RUL is within the next 6 months
                    esv_plan['UID'].append(engine.uid)  # Add engine UID
                    esv_plan['TIME'].append(engine.next_esv['TIME'])  # Add ESV time
                    esv_plan['TIMEmin'].append(engine.next_esv['BOUNDS'][0]),
                    esv_plan['TIMEmax'].append(engine.next_esv['BOUNDS'][1]),
                    esv_plan['OBJs'].append(engine)
                    overlap_potential += 1

        if overlap_potential <= 0:  # less than two engines in there
            return


        self._time_bounds = []
        for uid in esv_plan['UID']:
            index = esv_plan['UID'].index(uid)
            timebounddict = {
                'UID': uid,
                'START': esv_plan['TIMEmin'][index],
                'END': esv_plan['TIMEmax'][index] + dt.timedelta(days=25),
            }
            if esv_plan['OBJs'][index].next_scheduled_esv is not None:
                timebounddict = esv_plan['OBJs'][index].time_bounds
            self._time_bounds.append(timebounddict)


        overlapping_groups = self._find_overlaps()

        while len(overlapping_groups) > 0:

            for group in overlapping_groups:

                no_opp_windows_found = 0

                for uid in group:

                    index = esv_plan['UID'].index(uid)

                    if esv_plan['OBJs'][index].next_scheduled_esv is not None:
                        no_opp_windows_found += 1
                        continue

                    scope_start = self.SimTime
                    scope_end = esv_plan['TIMEmax'][index]

                    opportunity_windows = self._find_opportunity_windows(scope_start, scope_end)
                    # If no opportunity windows exist within the scope, skip this engine
                    if not opportunity_windows:
                        no_opp_windows_found += 1
                        continue
                    opportunity_windows = sorted(opportunity_windows, key=lambda x: x[0])

                    selected_window = opportunity_windows[-1]
                    target_time = selected_window[1] - dt.timedelta(days=7+25)

                    #visualize_gantt_with_scope_and_windows(
                    #    self._time_bounds, opportunity_windows, scope_start, scope_end, self)

                    esv_plan['OBJs'][index].next_scheduled_esv = target_time
                    esv_plan['OBJs'][index].time_bounds = {'UID': uid, 'START': target_time, 'END': target_time + dt.timedelta(days=25)}
                    esv_plan['TIMEmin'][index] = target_time
                    esv_plan['TIMEmax'][index] = target_time

                    self._time_bounds = []
                    for uid in esv_plan['UID']:
                        index = esv_plan['UID'].index(uid)
                        timebounddict = {
                            'UID': uid,
                            'START': esv_plan['TIMEmin'][index],
                            'END': esv_plan['TIMEmax'][index] + dt.timedelta(days=25),
                        }
                        if esv_plan['OBJs'][index].next_scheduled_esv is not None:
                            timebounddict = esv_plan['OBJs'][index].time_bounds
                        self._time_bounds.append(timebounddict)

                    # visualize_gantt_with_scope_and_windows(
                    #     self._time_bounds, opportunity_windows, scope_start, scope_end, self)

                    print("Setting ESV of %d to %s" % (uid, str(target_time)))

                    break

            if no_opp_windows_found == len(group):
                break
            overlapping_groups = self._find_overlaps()




    def adjust_engines_target_ratio(self):
        """
        Updates the ESV (Engine Shop Visit) plans for all aircraft in the fleet,
        identifies overlapping ESVs, and redistributes them to avoid conflicts.

        This method is called periodically in the simulation to ensure ESV
        schedules are optimized.
        """

        overwrite_uid = self.config.getint('Other', 'ac_overwrite_uid')
        if overwrite_uid >= 0:
            ac_index = next((i for i, obj in enumerate(self.aircraft_active) if obj.uid == overwrite_uid))
            self.aircraft_active[ac_index].goal_fh_fc_ratio = self.config.getfloat('Other', 'ac_overwrite_fhr')
            return

        if not self.predictive == 'phm_case2':
            return

        # Log the adjustment details
        logging.info("(%s) | Updating Engine's Goal EFH/EFC Ratio"
                     % self.SimTime.strftime("%Y-%m-%d %H:%M:%S"))

        # Step 1: Initialize variables
        n_aircraft = len(self.aircraft_active)  # Total number of aircraft in the fleet
        rul_available = 0  # Counter for aircraft with Remaining Useful Life (RUL) available
        esv_plan = {'UID': [], 'OBJs': [], 'TIME': [], 'EFCs': []}  # Plan structure to store ESV data

        # Step 2: Populate the ESV plan for each aircraft with available RUL data
        for aircraft in self.aircraft_active:
            engine = aircraft.Engines
            if hasattr(engine, 'next_esv'):  # Check if the engine has a defined next ESV
                rul_available += 1
                esv_plan['UID'].append(engine.uid)  # Add engine UID
                esv_plan['TIME'].append(engine.next_esv['TIME'])  # Add ESV time
                esv_plan['EFCs'].append(engine.next_esv['EFCs'])  # Add EFCs until ESV
                esv_plan['OBJs'].append(engine)

        # If not all aircraft have available RUL data, exit early
        # todo change this!
        if rul_available < n_aircraft:
            return

        # Step 3: Identify overlapping ESVs using time bounds
        # Define time windows for each ESV (1 week before to 3 weeks after)
        self._identify_time_bounds(esv_plan)

        overlapping_groups = self._find_overlaps()

        # Adjust ESV timings within the calculated scopes
        while len(overlapping_groups) > 0:

            # Step 4: Redistribute overlapping ESVs to avoid conflicts
            for group in overlapping_groups:

                # I THINK THERE'S AN ERROR HERE

                # Calculate the scope (min/max possible times) for each engine in the group
                scopes = {}
                for uid in group:
                    # Find the index of the current UID in the esv_plan['UID'] list
                    index = esv_plan['UID'].index(uid)

                    # Use the index to retrieve the corresponding EFC and TIME
                    efc = esv_plan['EFCs'][index]
                    time = esv_plan['TIME'][index]

                    # Calculate the scope and store it
                    #scopes[uid] = self._calculate_scope(efc, time)
                    scopes[uid] = self._calculate_scope(efc)

                no_opp_windows_found = 0

                for uid in group:

                    # Find the index of the current UID in the esv_plan['UID'] list again
                    index = esv_plan['UID'].index(uid)

                    scope_start, scope_end = scopes[uid]  # Get scope bounds for this engine
                    opportunity_windows = self._find_opportunity_windows(scope_start, scope_end)

                    # If no opportunity windows exist within the scope, skip this engine
                    if not opportunity_windows:
                        logging.info("(%s) | No opportunity windows for UID %d within scope"
                                     % (self.SimTime.strftime("%Y-%m-%d %H:%M:%S"), uid))
                        no_opp_windows_found += 1
                        continue

                    # Select the first available window (e.g., midpoint of the window)
                    selected_window = opportunity_windows[np.random.randint(len(opportunity_windows))]
                    target_time = min(selected_window) + np.abs(selected_window[0] - selected_window[1]) / 2
                    remaining_efc = esv_plan['EFCs'][index]  # Remaining EFCs until ESV

                    # Retrieve the turnaround time in hours
                    avg_tat_hrs = self.config.getfloat('Aircraft', '_avg_tat_hrs')

                    time_diff = target_time - self.SimTime
                    new_fh_fc_ratio = ((time_diff / remaining_efc) - dt.timedelta(
                        hours=avg_tat_hrs)).total_seconds() / 3600

                    # Update the ESV plan with the adjusted time
                    esv_plan['OBJs'][index].set_fh_fc_ratio(new_fh_fc_ratio)
                    esv_plan['TIME'][index] = target_time
                    esv_plan['EFCs'][index] = new_fh_fc_ratio  # Update the EFCs with the new ratio

                    self._identify_time_bounds(esv_plan)

                    # Log the adjustment details
                    logging.info("(%s) | Updated Engine %d's Ratio to %.2f"
                                 % (self.SimTime.strftime("%Y-%m-%d %H:%M:%S"), esv_plan['OBJs'][index].uid,
                                    new_fh_fc_ratio))

            if no_opp_windows_found == len(group):
                break
            overlapping_groups = self._find_overlaps()

        return




def main():
    """
    Main function to initialize and run the simulation.
    """
    mng = SimulationManager(config_path='config.ini')
    prog_mng = PrognosticManager(mng)
    for aircraft in mng.aircraft_active:
        prog_mng.add_engine_esv_plan(aircraft.Engines)

    # Perform initial tail assignment
    mng = tap.initial(mng)

    mng.yemo.append((mng.SimTime.year, mng.SimTime.month))

    # Read execution frequencies from the config file
    predict_rul_frequency = mng.config['Simulation'].get('predict_rul', 'monthly')

    while True:

        # Find the next aircraft to process
        #idx_aircraft = next_aircraft_in_line(mng.aircraft_active)
        #aircraft = mng.aircraft_active[idx_aircraft]

        aircraft, _ = mng.next_aircraft_in_line()

        # Check for engine maintenance needs
        if aircraft.Engines.maintenance_due(mng):

            skip_maintenance = False

            if not skip_maintenance:

                # import pprint
                # print('--- Engine %d' % aircraft.Engines.uid)
                # print(aircraft.Engines.rul['EGTM']['TIME'][-1], aircraft.Engines.delete_esv_efc_abs)
                # print(mng.SimTime, aircraft.Engines.fc_counter)
                # print(mng.SimTime - aircraft.Engines.rul['EGTM']['TIME'][-1],
                #       aircraft.Engines.fc_counter - aircraft.Engines.delete_esv_efc_abs)

                # Add a maintenance event
                maintenance_event = MaintenanceEvent(
                    location=aircraft.location,
                    t_beg=aircraft.last_tstamp,
                    t_dur=dt.timedelta(hours=6),  # Line maintenance duration
                    workscope='EngineExchange',
                )

                # if aircraft.Engines.uid < 999:
                #     x_time = aircraft.Engines.history['TIME'][-100:]
                #     y_egtm = aircraft.Engines.history['EGTM'][-100:]
                #     x_time_num = [(el - x_time[0]).total_seconds() / (3600 * 24) for el in x_time]
                #     a, b = np.polyfit(x_time_num, y_egtm, 1)
                #
                #     # Append data to CSV file
                #     with open(csv_filename, mode="a", newline="") as file:
                #         writer = csv.writer(file)
                #         writer.writerow([aircraft.Engines.uid, aircraft.Engines.goal_fh_fc_ratio, a, b])




                # Detach engine from aircraft
                engine4shop = aircraft.detach_engines(mng.SimTime)

                # Perform the maintenance via the Engines class
                engine4shop.restore(mng.SimTime)

                # Move the engine to the shop
                mng.engines_in_shop.append(engine4shop)

                # Replace the engine with a spare, creating a new one if necessary
                try:
                    spare_engine = mng.engines_in_pool.pop(0)
                except IndexError:
                    mng.num_needed_spares += 1
                    logging.info(f"Need {mng.num_needed_spares} spare engines.")
                    spare_engine = Engines(uid=1000 + mng.num_needed_spares, config=mng.config)
                    prog_mng.add_engine_esv_plan(spare_engine)

                aircraft.attach_engines(spare_engine, mng.SimTime)
                # spare_engine.attach_aircraft(aircraft)
                aircraft.add_event(maintenance_event)



        # Simulate a Turnaround Event
        tat_hrs = np.random.uniform(low=aircraft.avg_tat_hrs_min, high=aircraft.avg_tat_hrs_max)
        tat = TurnaroundEvent(
            location=aircraft.location,
            t_beg=aircraft.last_tstamp,
            t_dur=dt.timedelta(hours=tat_hrs)
        )
        aircraft.add_event(tat)
        aircraft.last_tstamp = tat.t_end

        # Simulate a Flight Event
        flight_params = aircraft.next_flights.pop(0)
        #flight_params['t_beg'] = aircraft.last_tstamp
        flightevent = FlightEvent(**flight_params, t_beg=aircraft.last_tstamp)
        aircraft.add_event(flightevent)

        # Update global clock
        mng.SimTime = aircraft.last_tstamp

        yemo = (mng.SimTime.year, mng.SimTime.month)
        if yemo not in mng.yemo:

            yemo_dt = dt.datetime(year=yemo[0], month=yemo[1], day=1)

            new_aircraft_added = False
            for aircraft in mng.aircraft_inactive[:]:
                if yemo_dt >= aircraft.eis:
                    mng.aircraft_active.append(aircraft)
                    mng.aircraft_inactive.remove(aircraft)
                    new_aircraft_added = True

            if new_aircraft_added:
                mng = tap.initial(mng)

            if predict_rul_frequency == 'monthly':
                prog_mng.predict_rul()
                prog_mng.fix_esv_planning_with_fhratio()
                prog_mng.fix_esv_planning()
            elif predict_rul_frequency == 'annually':
                if yemo[0] not in [el[0] for el in mng.yemo]:
                    prog_mng.predict_rul()
                    prog_mng.fix_esv_planning_with_fhratio()
                    prog_mng.fix_esv_planning()

            else:
                if str(yemo) == predict_rul_frequency:
                    prog_mng.predict_rul()
                    prog_mng.fix_esv_planning_with_fhratio()
                    prog_mng.fix_esv_planning()

            # print("rescheduling on %s" % str(mng.SimTime))
            mng = tap.reschedule(mng)

            # Check if engines in the shop are ready to be returned
            for idx, engine in enumerate(mng.engines_in_shop):
                if mng.SimTime > engine.esv_until:
                    mng.engines_in_pool.append(engine)
                    del mng.engines_in_shop[idx]

            mng.yemo.append(yemo)

        if mng.SimTime >= mng.EndTime:
            return mng


if __name__ == "__main__":
    # csv_filename = "simulation_results.csv"
    # headers = ['Engine UID', 'FH/FC-Ratio', 'a', 'b']
    # with open(csv_filename, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(headers)

    manager = main()
    post.visualize_health(manager)
    post.minimal_report(manager)
