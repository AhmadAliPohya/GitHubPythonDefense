# /main.py

import logging
import configparser as cp
from classes.aircraft import Aircraft, Engines
import classes.tailassignment as tap
import classes.prediction as pred
from classes.events import FlightEvent, TurnaroundEvent, MaintenanceEvent
import numpy as np
import datetime as dt
from post import post
from utils.debug_plots import *

# Configure logging to write debug and above to a file called log.txt
logging.basicConfig(
    filename='log.txt',  # file path
    filemode='w',  # <--- Overwrite the file each time
    level=logging.DEBUG,  # minimum severity level to capture
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Only display to the nearest second
)


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
        self.time_bounds = []
        self.current_time = None
        self.num_needed_spares = 0
        self.aog_events = 0
        self.event_calendar = None
        self.config = None
        self.mro_base = None
        self.aircraft_fleet = []
        self.shop_engine_fleet = []
        self.spare_engine_fleet = []
        self.yemo = []

        # Read configuration
        self.read_config(config_path)

        # Initiate Simulation Time
        SimTime_str = self.config.get('Simulation', 'initial_time')
        self.SimTime = dt.datetime.strptime(SimTime_str, '%Y-%m-%d %H:%M:%S')
        sim_duration = self.config.getint('Simulation', 'sim_duration')
        self.EndTime = self.SimTime + dt.timedelta(days=365 * sim_duration)

        # Initialize settings
        self.mro_base = self.config.get('Other', 'mro_base')
        if self.config.get('Simulation', 'maint_strategy') == 'predictive':
            self.predictive = True
        else:
            self.predictive = False

        # Create aircraft fleet
        num_aircraft = self.config.getint('Aircraft', 'num_aircraft')
        self.aircraft_fleet = self.create_aircraft_fleet(num_aircraft)
        self.no_aircraft = len(self.aircraft_fleet)

        # Create and attach engine sets to aircraft
        self.create_and_attach_engines(num_aircraft, self.aircraft_fleet)

        logging.info("Initialization complete. Ready to start operations.")

    def read_config(self, file_path: str):
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
        return aircraft_fleet

    def create_and_attach_engines(self, num_aircraft: int, aircraft_fleet: list) -> list:
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
        engine_fleet = []
        for n in range(num_aircraft):
            engines = Engines(uid=n, config=self.config, aircraft=aircraft_fleet[n])
            aircraft_fleet[n].attach_engines(engines, self.SimTime)
            #engines.attach_aircraft(aircraft_fleet[n])
            engine_fleet.append(engines)
        logging.info("Created and attached %d engine sets" % len(engine_fleet))
        return engine_fleet

    def create_spare_engines(self, num_aircraft: int, spare_engine_ratio: float) -> list:
        """
        Creates a fleet of spare engines.

        Parameters
        ----------
        num_aircraft : int
            Number of aircraft in the fleet.
        spare_engine_ratio : float
            Ratio of spare engines to aircraft.

        Returns
        -------
        list of Engines
            List of created spare engine objects.
        """
        num_spare_engines = int(np.ceil(spare_engine_ratio * num_aircraft))
        spare_engine_fleet = [
            Engines(uid=num_aircraft + n, config=self.config) for n in range(num_spare_engines)
        ]
        logging.info("Created %d spare engine sets" % len(spare_engine_fleet))
        return spare_engine_fleet

    def calculate_scope(self, efc, esv_time):
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

        # delta_t = esv_time - self.SimTime
        # delta_fc = efc
        # avg_tat = self.config.getfloat('Aircraft', 'avg_tat_hrs')
        # a = delta_t / delta_fc - dt.timedelta(hours=avg_tat)

        # 6 FH/FC for delay (longer flights) and 2 FH/FC for early (shorter flights)
        lower_bound = self.SimTime + dt.timedelta(hours=efc * (2 + self.config.getfloat('Aircraft', 'avg_tat_hrs')))
        upper_bound = self.SimTime + dt.timedelta(hours=efc * (6 + self.config.getfloat('Aircraft', 'avg_tat_hrs')))

        return lower_bound, upper_bound

    def find_opportunity_windows(self, scope_start, scope_end):
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
        sorted_bounds = sorted(self.time_bounds, key=lambda x: x['START'])

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
            if gap_start < gap_end and (gap_end - gap_start).days >= 35:  # Check if at least 5 weeks
                windows.append((gap_start, gap_end))

        # Handle the gap after the last ESV
        if merged_bounds:
            last_bound_end = merged_bounds[-1]['END']
            if scope_end > last_bound_end:
                post_gap_start = last_bound_end
                post_gap_end = scope_end
                if (post_gap_end - post_gap_start).days >= 35:  # Check if at least 5 weeks
                    windows.append((post_gap_start, post_gap_end))

        return windows

    def find_overlaps(self):

        # Initialize variables for overlap detection
        overlapping_groups = []  # List to store groups of overlapping ESVs
        visited = set()  # Set to track UIDs already processed

        # Compare each ESV's time bounds with all others
        for i, current in enumerate(self.time_bounds):
            if current['UID'] in visited:  # Skip UIDs already grouped
                continue

            group = [current['UID']]  # Start a new group with the current UID

            # Check for overlaps with other ESVs
            for j, other in enumerate(self.time_bounds):
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

    def identify_time_bounds(self, esv_plan):

        self.time_bounds = [
            {
                'UID': uid,
                'START': time - dt.timedelta(weeks=4),  # Start time with uncertainty
                'END': time + dt.timedelta(weeks=3 + 4)  # End time with uncertainty
            }
            for uid, time in zip(esv_plan['UID'], esv_plan['TIME'])
        ]

    def update_esv_plans(self):
        """
        Updates the ESV (Engine Shop Visit) plans for all aircraft in the fleet,
        identifies overlapping ESVs, and redistributes them to avoid conflicts.

        This method is called periodically in the simulation to ensure ESV
        schedules are optimized.
        """

        overwrite_uid = self.config.getint('Other', 'ac_overwrite_uid')
        if overwrite_uid >= 0:
            ac_index = next((i for i, obj in enumerate(self.aircraft_fleet) if obj.uid == overwrite_uid))
            self.aircraft_fleet[ac_index].goal_fh_fc_ratio = self.config.getfloat('Other', 'ac_overwrite_fhr')
            return

        if not self.predictive:
            return

        # Log the adjustment details
        logging.info("(%s) | Updating Engine's Goal EFH/EFC Ratio"
                     % self.SimTime.strftime("%Y-%m-%d %H:%M:%S"))

        # Step 1: Initialize variables
        n_aircraft = len(self.aircraft_fleet)  # Total number of aircraft in the fleet
        rul_available = 0  # Counter for aircraft with Remaining Useful Life (RUL) available
        esv_plan = {'UID': [], 'OBJs': [], 'TIME': [], 'EFCs': []}  # Plan structure to store ESV data

        # Step 2: Populate the ESV plan for each aircraft with available RUL data
        for aircraft in self.aircraft_fleet:
            engine = aircraft.Engines
            if hasattr(engine, 'next_esv'):  # Check if the engine has a defined next ESV
                rul_available += 1
                esv_plan['UID'].append(engine.uid)  # Add engine UID
                esv_plan['TIME'].append(engine.next_esv['TIME'])  # Add ESV time
                esv_plan['EFCs'].append(engine.next_esv['EFCs'])  # Add EFCs until ESV
                esv_plan['OBJs'].append(engine)

        # If not all aircraft have available RUL data, exit early
        if rul_available < n_aircraft:
            return

        # Step 3: Identify overlapping ESVs using time bounds
        # Define time windows for each ESV (1 week before to 3 weeks after)
        self.identify_time_bounds(esv_plan)

        overlapping_groups = self.find_overlaps()

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
                    scopes[uid] = self.calculate_scope(efc, time)

                no_opp_windows_found = 0

                for uid in group:

                    # Find the index of the current UID in the esv_plan['UID'] list again
                    index = esv_plan['UID'].index(uid)

                    scope_start, scope_end = scopes[uid]  # Get scope bounds for this engine
                    opportunity_windows = self.find_opportunity_windows(scope_start, scope_end)

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
                    avg_tat_hrs = self.config.getfloat('Aircraft', 'avg_tat_hrs')

                    time_diff = target_time - self.SimTime
                    new_fh_fc_ratio = ((time_diff / remaining_efc) - dt.timedelta(
                        hours=avg_tat_hrs)).total_seconds() / 3600

                    # Update the ESV plan with the adjusted time
                    esv_plan['OBJs'][index].set_fh_fc_ratio(new_fh_fc_ratio)
                    esv_plan['TIME'][index] = target_time
                    esv_plan['EFCs'][index] = new_fh_fc_ratio  # Update the EFCs with the new ratio

                    self.identify_time_bounds(esv_plan)

                    # Log the adjustment details
                    logging.info("(%s) | Updated Engine %d's Ratio to %.2f"
                                 % (self.SimTime.strftime("%Y-%m-%d %H:%M:%S"), esv_plan['OBJs'][index].uid,
                                    new_fh_fc_ratio))

            if no_opp_windows_found == len(group):
                break
            overlapping_groups = self.find_overlaps()

        return


def find_next_aircraft(aircraft_fleet):
    # Find the aircraft with the earliest next timestamp
    idx_aircraft = 0
    tstamp = aircraft_fleet[0].last_tstamp
    for idx, aircraft in enumerate(aircraft_fleet):
        tstamp_idx = aircraft.last_tstamp
        if tstamp_idx < tstamp:
            tstamp = tstamp_idx
            idx_aircraft = idx

    return idx_aircraft


def main():
    """
    Main function to initialize and run the simulation.
    """
    mng = SimulationManager(config_path='config.ini')

    # Perform initial tail assignment
    mng = tap.initial(mng)

    mng.yemo.append((mng.SimTime.year, mng.SimTime.month))

    # Read execution frequencies from the config file
    predict_rul_frequency = mng.config['Simulation'].get('predict_rul', 'monthly')

    while True:

        # Find the next aircraft to process
        idx_aircraft = find_next_aircraft(mng.aircraft_fleet)
        aircraft = mng.aircraft_fleet[idx_aircraft]

        # Check for engine maintenance needs
        if aircraft.Engines.maintenance_due():

            skip_maintenance = False

            if aircraft.location != mng.mro_base:

                # here it matters what the ESV driver is.
                # If a random failure occurred, the engine is not operable.
                # This means that the airline either has to try to send another
                # aircraft or distribute passengers and whatnot - or - we just
                # cancel this flight. Either way, the aircraft/engine will stay
                # "grounded" for 24 hours. Think of this as the waiting time until
                # the spare engines come in by freighter.

                if aircraft.Engines.critical_soh_due and not aircraft.Engines.predictive:

                    downtime_event = MaintenanceEvent(
                        location=aircraft.location,
                        t_beg=aircraft.last_tstamp,
                        t_dur=dt.timedelta(hours=24),
                        workscope='AircraftOnGround',
                    )

                    aircraft.add_event(downtime_event)
                    aircraft.aog_events += 1
                    mng.aog_events += 1

                else:

                    # Here, we just schedule the maintenance to the next time
                    # when the aircraft is at its MRO base. Because the MRO
                    # base is the most likely the most frequent travelled
                    # airport, it'll be visited within the timeframe anyways
                    # without our need for intervention

                    if mng.mro_base in [el['dest'] for el in aircraft.next_flights]:

                        skip_maintenance = True

                    else:

                        # Just do maintenance there (assumption, otherwise I'd
                        # have to recall the reschedule thing and have it
                        # choose the base with higher priority - this complexity
                        # is not worth it).
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

                # Detach engine from aircraft
                engine4shop = aircraft.detach_engines(mng.SimTime)

                # Perform the maintenance via the Engines class
                engine4shop.restore(mng.SimTime)

                # Move the engine to the shop
                mng.shop_engine_fleet.append(engine4shop)

                # Replace the engine with a spare, creating a new one if necessary
                try:
                    spare_engine = mng.spare_engine_fleet.pop(0)
                except IndexError:
                    mng.num_needed_spares += 1
                    logging.info(f"Need {mng.num_needed_spares} spare engines.")
                    spare_engine = Engines(uid=1000 + mng.num_needed_spares, config=mng.config)

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

            if predict_rul_frequency == 'monthly':
                mng = pred.predict_rul(mng)
                mng.update_esv_plans()
            elif predict_rul_frequency == 'annually':
                if yemo[0] not in [el[0] for el in mng.yemo]:
                    mng = pred.predict_rul(mng)
                    mng.update_esv_plans()
            else:
                if str(yemo) == predict_rul_frequency:
                    mng = pred.predict_rul(mng)
                    mng.update_esv_plans()

            # print("rescheduling on %s" % str(mng.SimTime))
            mng = tap.reschedule(mng)

            # Check if engines in the shop are ready to be returned
            for idx, engine in enumerate(mng.shop_engine_fleet):
                if mng.SimTime > engine.esv_until:
                    mng.spare_engine_fleet.append(engine)
                    del mng.shop_engine_fleet[idx]

            mng.yemo.append(yemo)

        if mng.SimTime >= mng.EndTime:
            return mng


def should_execute(mng, frequency, yemo):
    """
    Determine if a task should be executed based on its frequency.

    Args:
        mng: The management object.
        frequency (str): The execution frequency ('monthly', 'annually', 'once').
        yemo (tuple): The current year and month.

    Returns:
        bool: True if the task should be executed, False otherwise.
    """
    if frequency == 'monthly':
        return True
    elif frequency == 'annually':
        # Execute only in the first month of the year
        return yemo[1] == 1
    elif frequency == 'once':
        # Execute only in the first year-month of the simulation
        return len(mng.yemo) == 0
    return False


if __name__ == "__main__":
    manager = main()
    post.visualize_health(manager)
    post.minimal_report(manager)
