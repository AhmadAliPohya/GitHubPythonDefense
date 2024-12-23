# /main.py

import configparser as cp
from classes.aircraft import Aircraft, Engines
import classes.tailassignment as tap
from classes.events import FlightEvent, TurnaroundEvent, MaintenanceEvent
import numpy as np
import datetime as dt
from post import post


class Manager:
    def __init__(self, config_path: str):
        """
        Initializes the Manager class with configuration, aircraft fleet,
        and engine tracking.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        """
        self.current_time = None
        self.num_needed_spares = 0
        self.ago_events = 0
        self.event_calendar = None
        self.config = None
        self.mro_base = None
        self.aircraft_fleet = []
        self.shop_engine_fleet = []
        self.spare_engine_fleet = []

        # Read configuration
        self.read_config(config_path)

        # Initialize settings
        self.mro_base = self.config.get('Other', 'mro_base')

        # Create aircraft fleet
        num_aircraft = self.config.getint('Aircraft', 'num_aircraft')
        self.aircraft_fleet = self.create_aircraft_fleet(num_aircraft)

        # Create and attach engine sets to aircraft
        self.create_and_attach_engines(num_aircraft, self.aircraft_fleet)

        # Initiate Simulation Time
        SimTime_str = self.config.get('Simulation', 'initial_time')
        self.SimTime = dt.datetime.strptime(SimTime_str, '%Y-%m-%d %H:%M:%S')
        sim_duration = self.config.getint('Simulation', 'sim_duration')
        self.EndTime = self.SimTime + dt.timedelta(days=365 * sim_duration)


        print("Initialization complete. Ready to start operations.")

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
        print("Created %d aircraft" % num_aircraft)
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
            aircraft_fleet[n].attach_engines(engines)
            engine_fleet.append(engines)
        print("Created and attached %d engine sets" % len(engine_fleet))
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
        print("Created %d spare engine sets" % len(spare_engine_fleet))
        return spare_engine_fleet


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
    mng = Manager(config_path='config.ini')

    # Perform initial tail assignment
    mng = tap.initial(mng)

    # Track the last time scheduling was performed
    last_scheduled_date = None

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

                if aircraft.Engines.random_due:

                    downtime_event = MaintenanceEvent(
                        location=aircraft.location,
                        t_beg=aircraft.last_tstamp,
                        t_dur=dt.timedelta(hours=24),
                        workscope='AircraftOnGround',
                    )

                    aircraft.add_event(downtime_event)

                else:

                    # Here, we just schedule the maintenance to the next time
                    # when the aircraft is at its MRO base. Because the MRO
                    # base is the most likely the most frequent travelled
                    # airport, it'll be visited within the timeframe anyways
                    # without our need for intervention

                    skip_maintenance = True

            if not skip_maintenance:

                # Add a maintenance event
                maintenance_event = MaintenanceEvent(
                    location=aircraft.location,
                    t_beg=aircraft.last_tstamp,
                    t_dur=dt.timedelta(hours=6),  # Line maintenance duration
                    workscope='EngineExchange',
                )

                # Perform the maintenance via the Engines class
                aircraft.Engines.restore(mng.SimTime)

                # Move the engine to the shop
                mng.shop_engine_fleet.append(aircraft.Engines)

                # Replace the engine with a spare, creating a new one if necessary
                try:
                    spare_engine = mng.spare_engine_fleet.pop(0)
                except IndexError:
                    mng.num_needed_spares += 1
                    print(f"Need {mng.num_needed_spares} spare engines.")
                    spare_engine = Engines(uid=1000 + mng.num_needed_spares, config=mng.config)

                aircraft.Engines = spare_engine
                aircraft.add_event(maintenance_event)

        # Simulate a Turnaround Event
        tat_hrs = np.random.uniform(low=aircraft.avg_tat_hrs_min, high=aircraft.avg_tat_hrs_max)
        tat = TurnaroundEvent(
            location=aircraft.location,
            t_beg=aircraft.last_tstamp,
            t_dur=dt.timedelta(hours=tat_hrs)
        )
        aircraft.event_calendar.append(tat)
        aircraft.last_tstamp = tat.t_end

        # Simulate a Flight Event
        flight_params = aircraft.next_flights.pop(0)
        flight_params['t_beg'] = aircraft.last_tstamp
        flightevent = FlightEvent(**flight_params)
        aircraft.add_event(flightevent)

        # Update global clock
        mng.SimTime = aircraft.last_tstamp

        # Reschedule if a week has passed
        if last_scheduled_date is None or (mng.SimTime - last_scheduled_date) >= dt.timedelta(days=7):
            mng = tap.reschedule(mng)
            last_scheduled_date = mng.SimTime

            # Check if engines in the shop are ready to be returned
            for idx, engine in enumerate(mng.shop_engine_fleet):
                if mng.SimTime > engine.esv_until:
                    mng.spare_engine_fleet.append(engine)
                    del mng.shop_engine_fleet[idx]

        # End simulation if the time limit is reached
        if mng.SimTime > mng.EndTime:
            print("Simulation Done")
            break

    return mng



if __name__ == "__main__":
    manager = main()
    post.postprocessingSOH(manager)