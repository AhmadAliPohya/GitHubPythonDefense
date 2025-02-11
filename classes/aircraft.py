import logging
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

class Aircraft:
    def __init__(self, uid, config):
        # Store basic attributes
        self.uid = uid
        self.config = config

        self.age = dt.timedelta(hours=0) # Aircraft age as a timedelta
        self.fc_counter = 0  # Flight cycle count
        self.fh_counter = dt.timedelta(hours=0)  # Flight hours as a timedelta
        self.set_entries_into_service()


        # Initialize empty lists and state variables
        self.event_calendar = []  # List to store scheduled or past events
        self.next_flights = []  # List for upcoming flights
        self.current_state = None  # Placeholder for the aircraft's operational state
        self.Engines = None  # Placeholder for attached engines
        self.location = None  # Current location of the aircraft
        self.last_tstamp = None  # Timestamp of the last recorded event
        self.aog_events = 0
        self.past_flights = []
        self.util_category = None
        self.scheduled_until = dt.timedelta(hours=0)
        self.next_flights_fh_fc_ratio = 0.
        self.goal_fh_fc_ratio = 0.


    def set_entries_into_service(self, first=False):

        if first:
            random_agedelta = 0
        else:
            age_range = self.config.getint('Aircraft', 'age_range')
            if age_range == 0:
                random_agedelta = 0
            else:
                age_range_months = 12*age_range
                random_agedelta = np.random.randint(0, age_range_months)

        SimTime_str = self.config.get('Simulation', 'initial_time')
        sim_start = dt.datetime.strptime(SimTime_str, '%Y-%m-%d %H:%M:%S')
        yemo_start = dt.datetime(year=sim_start.year, month=sim_start.month, day=sim_start.day)

        self.eis = yemo_start + relativedelta(months=random_agedelta)



    def attach_engines(self, engine_set, current_time=None):
        """
        Attach a set of engines to the aircraft.

        Parameters
        ----------
        engine_set : Any
            The engine (or engines) to be attached to the aircraft.
        """
        self.Engines = engine_set  # Store the engine set in the aircraft attribute
        self.Engines.attach_aircraft(self, current_time)

    def detach_engines(self, current_time):
        self.Engines.detach_aircraft(current_time)
        engine2return = self.Engines
        self.Engines = None
        return engine2return

    def add_event(self, event):
        """
        Add an event to the aircraft's event calendar and update its state.

        Parameters
        ----------
        event : Any
            An event object that includes:

            - t_end (datetime.datetime): End time of the event.
            - location (str): The location where the event finishes.
            - t_dur (datetime.timedelta): Duration of the event.
            - type (str): Type of event (e.g., 'Flight', 'Maintenance').

        Notes
        -----
        If the event is a flight, this method increments flight cycles (fc_counter)
        and flight hours (fh_counter) for both the aircraft and its engines, and calls
        the engines' deteriorate method.
        """
        # Append event to the calendar
        self.event_calendar.append(event)

        # Update relevant attributes based on the event
        self.last_tstamp = event.t_end
        self.location = event.location
        self.age += event.t_dur
        self.Engines.age += event.t_dur

        # If this event is a flight, increment counters and deteriorate engines
        if event.type == 'Flight':
            self.fc_counter += 1
            self.fh_counter += event.t_dur
            self.Engines.fc_counter += 1
            self.Engines.fc_since_esv += 1
            self.Engines.fh_counter += event.t_dur
            self.Engines.deteriorate(event)
            self.past_flights.append(event)

    def add_next_flight(self, next_flight_info):
        self.next_flights.append(next_flight_info)
        self.scheduled_until += next_flight_info['t_dur']
        self.next_flights_fh_fc_ratio = (
                self.scheduled_until.total_seconds() / (3600 * len(self.next_flights)))

    def __repr__(self):
        """
        Return a string representation of the aircraft.

        Returns
        -------
        str
            A string describing the aircraft's location and last timestamp.
        """
        # Return a formatted string with location and timestamp
        return f"Aircraft in [{self.location}] at {self.last_tstamp.strftime('%Y-%m-%d %H:%M')}"


