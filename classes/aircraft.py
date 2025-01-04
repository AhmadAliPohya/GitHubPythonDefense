import logging
import numpy as np
from scipy.stats import truncnorm
import datetime as dt


class Aircraft:
    """
    Class representing an aircraft and its associated state (age, flight cycles, etc.).

    Parameters
    ----------
    uid : int
        Unique identifier for the aircraft.
    config : configparser.ConfigParser
        Configuration object that contains relevant settings for the aircraft.

    Attributes
    ----------
    uid : int
        Unique identifier for the aircraft.
    config : configparser.ConfigParser
        Configuration parser with settings under the 'Aircraft' section.
    age : datetime.timedelta
        Timedelta representing the age of the aircraft.
    fc_counter : int
        The number of completed flight cycles for the aircraft.
    fh_counter : datetime.timedelta
        The accumulated flight hours for the aircraft, represented as a timedelta.
    event_calendar : list
        A list of events (maintenance, flight, etc.) associated with the aircraft.
    next_flights : list
        A placeholder list for upcoming flights that the aircraft will operate.
    current_state : Any
        Current operational state of the aircraft (could be an enum, string, etc.).
    Engines : Any
        The engine set (or engine object) attached to this aircraft.
    location : str
        The current location of the aircraft (e.g., airport ICAO code).
    last_tstamp : datetime.datetime
        Timestamp of the aircraft’s last recorded event.
    """

    def __init__(self, uid, config):

        # Store basic attributes
        self.uid = uid
        self.config = config

        # Generate random age, flight cycles, and flight hours based on config
        self.age = self._generate_random_age()  # Aircraft age as a timedelta
        self.fc_counter = self._generate_fc_counter()  # Flight cycle count
        self.fh_counter = self._generate_fh_counter()  # Flight hours as a timedelta

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

    def _generate_random_age(self):
        """
        Generate a random age for the aircraft using configuration bounds.

        Returns
        -------
        datetime.timedelta
            The randomly generated age of the aircraft.
        """
        # Read min and max ages in years from config
        min_age = self.config.getint('Aircraft', 'min_age')
        max_age = self.config.getint('Aircraft', 'max_age')

        # Uniformly sample between min and max age
        random_age_in_years = np.random.uniform(min_age, max_age)

        # Convert years to days (approx. 365 days/year)
        random_age_in_days = random_age_in_years * 365

        # Return the age as a timedelta object
        return dt.timedelta(days=random_age_in_days)

    def _generate_fc_counter(self):
        """
        Generate the flight cycle counter using a truncated normal distribution.

        Returns
        -------
        int
            The flight cycle count for the aircraft.
        """
        # Retrieve average flight cycles per year and std deviation from config
        avg_fc_per_year = self.config.getfloat('Aircraft', 'avg_fc_per_year')
        std_fc_per_year = self.config.getfloat('Aircraft', 'std_fc_per_year')

        # Define lower and upper bounds for truncated normal
        # Ensure a minimum of 12 cycles/year in the worst case (example)
        lower_bound = max(avg_fc_per_year - 2 * std_fc_per_year, 12)
        upper_bound = avg_fc_per_year + 2 * std_fc_per_year

        # Sample a realistic flight cycle rate from truncated normal
        fc_per_year = self._sample_truncated_normal(
            mean=avg_fc_per_year,
            std=std_fc_per_year,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        # Convert the aircraft's age (in seconds) to years
        age_years = self.age.total_seconds() / (60 * 60 * 24 * 365)

        # Calculate the total flight cycles (convert to int)
        return int(fc_per_year * age_years)

    def _generate_fh_counter(self):
        """
        Generate the flight hour counter using a truncated normal distribution.

        Returns
        -------
        datetime.timedelta
            Total flight hours as a timedelta.
        """
        # Retrieve average flight hours per flight cycle and std deviation
        avg_fh_per_fc = self.config.getfloat('Aircraft', 'avg_fh_per_fc')
        std_fh_per_fc = self.config.getfloat('Aircraft', 'std_fh_per_fc')

        # Define lower and upper bounds for truncated normal
        # Ensure a minimum of 0.5 flight hours per cycle in the worst case (example)
        lower_bound = max(avg_fh_per_fc - 2 * std_fh_per_fc, 0.5)
        upper_bound = avg_fh_per_fc + 2 * std_fh_per_fc

        # Sample a realistic flight hours-per-cycle from truncated normal
        fh_per_fc = self._sample_truncated_normal(
            mean=avg_fh_per_fc,
            std=std_fh_per_fc,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        # Multiply by the current fc_counter to get total hours
        fh_float = fh_per_fc * self.fc_counter

        # Return flight hours as a timedelta object
        return dt.timedelta(hours=fh_float)

    @staticmethod
    def _sample_truncated_normal(mean, std, lower_bound, upper_bound):
        """
        Sample a value from a truncated normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the original (non-truncated) normal distribution.
        std : float
            Standard deviation of the original normal distribution.
        lower_bound : float
            Lower bound for the truncated distribution.
        upper_bound : float
            Upper bound for the truncated distribution.

        Returns
        -------
        float
            A single sample drawn from the truncated normal distribution.
        """
        # Convert bounds to 'a' and 'b' for scipy's truncnorm
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std

        # Return the random sample from the truncated normal distribution
        return truncnorm.rvs(a, b, loc=mean, scale=std)

    def attach_engines(self, engine_set):
        """
        Attach a set of engines to the aircraft.

        Parameters
        ----------
        engine_set : Any
            The engine (or engines) to be attached to the aircraft.
        """
        self.Engines = engine_set  # Store the engine set in the aircraft attribute

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


class Engines:
    """
    Class representing a set of engines and their operational state (deterioration,
    maintenance thresholds, etc.).

    Parameters
    ----------
    uid : int
        Unique identifier for the engine set.
    config : configparser.ConfigParser
        Configuration parser with relevant settings for engines.
    aircraft : Aircraft, optional
        An optional Aircraft instance to which these engines are attached. If provided,
        engine attributes will be derived from the aircraft's current state.

    Attributes
    ----------
    uid : int
        Unique identifier for the engine set.
    config : configparser.ConfigParser
        Configuration parser containing the 'Engine' section with thresholds and parameters.
    event_calendar : list
        A list of events associated with the engines (maintenance, inspections, etc.).
    current_state : Any
        Current state of the engines (could be an enum, string, or custom state object).
    egtm_init : float
        Initial Exhaust Gas Temperature Margin (EGTM) setting for new or restored engines.
    llp_life_init : float
        Initial remaining life (in EFC, Engine Flight Cycles) for the Life-Limited Parts (LLPs).
    egt_resets : int
        Count of how many times the engine’s EGTM has been restored.
    llp_resets : int
        Count of how many times the engine’s LLP life has been restored.
    egti_per_fc : float
        EGTM increase rate per flight cycle, scaled by 1/1000 for smaller increments.
    egtm_due : bool
        Whether an EGTM-based restoration is immediately due.
    llp_due : bool
        Whether an LLP-based restoration is immediately due.
    random_due : bool
        Whether a random failure event has occurred, indicating maintenance is due.
    esv_until : datetime.datetime
        Tracks the date until which the engine is unavailable (e.g., during scheduled maintenance).
    history : dict
        Dictionary storing historical states of 'EGTM', 'LLP', 'TIME', and 'EFCs' (Engine Flight Cycles).
    failure_efcs : list
        Sorted list of flight cycle counts at which random failures are expected to occur.
    age : datetime.timedelta
        Current age of the engine set, taken from the aircraft’s age if attached.
    initial_age : datetime.timedelta
        Age of the engines at the time of initialization.
    fc_counter : int
        Current number of flight cycles, taken from the aircraft if attached.
    fh_counter : datetime.timedelta
        Accumulated flight hours, taken from the aircraft if attached.
    egtm : float
        Current EGTM value, decreased over time as the engine deteriorates.
    llp_life : float
        Current remaining LLP life in engine flight cycles (EFC).
    critical_egtm_due : bool
        Internal indicator for critical-level EGTM threshold.
    warning_egtm_due : bool
        Internal indicator for warning-level EGTM threshold.
    critical_llp_due : bool
        Internal indicator for critical-level LLP threshold.
    warning_llp_due : bool
        Internal indicator for warning-level LLP threshold.
    """

    def __init__(self, uid, config, aircraft=None):
        """
        Initialize an Engines instance, optionally attached to an Aircraft.

        Parameters
        ----------
        uid : int
            Unique identifier for the engines.
        config : configparser.ConfigParser
            Configuration object with thresholds and parameters (under the 'Engine' section).
        aircraft : Aircraft, optional
            If provided, references the Aircraft object to copy age, flight cycles, and flight
            hours for engine initialization.
        """
        # Store basic identifiers
        self.warning_egtm_due = False
        self.critical_egtm_due = False
        self.warning_llp_due = False
        self.critical_llp_due = False
        self.uid = uid
        self.config = config
        self.event_calendar = []
        self.current_state = None

        # Read engine thresholds and counters from config
        self.egtm_init = config.getfloat('Engine', 'egtm_init')
        self.llp_life_init = config.getfloat('Engine', 'llp_life_init')
        self.egt_resets = 0
        self.llp_resets = 0

        # component state of health
        self.random_soh = 1

        # EGTM increment per flight cycle (scaled down by 1/1000)
        self.egti_per_fc = config.getfloat('Engine', 'egti') / 1000

        # Booleans that track whether the engine is due for certain maintenance
        self.egtm_due = False
        self.llp_due = False
        self.random_due = False

        # Tracks until which date the engine is out of service
        self.esv_until = None

        # Dictionary to keep history of engine parameters over time
        self.history = {'EGTM': [], 'LLP': [], 'SOH': [],
                        'TIME': [], 'EFCs': [], 'EFHs': []}

        # Retrieve the mean time between failures (MTBF) in EFC from config
        mtbf_efh = config.getfloat('Engine', 'mtbf_efh')
        if mtbf_efh == 0:
            mtbf_efh = 1000
            self.random_failures = False
        else:
            self.random_failures = True

        self.random_params = {'a': -np.random.normal(1/mtbf_efh, 0.2*(1/mtbf_efh)),
                              'b': 1}

        # If no aircraft is provided, initialize engine attributes to zero or defaults
        if aircraft is None:
            self.age = dt.timedelta(days=0)
            self.initial_age = self.age
            self.fc_counter = 0
            self.fh_counter = dt.timedelta(hours=0)
            self.egtm = self.egtm_init
            self.llp_life = self.llp_life_init
            self.random_soh = 1
        else:
            # If attached to an aircraft, synchronize engine attributes
            self.age = aircraft.age
            self.initial_age = self.age
            self.fc_counter = aircraft.fc_counter
            self.fh_counter = aircraft.fh_counter

            # Adjust the initial EGTM, LLP, and random life based on existing usage
            self.egtm = self.egtm_init - self.fc_counter * self.egti_per_fc
            self.llp_life = self.llp_life_init - self.fc_counter
            self.random_soh = (
                    self.random_params['a']*self.fh_counter.total_seconds()/3600
                    + self.random_params['b'])

            # If EGTM has dropped below a threshold, increment resets until it's above 7.5
            while self.egtm < 7.5:
                self.egtm += self.egtm_init
                self.egt_resets += 1

            # If LLP life has dropped below 1750 EFC, increment resets until above that threshold
            while self.llp_life < 1750:
                self.llp_life += self.llp_life_init
                self.llp_resets += 1

            while self.random_soh < 0:
                self.random_soh += 1

            # Remove any random failures that occurred before the current fc_counter
            # so the next failure will be in the future
            #while self.failure_efcs and self.failure_efcs[0] < self.fc_counter:
            #    del self.failure_efcs[0]


    def attach_aircraft(self, aircraft):

        self.aircraft = aircraft

    def deteriorate(self, flight):
        """
        Simulate engine deterioration during a flight.

        Parameters
        ----------
        flight : Any
            An object representing a flight, which should include:

            - t_end (datetime.datetime): The end time of the flight.
            - t_dur (datetime.timedelta): The duration of the flight (not directly used here,
              but might be needed in advanced calculations).
            - Additional attributes as needed by the simulation.
        """
        # Decrease the EGTM by a random increment based on engine maturity
        self.egtm -= self.egt_increase()
        # Decrease LLP life by 1 EFC for every flight
        self.llp_life -= 1
        # Decrease SOH of component
        self.random_soh -= self.soh_update(flight.t_dur)

        # Record the updated engine state to the history dictionary
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['SOH'].append(self.random_soh)
        self.history['TIME'].append(flight.t_end)
        self.history['EFCs'].append(self.fc_counter)
        self.history['EFHs'].append(self.fh_counter)

    def soh_update(self, duration):

        new_soh = (self.random_params['a']*duration.total_seconds()/3600
                   + self.random_params['b'])
        increment = 1 - new_soh

        return increment*np.random.uniform(-1, 3)

    def egt_increase(self):
        """
        Calculate the incremental deterioration in EGTM based on engine maturity.

        Returns
        -------
        float
            A random increment (in degrees Celsius) to reduce the EGTM value by.
        """

        efc_rate = 3 if self.fc_counter > 1000 else 8
        efc_rate_with_noise = efc_rate * np.random.uniform(-1, 3)
        return efc_rate_with_noise / 1000

    def maintenance_due(self):
        """
        Determine if maintenance is required based on EGTM, LLP, or random
        failure thresholds.

        Returns
        -------
        bool
            True if any of the critical thresholds are met, otherwise False.
        """
        # Critical threshold for EGTM
        self.critical_egtm_due = self.egtm < 5
        # Warning threshold for EGTM
        self.warning_egtm_due = 5 <= self.egtm < 10

        # Critical threshold for LLP
        self.critical_llp_due = self.llp_life < 500
        # Warning threshold for LLP
        self.warning_llp_due = 500 <= self.llp_life < 3000

        # Check if the next random failure is past the current fc_counter
        if self.random_failures:
            self.random_due = self.random_soh < 0
        else:
            self.random_due = False

        # If any critical or random due condition is met, return True
        if self.critical_egtm_due or self.critical_llp_due or self.random_due:
            return True
        else:
            return False

    def restore(self, SimTime):
        """
        Perform restorations based on current engine condition.

        Parameters
        ----------
        SimTime : datetime.datetime
            The current simulation time.

        Notes
        -----
        - Critical restorations for EGTM or LLP automatically cover the
          respective warning-level condition.
        - If a random failure has occurred, a partial restoration is performed
          and the engine is marked unavailable for a shorter period (7 days)
          compared to other maintenance (25 days).
        """
        # Print a maintenance log message
        logging.info("ESV for Engine %d on %s at %d EFCs"
              % (self.uid, SimTime.strftime("%d.%m.%Y %H:%M"), self.fc_counter))

        # Handle random failure repairs first
        if self.random_due:
            self.random_restoration(SimTime)
            # If EGTM was at warning level, also restore EGTM
            if self.warning_egtm_due:
                self.egtm_restoration(SimTime)
            # If LLP was at warning level, also restore LLP
            if self.warning_llp_due:
                self.llp_restoration(SimTime)

        # Handle critical EGTM restoration
        if self.critical_egtm_due:
            self.egtm_restoration(SimTime)
            # Also handle warning-level LLP if present
            if self.warning_llp_due:
                self.llp_restoration(SimTime)

        # Handle critical LLP restoration
        if self.critical_llp_due:
            self.llp_restoration(SimTime)
            # Also handle warning-level EGTM if present
            if self.warning_egtm_due:
                self.egtm_restoration(SimTime)

        # Record the updated engine state after restoration
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['SOH'].append(self.random_soh)
        self.history['TIME'].append(SimTime)
        self.history['EFCs'].append(self.fc_counter)

    def egtm_restoration(self, SimTime):
        """
        Restore EGTM to near-initial state and mark engine as unavailable
        for 25 days.

        Parameters
        ----------
        SimTime : datetime.datetime
            The current simulation time.
        """
        # Store old EGTM for logging
        old_egtm = self.egtm
        # New EGTM is the initial value minus a random uniform factor
        new_egtm = self.egtm_init - np.random.uniform(0, 5)

        # Increment EGTM reset count
        self.egt_resets += 1

        # Set the new EGTM value
        self.egtm = new_egtm

        # Maintenance resets the engine’s due flags
        self.egtm_due = False
        self.esv_until = SimTime + dt.timedelta(days=25)
        self.critical_egtm_due = False
        self.warning_egtm_due = False

        # Log the restoration action
        logging.info(" - EGTM restoration from %.1f°C to %.1f°C"
              % (old_egtm, new_egtm))

    def llp_restoration(self, SimTime):
        """
        Restore LLP life to the initial value and mark engine as unavailable
        for 25 days.

        Parameters
        ----------
        SimTime : datetime.datetime
            The current simulation time.
        """
        # Store old LLP life for logging
        old_llp_life = self.llp_life
        # Reset LLP life to its initial state
        new_llp_life = self.llp_life_init

        # Increment LLP reset count
        self.llp_resets += 1

        # Set the new LLP life value
        self.llp_life = new_llp_life

        # Maintenance resets the engine’s due flags
        self.llp_due = False
        self.esv_until = SimTime + dt.timedelta(days=25)
        self.critical_llp_due = False
        self.warning_llp_due = False

        # Log the restoration action
        logging.info(" - LLP-RUL restoration from %.1fEFC to %.1fEFC"
              % (old_llp_life, new_llp_life))

    def random_restoration(self, SimTime):
        """
        Perform a minimal restoration to address a random failure and mark
        engine as unavailable for 7 days.

        Parameters
        ----------
        SimTime : datetime.datetime
            The current simulation time.
        """
        # Clear the random failure flag and move the next random failure
        self.random_due = False
        self.esv_until = SimTime + dt.timedelta(days=7)
        self.random_soh = 1

        # Remove the failure count that just triggered the restoration
        # del self.failure_efcs[0]

        # Log the random failure restoration action
        logging.info(" - replacement of failed part")

    def set_fh_fc_ratio(self, new_fh_fc_ratio):

        self.goal_fh_fc_ratio = new_fh_fc_ratio
        self.aircraft.goal_fh_fc_ratio = new_fh_fc_ratio