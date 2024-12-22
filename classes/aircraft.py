import math
import numpy as np
from scipy.stats import truncnorm
import datetime as dt


class Aircraft:
    def __init__(self, uid, config):
        """
        Initialize an Aircraft instance with configuration settings.
        """
        self.uid = uid
        self.config = config

        # Attributes for tracking the aircraft state
        self.age = self._generate_random_age()
        self.fc_counter = self._generate_fc_counter()
        self.fh_counter = self._generate_fh_counter()
        self.event_calendar = []
        self.next_flights = []
        self.current_state = None
        self.Engines = None
        self.location = None
        self.last_tstamp = None

    def _generate_random_age(self):
        """
        Generate a random age for the aircraft as a timedelta.
        """
        min_age = self.config.getint('Aircraft', 'min_age')
        max_age = self.config.getint('Aircraft', 'max_age')
        random_age_in_years = np.random.uniform(min_age, max_age)
        random_age_in_days = random_age_in_years * 365  # Approximate days per year
        return dt.timedelta(days=random_age_in_days)

    def _generate_fc_counter(self):
        """
        Generate the flight cycle counter based on a truncated normal distribution.
        """
        avg_fc_per_year = self.config.getfloat('Aircraft', 'avg_fc_per_year')
        std_fc_per_year = self.config.getfloat('Aircraft', 'std_fc_per_year')
        lower_bound = max(avg_fc_per_year - 2 * std_fc_per_year, 12)
        upper_bound = avg_fc_per_year + 2 * std_fc_per_year

        fc_per_year = self._sample_truncated_normal(avg_fc_per_year, std_fc_per_year, lower_bound, upper_bound)
        age_years = self.age.total_seconds() / (60 * 60 * 24 * 365)  # Convert age to years
        return int(fc_per_year * age_years)

    def _generate_fh_counter(self):
        """
        Generate the flight hour counter based on a truncated normal distribution.
        """
        avg_fh_per_fc = self.config.getfloat('Aircraft', 'avg_fh_per_fc')
        std_fh_per_fc = self.config.getfloat('Aircraft', 'std_fh_per_fc')
        lower_bound = max(avg_fh_per_fc - 2 * std_fh_per_fc, 0.5)
        upper_bound = avg_fh_per_fc + 2 * std_fh_per_fc

        fh_per_fc = self._sample_truncated_normal(avg_fh_per_fc, std_fh_per_fc, lower_bound, upper_bound)
        fh_float = fh_per_fc * self.fc_counter
        return dt.timedelta(hours=fh_float)

    @staticmethod
    def _sample_truncated_normal(mean, std, lower_bound, upper_bound):
        """
        Sample a value from a truncated normal distribution.
        """
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)

    def attach_engines(self, engine_set):
        """
        Attach a set of engines to the aircraft.
        """
        self.Engines = engine_set

    def add_event(self, event):
        """
        Add an event to the aircraft's event calendar and update its state.
        """
        self.event_calendar.append(event)
        self.last_tstamp = event.t_end
        self.location = event.location
        self.age += event.t_dur
        self.Engines.age += event.t_dur

        if event.type == 'Flight':
            self.fc_counter += 1
            self.fh_counter += event.t_dur
            self.Engines.fc_counter += 1
            self.Engines.fh_counter += event.t_dur
            self.Engines.deteriorate(event)

    def __repr__(self):
        """
        Representation of the aircraft's state.
        """
        return f"Aircraft in [{self.location}] at {self.last_tstamp.strftime('%Y-%m-%d %H:%M')}"


class Engines:
    def __init__(self, uid, config, aircraft=None):
        """
        Initialize an Engines instance, optionally attached to an aircraft.
        """
        self.uid = uid
        self.config = config
        self.event_calendar = []
        self.current_state = None

        # Engine thresholds and counters
        self.egtm_init = config.getfloat('Engine', 'egtm_init')
        self.llp_life_init = config.getfloat('Engine', 'llp_life_init')
        self.egt_resets = 0
        self.llp_resets = 0
        self.egti_per_fc = config.getfloat('Engine', 'egti') / 1000
        self.egtm_due = False
        self.llp_due = False
        self.random_due = False
        self.esv_until = None
        self.history = {'EGTM': [], 'LLP': [], 'TIME': [], 'EFCs': []}
        self.failure_efcs = []

        # Failure scheduling
        mtbf = config.getfloat('Engine', 'mtbf_efc')
        efc_counter = 0

        # Weibull distribution parameters
        beta = 3  # Shape parameter (>1 for increasing failure rate, <1 for decreasing)
        scale = mtbf / math.gamma(1 + 1 / beta)  # Adjust scale to match MTBF

        while efc_counter < 60000:  # Assumes simulation doesn't exceed this
            failure_efc = np.random.weibull(beta) * scale
            self.failure_efcs.append(failure_efc + efc_counter)
            efc_counter += failure_efc

        if aircraft is None:
            # Initialize engine attributes if not attached to an aircraft
            self.age = dt.timedelta(days=0)
            self.initial_age = self.age
            self.fc_counter = 0
            self.fh_counter = dt.timedelta(hours=0)
            self.egtm = self.egtm_init
            self.llp_life = self.llp_life_init
        else:
            # Attach engine attributes to the aircraft's state
            self.age = aircraft.age
            self.initial_age = self.age
            self.fc_counter = aircraft.fc_counter
            self.fh_counter = aircraft.fh_counter
            self.egtm = self.egtm_init - self.fc_counter * self.egti_per_fc
            self.llp_life = self.llp_life_init - self.fc_counter

            while self.egtm < 7.5:
                self.egtm += self.egtm_init
                self.egt_resets += 1

            while self.llp_life < 1750:
                self.llp_life += self.llp_life_init
                self.llp_resets += 1

            while self.failure_efcs[0] < self.fc_counter:
                del self.failure_efcs[0]

    def deteriorate(self, flight):
        """
        Simulate engine deterioration during a flight.
        """
        self.egtm -= self.egt_increase()
        self.llp_life -= 1
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(flight.t_end)
        self.history['EFCs'].append(self.fc_counter)

    def egt_increase(self):
        """
        Calculate the incremental deterioration in EGTM based on engine maturity.
        """
        efc_rate = 3 if self.fc_counter > 1000 else 8
        efc_rate_with_noise = efc_rate * np.random.uniform(-1, 3)
        return efc_rate_with_noise / 1000

    def maintenance_due(self):
        """
        Determine if maintenance is required based on EGTM, LLP, or random failure thresholds.
        """
        self.critical_egtm_due = self.egtm < 5
        self.warning_egtm_due = 5 <= self.egtm < 10
        self.critical_llp_due = self.llp_life < 500
        self.warning_llp_due = 500 <= self.llp_life < 3000
        self.random_due = self.fc_counter > self.failure_efcs[0] if self.failure_efcs else False

        if self.critical_egtm_due or self.critical_llp_due or self.random_due:
            return True
        else:
            return False

    def restore(self, globalClock):
        """
        Perform restorations based on critical thresholds.
        Warning-level restorations are handled automatically when critical restorations occur.
        """

        print("\nESV for Engine %d on %s at %d EFCs"
              % (self.uid, globalClock.strftime("%d.%m.%Y %H:%M"), self.fc_counter))

        # Handle random failure repair
        if self.random_due:
            self.random_restoration(globalClock)
            if self.warning_egtm_due:
                self.egtm_restoration(globalClock)
            if self.warning_llp_due:
                self.llp_restoration(globalClock)

        # Handle critical EGTM restoration
        if self.critical_egtm_due:
            self.egtm_restoration(globalClock)
            if self.warning_llp_due:
                self.llp_restoration(globalClock)

        # Handle critical LLP restoration
        if self.critical_llp_due:
            self.llp_restoration(globalClock)
            if self.warning_egtm_due:
                self.egtm_restoration(globalClock)

        # Record history entry for all state changes
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(globalClock)
        self.history['EFCs'].append(self.fc_counter)

    def egtm_restoration(self, globalClock):
        """
        Restore EGTM to initial state and mark engine as unavailable for 25 days.
        """

        old_egtm = self.egtm
        new_egtm = self.egtm_init - np.random.uniform(0, 5)

        self.egt_resets += 1
        self.egtm = new_egtm
        self.egtm_due = False
        self.esv_until = globalClock + dt.timedelta(days=25)
        self.critical_egtm_due = False
        self.warning_egtm_due = False

        print("\t - EGTM restoration from %.1f°C to %.1f°C"
              % (old_egtm, new_egtm))

    def llp_restoration(self, globalClock):
        """
        Restore LLP life to initial state and mark engine as unavailable for 25 days.
        """

        old_llp_life = self.llp_life
        new_llp_life = self.llp_life_init

        self.llp_resets += 1
        self.llp_life = new_llp_life
        self.llp_due = False
        self.esv_until = globalClock + dt.timedelta(days=25)
        self.critical_llp_due = False
        self.warning_llp_due = False

        print("\t - LLP-RUL restoration from %.1fEFC to %.1fEFC"
              % (old_llp_life, new_llp_life))

    def random_restoration(self, globalClock):
        self.random_due = False
        self.esv_until = globalClock + dt.timedelta(days=7)
        del self.failure_efcs[0]

        print("\t - replacement of failed part")
