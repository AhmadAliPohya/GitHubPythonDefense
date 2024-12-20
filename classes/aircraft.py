import numpy as np
from scipy.stats import truncnorm
import datetime as dt


class Aircraft:
    def __init__(self, uid, config):
        self.uid = uid
        self.config = config

        # Initialize attributes
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
        """Generate a random age for the aircraft as a timedelta."""
        min_age = self.config.getint('Aircraft', 'min_age')
        max_age = self.config.getint('Aircraft', 'max_age')
        random_age_in_years = np.random.uniform(min_age, max_age)
        random_age_in_days = random_age_in_years * 365  # Approximate days per year
        return dt.timedelta(days=random_age_in_days)

    def _generate_fc_counter(self):
        """Generate the flight cycle counter based on a truncated normal distribution."""
        avg_fc_per_year = self.config.getfloat('Aircraft', 'avg_fc_per_year')
        std_fc_per_year = self.config.getfloat('Aircraft', 'std_fc_per_year')
        lower_bound = max(avg_fc_per_year - 2 * std_fc_per_year, 12)
        upper_bound = avg_fc_per_year + 2 * std_fc_per_year

        fc_per_year = self._sample_truncated_normal(avg_fc_per_year, std_fc_per_year, lower_bound, upper_bound)
        age_years = self.age.total_seconds() / (60 * 60 * 24 * 365)  # Convert age to years
        return int(fc_per_year * age_years)

    def _generate_fh_counter(self):
        """Generate the flight hour counter based on a truncated normal distribution."""
        avg_fh_per_fc = self.config.getfloat('Aircraft', 'avg_fh_per_fc')
        std_fh_per_fc = self.config.getfloat('Aircraft', 'std_fh_per_fc')
        lower_bound = max(avg_fh_per_fc - 2 * std_fh_per_fc, 0.5)
        upper_bound = avg_fh_per_fc + 2 * std_fh_per_fc

        fh_per_fc = self._sample_truncated_normal(avg_fh_per_fc, std_fh_per_fc, lower_bound, upper_bound)
        fh_float = fh_per_fc * self.fc_counter
        return dt.timedelta(hours=fh_float)

    @staticmethod
    def _sample_truncated_normal(mean, std, lower_bound, upper_bound):
        """Sample a value from a truncated normal distribution."""
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)

    def attach_engines(self, engine_set):
        self.Engines = engine_set

    def add_event(self, event):
        """Add an event to the aircraft's event calendar."""
        self.event_calendar.append(event)
        self.last_tstamp = event.t_end
        self.location = event.location
        self.age += event.t_dur
        self.Engines.age += event.t_dur

        if event.type == 'Flight':
            self.fc_counter += 1
            self.fh_counter += + event.t_dur
            self.Engines.fc_counter += 1
            self.Engines.fh_counter += event.t_dur
            self.Engines.deteriorate(event)


    def __repr__(self):
        return ("Aircraft in [%s] at %s"
              % (self.location, self.last_tstamp.strftime('%Y-%m-%d %H:%M')))


class Engines:
    def __init__(self, uid, config, aircraft=None):

        self.uid = uid
        self.config = config
        self.event_calendar = []
        self.current_state = None
        self.egtm_init = config.getfloat('Engine', 'egtm_init')
        self.llp_life_init = config.getfloat('Engine', 'llp_life_init')
        self.egt_resets = 0
        self.llp_resets = 0
        self.egti_per_fc = config.getfloat('Engine', 'egti') / 1000
        self.egtm_due = False
        self.llp_due = False
        self.ready = True
        self.esv_until = None
        self.history = {'EGTM': [], 'LLP': [], 'TIME': []}


        if aircraft is None:
            self.age = dt.timedelta(days=0)
            self.initial_age = self.age
            self.fc_counter = 0
            self.fh_counter = dt.timedelta(hours=0)
            self.egtm = self.egtm_init
            self.llp_life = self.llp_life_init
        else:

            # Initialize attributes
            self.age = aircraft.age
            self.initial_age = self.age
            self.fc_counter = aircraft.fc_counter
            self.fh_counter = aircraft.fh_counter

            self.egtm = (self.egtm_init
                         - self.fc_counter * self.config.getfloat('Engine', 'egti') / 1000)

            while self.egtm < 7.5:
                self.egtm += self.egtm_init
                self.egt_resets += 1

            self.llp_life = self.llp_life_init - self.fc_counter

            while self.llp_life < 1750:
                self.llp_life += self.llp_life_init
                self.llp_resets += 1

    def deteriorate(self, flight):
        self.egtm -= self.egt_increase()
        self.llp_life -= 1

        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(flight.t_end)


    def egt_increase(self):

        # Determine maturity-based rate
        efc_rate = 3 if self.fc_counter > 1000 else 8

        efc_rate_withnoise = efc_rate * np.random.uniform(-1, 3)

        egt_increase = efc_rate_withnoise / 1000

        return egt_increase



    def maintenance_due(self):

        egtm_due = False
        llp_due = False

        if self.egtm < 5:
            egtm_due = True
            if self.llp_life < 3000:
                llp_due = True
        elif self.llp_life < 500:
            llp_due = True
            if self.egtm < 10:
                egtm_due = True

        self.egtm_due = egtm_due
        self.llp_due = llp_due

    def restore(self, globalClock):
        if self.egtm_due and self.llp_due:
            self.complete_restoration(globalClock)
        elif self.egtm_due:
            self.egtm_restoration(globalClock)
        elif self.llp_due:
            self.llp_restoration(globalClock)

        # self.ready = False

    def egtm_restoration(self, globalClock):

        self.egt_resets += 1
        self.egtm = self.egtm_init - np.random.uniform(0, 5)
        self.egtm_due = False
        self.esv_until = globalClock + dt.timedelta(days=25)
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(globalClock)


    def llp_restoration(self, globalClock):

        self.llp_resets += 1
        self.llp_life = self.llp_life_init
        self.llp_due = False
        self.esv_until = globalClock + dt.timedelta(days=25)
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(globalClock)

    def complete_restoration(self, globalClock):

        self.egtm_restoration(globalClock)
        self.llp_restoration(globalClock)