import logging
import numpy as np
import datetime as dt

class Engines:


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
        self.fc_since_esv = 0
        self.next_scheduled_esv = None
        self.egtm_lost = 0
        self.llp_rul_lost = 0
        self.ages_dt = dict()
        self.aircraft = None
        self.warning_egtm_due = False
        self.critical_egtm_due = False
        self.warning_llp_due = False
        self.critical_llp_due = False
        self.uid = uid
        self.config = config
        self.event_calendar = []
        self.current_state = None

        self.next_esv = {'UID': self.uid,
                         'TP_MIN': None,
                         'TP_EST': None,
                         'TP_MAX': None,
                         'T_END': None,
                         'OBJ': self,
                         'DRIVER': None}


        # Read engine thresholds and counters from config
        self.egtm_init = config.getfloat('Engine', 'egtm_init')
        self.llp_life_init = config.getfloat('Engine', 'llp_life_init')
        self.egt_resets = 0
        self.llp_resets = 0

        # EGTM increment per flight cycle (scaled down by 1/1000)
        self.egti_per_fc = config.getfloat('Engine', 'egti') / 1000

        # Booleans that track whether the engine is due for certain maintenance
        self.egtm_due = False
        self.llp_due = False
        # self.random_due = False

        # Tracks until which date the engine is out of service
        self.esv_until = None

        # Dictionary to keep history of engine parameters over time
        self.history = {'EGTM': [], 'LLP': [], # 'SOH': [],
                        'TIME': [], 'EFCs': [], 'EFHs': []}

        self.age = dt.timedelta(days=0)
        self.initial_age = self.age
        self.fc_counter = 0
        self.fh_counter = dt.timedelta(hours=0)
        self.egtm = self.egtm_init
        self.llp_life = self.llp_life_init


    def detach_aircraft(self, current_time):

        self.ages_dt['off aircraft'] = current_time
        self.aircraft = None

    def attach_aircraft(self, aircraft, current_time):

        self.aircraft = aircraft
        self.ages_dt['on aircraft'] = current_time
        if 'off aircraft' in self.ages_dt.keys():
            self.age += current_time - self.ages_dt['off aircraft']
            # self.soh_update()
            # if self.maintenance_due() and self.warning_soh_due:
            #     self.random_restoration()

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
        # self.random_soh += self.soh_update()

        # Record the updated engine state to the history dictionary
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        # self.history['SOH'].append(self.random_soh)
        self.history['TIME'].append(flight.t_end)
        self.history['EFCs'].append(self.fc_counter)
        self.history['EFHs'].append(self.fh_counter)


    def egt_increase(self):
        """
        Calculate the incremental deterioration in EGTM based on engine maturity.

        Returns
        -------
        float
            A random increment (in degrees Celsius) to reduce the EGTM value by.
        """

        #efc_rate = 3 if self.fc_counter > 1000 else 8
        efc_rate = 5 if self.fc_counter > 800 else 12
        efc_rate_with_noise = efc_rate + 0 * np.random.uniform(-1, 1)
        #efc_rate_with_noise = efc_rate + 20 * np.random.uniform(-1, 1)
        return efc_rate_with_noise / 1000

    def maintenance_due(self, asmng):
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
        self.warning_egtm_due = self.egtm < 10

        # Critical threshold for LLP
        self.critical_llp_due = self.llp_life < 1000
        # Warning threshold for LLP
        self.warning_llp_due = self.llp_life < 2000

        if self.next_scheduled_esv is not None and self.next_scheduled_esv <= asmng.SimTime:
            return True

        # If any critical or random due condition is met, return True
        if self.critical_egtm_due or self.critical_llp_due: # or self.critical_soh_due:
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

        egtm_restored = False
        llps_restored = False
        # Handle critical EGTM restoration
        if self.critical_egtm_due:
            self.egtm_lost += self.egtm
            self.egtm_restoration(SimTime)
            egtm_restored = True
            # Also handle warning-level LLP if present
            if self.warning_llp_due:
                self.llp_rul_lost += self.llp_life
                self.llp_restoration(SimTime)
                llps_restored = True

        # Handle critical LLP restoration
        if self.critical_llp_due:
            self.llp_rul_lost += self.llp_life
            self.llp_restoration(SimTime)
            llps_restored = True
            # Also handle warning-level EGTM if present
            if self.warning_egtm_due:
                self.egtm_lost += self.egtm
                self.egtm_restoration(SimTime)
                egtm_restored = True

        if egtm_restored == False and llps_restored == False:
            self.egtm_lost += self.egtm
            self.egtm_restoration(SimTime)
            # Also handle warning-level LLP if present
            if self.warning_llp_due:
                self.llp_rul_lost += self.llp_life
                self.llp_restoration(SimTime)

        # Record the updated engine state after restoration
        self.history['EGTM'].append(self.egtm)
        self.history['LLP'].append(self.llp_life)
        self.history['TIME'].append(SimTime)
        self.history['EFCs'].append(self.fc_counter)
        self.next_scheduled_esv = None
        self.fc_since_esv = 0

        self.next_esv = {'UID': self.uid,
                         'TP_MIN': None,
                         'TP_EST': None,
                         'TP_MAX': None,
                         'T_END': None,
                         'OBJ': self,
                         'DRIVER': None}

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


    def set_fh_fc_ratio(self, new_fh_fc_ratio):

        self.goal_fh_fc_ratio = new_fh_fc_ratio
        self.aircraft.goal_fh_fc_ratio = new_fh_fc_ratio
