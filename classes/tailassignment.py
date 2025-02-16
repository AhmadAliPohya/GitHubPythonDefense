import logging
import pandas as pd
import numpy as np
import datetime as dt
from classes.events import FlightEvent
from collections import Counter


class AssignmentManager:
    def __init__(self):

        # Read flight data from a CSV file into a pandas DataFrame
        self.ferry_flights = []
        data = pd.read_csv('data\\flight_schedule.csv')


        data['FREQ'] = data['FREQ']/np.sum(data['FREQ'])
        self.data = data

        # Calculate various statistics using weighted averages based on 'FREQ' column
        self.stats = {
            'avg_dist': np.average(data['DIST'], weights=data['FREQ']),
            'avg_fh': np.average(data['TIME'], weights=data['FREQ']),
            #'avg_fc/a': 6011.2 / np.average(data['TIME'], weights=data['FREQ']),
            'avg_fc/a': 315
        }
        # Average flight cycles per week
        self.stats['avg_fc/week'] = self.stats['avg_fc/a'] / 52


        self.flights2assign = None
        self.fevent_params = {}
        self.fevent_info = {'name': [], 'p':[]}

        self.compatible_flights = {}

        for _, row in data.iterrows():

            # Create a dictionary of flight parameters for each route
            FLIGHT_PARAMS = {
                "name": row["ROUTE"],  # Route name (e.g., "BOS_SFO")
                "orig": row["ORIG"],  # Origin airport
                "dest": row["DEST"],  # Destination airport
                "dist": row["DIST"],  # Distance of the flight
                "t_dur": dt.timedelta(hours=row["TIME"]),  # Duration of the flight as a timedelta object
            }

            self.fevent_params[FLIGHT_PARAMS["name"]] = FLIGHT_PARAMS
            self.fevent_info['name'].append(FLIGHT_PARAMS["name"])
            self.fevent_info['p'].append(row['FREQ'])

            if row['ORIG'] not in self.compatible_flights:
                self.compatible_flights[row['ORIG']] = []

            self.compatible_flights[row['ORIG']].append(row['ROUTE'])

    def create_flights2assign(self, mng):

        min_occurrence = 1
        min_prob = min(self.fevent_info['p'])

        # Compute scaling factor
        scaling_factor = min_occurrence/min_prob

        # compute occurrences
        occurrences = [max(round(p * scaling_factor), 1) for p in self.fevent_info['p']]

        # Generate flight list
        flightlist = [s for s, count in zip(self.fevent_info['name'], occurrences) for _ in range(count)]

        self.flights2assign = Counter(flightlist)

        return Counter(flightlist)

        # # Calculate the total number of flights to assign based on fleet size and average flight cycles per week
        # num_flights2assign = int(len(mng.aircraft_active) * 8 * self.stats['avg_fc/week'])
        #
        # flights2assign_npstr = list(
        #     np.random.choice(
        #         list(self.fevent_params.keys()),
        #         size=num_flights2assign,
        #         p=self.fevent_info['p']
        #     )
        # )
        # flights2assign_str = [str(el) for el in flights2assign_npstr]
        # flights2assign_all = Counter(flights2assign_str)
        #
        # # Adjust each count to be even
        # self.flights2assign = Counter({
        #     flight: count + (count % 2) for flight, count in flights2assign_all.items()
        # })
        # return self.flights2assign

    def select_flight2assign(self, flights2assign, aircraft_locations=None):

        compatible_flights = []
        for location in aircraft_locations:
            compatible_flights.extend(self.compatible_flights[location])


        for route in np.random.permutation(compatible_flights):
            if flights2assign[route] > 0:
                return route

        route = np.random.choice(compatible_flights)
        self.ferry_flights.append(route)

        return route

        #return np.random.choice(self.fevent_info['name'], p=self.fevent_info['p'])


def initial(mng):

    # Randomly assign initial airports to each aircraft based on origin frequencies
    assigned_airports = list(
        np.random.choice(
            asmng.data["ORIG"],
            size=len(mng.aircraft_active),
            p=asmng.data["FREQ"]
        )
    )

    # Now assign the locations and some other things
    for idx, aircraft in enumerate(mng.aircraft_active):
        aircraft.location = assigned_airports[idx]
        aircraft.goal_fh_fc_ratio = asmng.stats['avg_fh']
        aircraft.next_flights.append({'dest': assigned_airports[idx],
                                      't_dur': dt.timedelta(hours=0)})

    # Create the flights 2 assign list
    flights2assign = asmng.create_flights2assign(mng)
    # num_flights2assign = sum(flights2assign.values())
    # remaining_flights = num_flights2assign

    earliest_clock = mng.SimTime
    assign_until_clock = earliest_clock + dt.timedelta(days=60)

    avg_tat_hrs = mng.config.getfloat('Aircraft', 'avg_tat_hrs')
    avg_tat_td = dt.timedelta(hours=avg_tat_hrs)

    while earliest_clock < assign_until_clock:
    #while remaining_flights > 0:

        aircraft_locations = []
        for aircraft in mng.aircraft_active:
            aircraft_locations.append(aircraft.next_flights[-1]['dest'])

        try:
            selected_flight_str = asmng.select_flight2assign(flights2assign, aircraft_locations)
            selected_flight = asmng.fevent_params[selected_flight_str]
        except KeyError:
            print("Break")


        availables_list = [
            (ac, ac.scheduled_until)
            for ac in mng.aircraft_active
            if ac.next_flights[-1]['dest'] == selected_flight['orig']]

        if availables_list:

            # Select the tuple with the smallest total_duration
            aircraft, _ = min(availables_list, key=lambda x: x[1])

            aircraft.add_next_flight(selected_flight)

            flights2assign[selected_flight_str] -= 1
            #remaining_flights -= 1

        earliest_clock_list = []

        for aircraft in mng.aircraft_active:
            assigned_until = mng.SimTime + np.sum([flight['t_dur'] + avg_tat_td for flight in aircraft.next_flights])
            earliest_clock_list.append(assigned_until)

            #earliest_clock = min([earliest_clock, assigned_until])

        earliest_clock = min(earliest_clock_list)
        #print(earliest_clock)

    for idx, aircraft in enumerate(mng.aircraft_active):

        del aircraft.next_flights[0]
        first_flight_params = aircraft.next_flights.pop(0)
        first_flight_event = FlightEvent(**first_flight_params,
                                         t_beg=mng.SimTime)

        aircraft.add_event(first_flight_event)

        # Assign minimum, average, and maximum TAT based on the calculated average
        aircraft.avg_tat_hrs_min = avg_tat_hrs - 0.25 * avg_tat_hrs  # 75% of average TAT
        aircraft.avg_tat_hrs_avg = avg_tat_hrs  # Average TAT
        aircraft.avg_tat_hrs_max = avg_tat_hrs + 0.25 * avg_tat_hrs  # 125% of average TAT

    return mng


def reschedule(mng):

    logging.debug("(%s) | Rescheduling Flights to Aircraft"
                 % mng.SimTime.strftime("%Y-%m-%d %H:%M:%S"))

    # Create the flights 2 assign list
    flights2assign = asmng.create_flights2assign(mng)

    earliest_clock = mng.SimTime
    assign_until_clock = earliest_clock + dt.timedelta(days=60)

    avg_tat_hrs = mng.config.getfloat('Aircraft', 'avg_tat_hrs')
    avg_tat_td = dt.timedelta(hours=avg_tat_hrs)

    for aircraft in mng.aircraft_active:
        # Ditch the excess flight list
        aircraft.next_flights = [aircraft.next_flights[0]]
        #aircraft.scheduled_until = aircraft.next_flights[-1]['t_dur']

    while earliest_clock < assign_until_clock:

        earliest_clock_list = []

        for aircraft in mng.aircraft_active:
            location = aircraft.next_flights[-1]['dest']
            compatible_flightnames_all = asmng.compatible_flights[location]

            compatible_flightnames_filtered = []
            t_durs = []
            devs = []

            for flight in compatible_flightnames_all:
                if asmng.flights2assign[flight] > 0:
                    compatible_flightnames_filtered.append(flight)
                    t_durs.append(asmng.fevent_params[flight]['t_dur'].total_seconds() / 3600)
                    devs.append(np.abs(aircraft.goal_fh_fc_ratio - t_durs[-1]))

            if len(compatible_flightnames_filtered) == 0:
                selected_flight_str = np.random.choice(compatible_flightnames_all)
            else:

                if np.random.random() >= mng.tap_threshold:
                    # assign flights so that the deviation is smallest
                    min_dev_idx = np.argmin(devs)
                    selected_flight_str = compatible_flightnames_filtered[min_dev_idx]
                else:
                    # just randomly assign a flight
                    selected_flight_str = np.random.choice(compatible_flightnames_filtered)

            selected_flight = asmng.fevent_params[selected_flight_str]
            aircraft.add_next_flight(selected_flight)
            flights2assign[selected_flight_str] -= 1

            assigned_until = mng.SimTime + np.sum([flight['t_dur'] + avg_tat_td for flight in aircraft.next_flights])
            earliest_clock_list.append(assigned_until)


        earliest_clock = min(earliest_clock_list)
    #
    #
    #
    #
    #     aircraft_locations = []
    #     for aircraft in mng.aircraft_active:
    #         aircraft_locations.append(aircraft.next_flights[-1]['dest'])
    #
    #     selected_flight_str = asmng.select_flight2assign(flights2assign, aircraft_locations)
    #     selected_flight = asmng.fevent_params[selected_flight_str]
    #
    #     availables_list = [
    #         (ac, ac.scheduled_until)
    #         for ac in mng.aircraft_active
    #         if ac.next_flights[-1]['dest'] == selected_flight['orig']]
    #
    #     if availables_list:
    #
    #         # Select the tuple with the smallest total_duration
    #         aircraft, _ = min(availables_list, key=lambda x: x[1])
    #
    #         aircraft.add_next_flight(selected_flight)
    #
    #         flights2assign[selected_flight_str] -= 1
    #         #remaining_flights -= 1
    #
    #     earliest_clock_list = []
    #
    #     for aircraft in mng.aircraft_active:
    #         assigned_until = mng.SimTime + np.sum([flight['t_dur'] + avg_tat_td for flight in aircraft.next_flights])
    #         earliest_clock_list.append(assigned_until)
    #
    #     earliest_clock = min(earliest_clock_list)
    #
    # while remaining_flights > 0:
    #
    #     for aircraft in mng.aircraft_active:
    #         location = aircraft.next_flights[-1]['dest']
    #         compatible_flightnames_all = asmng.compatible_flights[location]
    #
    #         compatible_flightnames_filtered = []
    #         t_durs = []
    #         devs = []
    #
    #         for flight in compatible_flightnames_all:
    #             if asmng.flights2assign[flight] > 0:
    #                 compatible_flightnames_filtered.append(flight)
    #                 t_durs.append(asmng.fevent_params[flight]['t_dur'].total_seconds() / 3600)
    #                 devs.append(np.abs(aircraft.goal_fh_fc_ratio - t_durs[-1]))
    #
    #         if len(compatible_flightnames_filtered) == 0:
    #             selected_flight_str = np.random.choice(compatible_flightnames_all)
    #         else:
    #
    #             if np.random.random() >= mng.tap_threshold:
    #                 # assign flights so that the deviation is smallest
    #                 min_dev_idx = np.argmin(devs)
    #                 selected_flight_str = compatible_flightnames_filtered[min_dev_idx]
    #             else:
    #                 # just randomly assign a flight
    #                 selected_flight_str = np.random.choice(compatible_flightnames_filtered)
    #
    #         selected_flight = asmng.fevent_params[selected_flight_str]
    #         aircraft.add_next_flight(selected_flight)
    #         flights2assign[selected_flight_str] -= 1
    #         remaining_flights -= 1

    mng.ferry_flights = asmng.ferry_flights
    return mng


asmng = AssignmentManager()
