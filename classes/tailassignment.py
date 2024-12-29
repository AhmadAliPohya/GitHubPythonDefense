import pandas as pd
import numpy as np
import datetime as dt
from classes.events import FlightEvent
from collections import Counter


class AssignmentManager():
    def __init__(self):

        # Read flight data from a CSV file into a pandas DataFrame
        data = pd.read_csv('data\\data.csv')
        self.data = data

        # Calculate various statistics using weighted averages based on 'FREQ' column
        self.stats = {
            'avg_dist': np.average(data['DIST'], weights=data['FREQ']),
            'avg_fh': np.average(data['TIME'], weights=data['FREQ']),
            'avg_fc/a': 6011.2 / np.average(data['TIME'], weights=data['FREQ']),
        }
        # Average flight cycles per week
        self.stats['avg_fc/week'] = self.stats['avg_fc/a'] / 52

        self.flights2assign = None
        self.fevent_params = {}
        self.fevent_info = {'name': [], 'p':[]}

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

    def create_flights2assign(self, mng):

        # Calculate the total number of flights to assign based on fleet size and average flight cycles per week
        num_flights2assign = int(len(mng.aircraft_fleet) * 6 * self.stats['avg_fc/week'])

        flights2assign_npstr = list(
            np.random.choice(
                list(self.fevent_params.keys()),
                size=num_flights2assign,
                p=self.fevent_info['p']
            )
        )
        flights2assign_str = [str(el) for el in flights2assign_npstr]
        flights2assign_all = Counter(flights2assign_str)

        # Adjust each count to be even
        self.flights2assign = Counter({
            flight: count + (count % 2) for flight, count in flights2assign_all.items()
        })
        return self.flights2assign

    def select_flight2assign(self):

        return np.random.choice(self.fevent_info['name'], p=self.fevent_info['p'])

asmng = AssignmentManager()


def initial(mng):

    # Randomly assign initial airports to each aircraft based on origin frequencies
    assigned_airports = list(
        np.random.choice(
            asmng.data["ORIG"],
            size=len(mng.aircraft_fleet),
            p=asmng.data["FREQ"]
        )
    )

    for idx, aircraft in enumerate(mng.aircraft_fleet):
        aircraft.location = assigned_airports[idx]
        aircraft.next_flights.append({'dest': assigned_airports[idx],
                                      't_dur': dt.timedelta(hours=0)})

    flights2assign = asmng.create_flights2assign(mng)
    num_flights2assign = sum(flights2assign.values())
    remaining_flights = num_flights2assign

    while remaining_flights > 0:

        selected_flight_str = asmng.select_flight2assign()
        selected_flight = asmng.fevent_params[selected_flight_str]

        availables_list = [
            (ac, ac.scheduled_until)
            for ac in mng.aircraft_fleet
            if ac.next_flights[-1]['dest'] == selected_flight['orig']]

        if availables_list:

            # Select the tuple with the smallest total_duration
            aircraft, _ = min(availables_list, key=lambda x: x[1])

            aircraft.add_next_flight(selected_flight)

            flights2assign[selected_flight_str] -= 1
            remaining_flights -= 1

    for idx, aircraft in enumerate(mng.aircraft_fleet):

        del aircraft.next_flights[0]
        first_flight_params = aircraft.next_flights.pop(0)
        first_flight_event = FlightEvent(**first_flight_params,
                                         t_beg=mng.SimTime)

        aircraft.add_event(first_flight_event)

        # Calculate Total Flight Time (TAT) statistics
        total_ftime = np.sum(
            [el['t_dur'] for el in aircraft.next_flights])  # Sum of flight durations
        total_ftime_hrs = total_ftime.total_seconds() / 3600  # Convert to hours

        # Calculate the remaining time gap over 6 weeks (in hours)
        gap_6weeks = 6 * 7 * 24 - total_ftime_hrs

        # Calculate average Turnaround Time (TAT) in hours
        avg_tat_hrs = gap_6weeks / len(aircraft.next_flights)

        # Assign minimum, average, and maximum TAT based on the calculated average
        aircraft.avg_tat_hrs_min = avg_tat_hrs - 0.25 * avg_tat_hrs  # 75% of average TAT
        aircraft.avg_tat_hrs_avg = avg_tat_hrs  # Average TAT
        aircraft.avg_tat_hrs_max = avg_tat_hrs + 0.25 * avg_tat_hrs  # 125% of average TAT

    return mng


def reschedule(mng):


    flights2assign = asmng.create_flights2assign(mng)
    num_flights2assign = sum(flights2assign.values())
    remaining_flights = num_flights2assign

    while remaining_flights > 0:

        selected_flight_str = asmng.select_flight2assign()
        selected_flight = asmng.fevent_params[selected_flight_str]

        availables_list = [
            (ac, ac.scheduled_until)
            for ac in mng.aircraft_fleet
            if ac.next_flights[-1]['dest'] == selected_flight['orig']]

        if availables_list:

            if mng.predictive:
                pass
            else:

                # Select the tuple with the smallest total_duration
                aircraft, _ = min(availables_list, key=lambda x: x[1])

                aircraft.add_next_flight(selected_flight)

                flights2assign[selected_flight_str] -= 1
                remaining_flights -= 1


    return mng
