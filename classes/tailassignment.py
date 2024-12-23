import pandas as pd
import numpy as np
import datetime as dt
from classes.events import FlightEvent, TurnaroundEvent

data = pd.read_csv('data\\data.csv')
stats = {'avg_dist': np.average(data['DIST'], weights=data['FREQ']),
         'avg_fh': np.average(data['TIME'], weights=data['FREQ']),
         'avg_fc/a': 6011.2 / np.average(data['TIME'], weights=data['FREQ']),
         }
stats['avg_fc/week'] = stats['avg_fc/a'] / 52

# Create a dictionary to store flight events
flightevents = {}

# Create a nested dictionary to store origin-destination pairs
od_pairs = {}

# Populate the dictionaries
for _, row in data.iterrows():
    # Store the parameters needed for a FlightEvent as a dictionary
    FLIGHT_PARAMS = {
        "name": row["ROUTE"],
        "orig": row["ORIG"],
        "dest": row["DEST"],
        "dist": row["DIST"],
        "t_dur": dt.timedelta(hours=row["TIME"])
    }

    # Store these parameters instead of a FlightEvent object
    flightevents[FLIGHT_PARAMS["name"]] = FLIGHT_PARAMS

    orig = row["ORIG"]
    dest = row["DEST"]
    freq = row["FREQ"]

    if orig not in od_pairs:
        od_pairs[orig] = {}

    od_pairs[orig][dest] = {
        "p": freq,
        "event_params": FLIGHT_PARAMS
    }

def initial(mng):

    sim_time_begin_str = mng.config.get('Simulation', 'initial_time')
    sim_time_begin = dt.datetime.strptime(sim_time_begin_str, '%Y-%m-%d %H:%M:%S')

    assigned_airports = list(np.random.choice(data["ORIG"],
                                              size=len(mng.aircraft_fleet),
                                              p=data["FREQ"]))

    for idx, aircraft in enumerate(mng.aircraft_fleet):
        aircraft.location = assigned_airports[idx]
        aircraft.next_flights = []
        orig = aircraft.location


        n_future_flights = 2 * int(stats['avg_fc/week'])
        for idx, _ in enumerate(range(n_future_flights)):
            dests, probs = zip(*[(dest, info["p"]) for dest, info in od_pairs[orig].items()])
            normalized_probs = [p / sum(probs) for p in probs]
            selected_dest = np.random.choice(dests, p=normalized_probs)

            # Get the parameters dictionary for the next flight
            next_flight_params = od_pairs[orig][selected_dest]["event_params"]

            aircraft.next_flights.append(next_flight_params)

            orig = selected_dest

        # Append first flight
        flight_params = aircraft.next_flights[0]
        flight_params['t_beg'] = sim_time_begin
        flightevent = FlightEvent(**flight_params)

        aircraft.add_event(flightevent)

        # Calculate TAT stats
        total_ftime = np.sum([el['t_dur'] for el in aircraft.next_flights])
        total_ftime_hrs = total_ftime.total_seconds() / 3600
        gap_2weeks = 14 * 24 - total_ftime_hrs
        avg_tat_hrs = gap_2weeks / len(aircraft.next_flights)
        aircraft.avg_tat_hrs_min = avg_tat_hrs - 0.25 * avg_tat_hrs
        aircraft.avg_tat_hrs_avg = avg_tat_hrs
        aircraft.avg_tat_hrs_max = avg_tat_hrs + 0.25 * avg_tat_hrs

    return mng


def reschedule(mng):

    for idx, aircraft in enumerate(mng.aircraft_fleet):
        orig = aircraft.location
        n_future_flights = 2*int(stats['avg_fc/week'])
        aircraft.next_flights = []
        for _ in range(n_future_flights):
            dests, probs = zip(*[(dest, info["p"]) for dest, info in od_pairs[orig].items()])
            normalized_probs = [p / sum(probs) for p in probs]
            selected_dest = np.random.choice(dests, p=normalized_probs)

            # Get the parameters dictionary for the next flight
            next_flight_params = od_pairs[orig][selected_dest]["event_params"]

            aircraft.next_flights.append(next_flight_params)
            orig = selected_dest

    return mng
