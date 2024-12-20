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

def initial(aircraft_fleet, config):

    sim_time_begin_str = config.get('Simulation', 'initial_time')
    sim_time_begin = dt.datetime.strptime(sim_time_begin_str, '%Y-%m-%d %H:%M:%S')

    assigned_airports = list(np.random.choice(data["ORIG"],
                                              size=len(aircraft_fleet),
                                              p=data["FREQ"]))

    for idx, aircraft in enumerate(aircraft_fleet):
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

    return aircraft_fleet


"""
def initialOld(aircraft_fleet, config):
    assigned_airports = list(np.random.choice(data["ORIG"],
                                              size=len(aircraft_fleet),
                                              p=data["FREQ"]))

    # assign a location to each aircraft
    for idx, aircraft in enumerate(aircraft_fleet):
        aircraft.location = assigned_airports[idx]
        aircraft.next_flights = []
        orig = aircraft.location

        n_future_flights = 2*int(stats['avg_fc/week'])
        for _ in range(n_future_flights):
            dests, probs = zip(*[(dest, info["p"]) for dest, info in od_pairs[orig].items()])
            normalized_probs = [p / sum(probs) for p in probs]
            selected_dest = np.random.choice(dests, p=normalized_probs)

            # Get the parameters dictionary for the next flight
            next_flight_params = od_pairs[orig][selected_dest]["event_params"]

            ## Create the FlightEvent object when needed
            #next_flight = FlightEvent(**next_flight_params)

            aircraft.next_flights.append(next_flight_params)
            orig = selected_dest

    # now calculate turnaround times to match utilization
    for idx, aircraft in enumerate(aircraft_fleet):

        total_ftime = np.sum([el['t_dur'] for el in aircraft.next_flights])
        total_ftime_hrs = total_ftime.total_seconds() / 3600
        gap_2weeks = 14 * 24 - total_ftime_hrs
        avg_tat_hrs = gap_2weeks / len(aircraft.next_flights)
        aircraft.avg_tat_hrs_min = avg_tat_hrs - 0.25 * avg_tat_hrs
        aircraft.avg_tat_hrs_avg = avg_tat_hrs
        aircraft.avg_tat_hrs_max = avg_tat_hrs + 0.25 * avg_tat_hrs

    sim_time_begin_str = config.get('Simulation', 'initial_time')
    sim_time_begin = dt.datetime.strptime(sim_time_begin_str, '%Y-%m-%d %H:%M:%S')

    for ac_idx, aircraft in enumerate(aircraft_fleet):
        first_run = True
        while True:
            flight_params = aircraft.next_flights.pop(0)
            if first_run:
                aircraft.last_tstamp = sim_time_begin - dt.timedelta(days=7)
                first_run = False
            flight_params['t_beg'] = aircraft.last_tstamp
            flightevent = FlightEvent(**flight_params)

            if flightevent.t_end > sim_time_begin:
                # If the flight ends later than the begin of the simulation,
                # then delete the turnaround so that the event list always
                # ends with a flight (important for later)
                del aircraft.event_calendar[-1]
                aircraft.last_tstamp = aircraft.event_calendar[-1].t_end
                break

            aircraft.event_calendar.append(flightevent)
            aircraft.last_tstamp = flightevent.t_end
            aircraft.location = flightevent.dest

            # Now add turnarounds
            tat_hrs = np.random.uniform(low=aircraft.avg_tat_hrs_min, high=aircraft.avg_tat_hrs_max)
            tat = TurnaroundEvent(
                location=flightevent.dest,
                t_beg=flightevent.t_end,
                t_dur=dt.timedelta(hours=tat_hrs))

            if tat.t_end > sim_time_begin:
                # if the tat ends later than the begin of the simulation,
                # then do not append the tat
                break

            aircraft.event_calendar.append(tat)
            aircraft.last_tstamp = tat.t_end

    return aircraft_fleet
"""

def reschedule(aircraft_fleet):

    for idx, aircraft in enumerate(aircraft_fleet):
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

    return aircraft_fleet
