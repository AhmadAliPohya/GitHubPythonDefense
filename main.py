# /main.py

import configparser as cp
import plotly.graph_objects as go
import plotly.express as px
from classes.aircraft import Aircraft, Engines
import classes.tailassignment as tap
from classes.events import FlightEvent, TurnaroundEvent, MaintenanceEvent
import numpy as np
import pandas as pd
import datetime as dt
from classes.tailassignment import od_pairs

num_needed_spares = 0

def read_config(file_path: str) -> cp.ConfigParser:
    """
    Reads and parses the configuration file.

    Parameters
    ----------
    file_path : str
        Path to the configuration file.

    Returns
    -------
    config : ConfigParser
        Parsed configuration object.
    """
    config = cp.ConfigParser()
    config.read(file_path)
    random_seed = config.getint('Simulation', 'random_seed')
    np.random.seed(random_seed)
    return config


def create_aircraft_fleet(num_aircraft: int, config: cp.ConfigParser) -> list:
    """
    Creates a fleet of aircraft based on the configuration.

    Parameters
    ----------
    num_aircraft : int
        Number of aircraft to create.
    config : ConfigParser
        Configuration object with settings for aircraft creation.

    Returns
    -------
    list of Aircraft
        List of created aircraft objects.
    """
    aircraft_fleet = [Aircraft(uid=n, config=config) for n in range(num_aircraft)]
    print("Created %d aircraft" % num_aircraft)
    return aircraft_fleet


def create_and_attach_engines(
    num_aircraft: int, aircraft_fleet: list, config: cp.ConfigParser
) -> list:
    """
    Creates engine sets, assigns them to aircraft, and returns the engine fleet.

    Parameters
    ----------
    num_aircraft : int
        Number of engines to create and assign.
    aircraft_fleet : list of Aircraft
        List of aircraft to which engines will be assigned.
    config : ConfigParser
        Configuration object with engine settings.

    Returns
    -------
    list of Engines
        List of created engine objects.
    """
    engine_fleet = []
    for n in range(num_aircraft):
        engines = Engines(uid=n, config=config, aircraft=aircraft_fleet[n])
        aircraft_fleet[n].attach_engines(engines)
        engine_fleet.append(engines)
    print("Created and attached %d engine sets" % len(engine_fleet))
    return engine_fleet


def create_spare_engines(
    num_aircraft: int, spare_engine_ratio: float, config: cp.ConfigParser
) -> list:
    """
    Creates a fleet of spare engines.

    Parameters
    ----------
    num_aircraft : int
        Number of aircraft in the fleet.
    spare_engine_ratio : float
        Ratio of spare engines to aircraft.
    config : ConfigParser
        Configuration object with engine settings.

    Returns
    -------
    list of Engines
        List of created spare engine objects.
    """
    num_spare_engines = int(np.ceil(spare_engine_ratio * num_aircraft))
    spare_engine_fleet = [
        Engines(uid=num_aircraft + n, config=config) for n in range(num_spare_engines)
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
    Main function to initialize the aircraft fleet and associated resources.

    Reads configuration settings, creates the aircraft fleet, attaches engines,
    creates spare engines, and initializes tail assignment.

    Returns
    -------
    None
    """
    global num_needed_spares

    # Read configuration
    config = read_config('config.ini')

    mro_base = config.get('Other', 'mro_base')

    # Create aircraft fleet
    num_aircraft = config.getint('Aircraft', 'num_aircraft')
    aircraft_fleet = create_aircraft_fleet(num_aircraft, config)

    # Create and attach engine sets to aircraft
    create_and_attach_engines(num_aircraft, aircraft_fleet, config)

    # Initialize engine tracking fleets
    shop_engine_fleet = []  # Engines currently in the shop
    spare_engine_fleet = []  # Spare engines available for replacement

    # Perform initial tail assignment
    aircraft_fleet = tap.initial(aircraft_fleet, config)

    print("Initialization complete. Ready to start operations.")

    # Set up global clock and simulation end time
    globalClock_str = config.get('Simulation', 'initial_time')
    globalClock = dt.datetime.strptime(globalClock_str, '%Y-%m-%d %H:%M:%S')
    sim_duration = config.getint('Simulation', 'sim_duration')
    end_time = globalClock + dt.timedelta(days=365 * sim_duration)

    # Track the last time scheduling was performed
    last_scheduled_date = None

    while True:
        # Find the next aircraft to process
        idx_aircraft = find_next_aircraft(aircraft_fleet)
        aircraft = aircraft_fleet[idx_aircraft]

        # Check for engine maintenance needs
        if aircraft.Engines.maintenance_due():

            skip_maintenance = False

            if aircraft.location != mro_base:

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
                aircraft.Engines.restore(globalClock)

                # Move the engine to the shop
                shop_engine_fleet.append(aircraft.Engines)

                # Replace the engine with a spare, creating a new one if necessary
                try:
                    spare_engine = spare_engine_fleet.pop(0)
                except IndexError:
                    num_needed_spares += 1
                    print(f"Need {num_needed_spares} spare engines.")
                    spare_engine = Engines(uid=1000 + num_needed_spares, config=config)

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
        globalClock = aircraft.last_tstamp

        # Reschedule if a week has passed
        if last_scheduled_date is None or (globalClock - last_scheduled_date) >= dt.timedelta(days=7):
            aircraft_fleet = tap.reschedule(aircraft_fleet)
            last_scheduled_date = globalClock

            # Check if engines in the shop are ready to be returned
            for idx, engine in enumerate(shop_engine_fleet):
                if globalClock > engine.esv_until:
                    spare_engine_fleet.append(engine)
                    del shop_engine_fleet[idx]

        # End simulation if the time limit is reached
        if globalClock > end_time:
            print("Simulation Done")
            break

    return aircraft_fleet, spare_engine_fleet, shop_engine_fleet



def postprocessingTAP(aircraft_fleet):

    # Example: assume you have a list of aircraft, each with event_calendar
    # events that have t_beg, t_end, and location.

    print("Preparing Airports")
    location_map = dict()
    for idx, key in enumerate(od_pairs.keys()):
        location_map[key] = idx

    print("preparing events")
    # Flatten data into a structure that can be easily processed:
    data = []
    for aircraft in aircraft_fleet:
        # Sort events by t_beg to ensure chronological order
        events = sorted(aircraft.event_calendar, key=lambda e: e.t_beg)

        for event in aircraft.event_calendar:
            # Check if it's a flight event or a ground event
            if event.type == 'Flight':
                # It's a flight event
                start_time = event.t_beg
                end_time = event.t_end
                origin_loc_val = location_map[event.orig]
                dest_loc_val = location_map[event.dest]

                data.append({
                    'aircraft_id': aircraft.uid,
                    'x': [start_time, end_time],
                    'y': [origin_loc_val, dest_loc_val]
                })
            else:
                # It's a ground event (maintenance/turnaround), same location for start and end
                start_time = event.t_beg
                end_time = event.t_end
                loc_val = location_map[event.location]

                data.append({
                    'aircraft_id': aircraft.uid,
                    'x': [start_time, end_time],
                    'y': [loc_val, loc_val]
                })

    # Now we have a list of line segments for each aircraft.
    # Create the figure
    fig = go.Figure()

    # Add one trace per aircraft (or per leg, depending on your preference)
    # If you want each aircraft to have one continuous line (if that makes sense for your data),
    # you could combine all segments for that aircraft. Otherwise, just plot segments individually.
    aircraft_ids = set(d['aircraft_id'] for d in data)

    print("Looping through aircraft IDs")
    for aircraft_id in aircraft_ids:
        # Filter segments for this aircraft
        segments = [d for d in data if d['aircraft_id'] == aircraft_id]

        # Each segment is a small line piece. If you want them connected, you can sort by start time
        # and flatten them into a single trace. If you prefer discrete segments, add them separately.

        # Flatten all segments for this aircraft:
        x_values = []
        y_values = []
        for seg in segments:
            # Add a gap between segments to avoid connecting them if times don't match
            # Or if you actually want a continuous line, ensure continuity.
            if x_values:  # If already have data, add None to create a break if needed
                x_values.append(None)
                y_values.append(None)
            x_values.extend(seg['x'])
            y_values.extend(seg['y'])

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=str(aircraft_id),
            line=dict(width=2)
        ))

    # Update the y-axis to show location names
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(location_map.values()),
        ticktext=list(location_map.keys()),
        title='Location'
    )

    fig.update_layout(
        title='Aircraft Movements Over Time',
        xaxis_title='Time',
        hovermode='x unified'
    )

    print("Showing Figure")
    fig.show(renderer="browser")


def postprocessingSOH(all_fleets):

    active_engines = [Aircraft.Engines for Aircraft in all_fleets[0]]
    spare_engines = all_fleets[1]
    shop_engines = all_fleets[2]

    # Combine the engine fleets
    engine_fleet = active_engines + spare_engines + shop_engines

    fig = go.Figure()

    # Assign a color per engine
    colors = px.colors.qualitative.Plotly
    engine_colors = {}
    for i, engine in enumerate(engine_fleet):
        # Map engine.uid to a color
        engine_colors[engine.uid] = colors[i % len(colors)]

    # Loop through each engine in the fleet
    for engine in engine_fleet:

        print(engine)
        history = engine.history

        # Convert the TIME (timedelta) to actual datetimes
        #time_series = [start_time - engine.initial_age + t for t in history['TIME']]
        time_series = history['TIME']

        color = engine_colors[engine.uid]

        # Add EGTM line (solid)
        fig.add_trace(go.Scatter(
            x=time_series,
            y=history['EGTM'],
            mode='lines',
            name=f'EGTM - Engine {engine.uid}',
            line=dict(color=color)
        ))

        # Add LLP line (dashed, same color)
        fig.add_trace(go.Scatter(
            x=time_series,
            y=history['LLP'],
            mode='lines',
            name=f'LLP - Engine {engine.uid}',
            line=dict(color=color, dash='dash'),
            yaxis='y2'
        ))

    fig.update_layout(
        title="EGTM and LLP over Time for All Engines",
        xaxis_title="Time",
        yaxis_title="EGTM",
        hovermode='x unified',
        yaxis2=dict(
            title="LLP",
            overlaying='y',
            side='right'
        )
    )

    fig.show(renderer='browser')

if __name__ == "__main__":
    fleets = main()
    #postprocessingTAP(fleets[0])
    postprocessingSOH(fleets)