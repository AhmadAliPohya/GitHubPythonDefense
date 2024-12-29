import plotly.graph_objects as go
import plotly.express as px
# from classes.tailassignment import od_pairs

def postprocessingTAP(mng):

    # Example: assume you have a list of aircraft, each with event_calendar
    # events that have t_beg, t_end, and location.

    print("Preparing Airports")
    location_map = dict()
    for idx, key in enumerate(od_pairs.keys()):
        location_map[key] = idx

    print("preparing events")
    # Flatten data into a structure that can be easily processed:
    data = []
    for aircraft in mng.aircraft_fleet:
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


def postprocessingSOH(mng):

    active_engines = [Aircraft.Engines for Aircraft in mng.aircraft_fleet]
    shop_engines = mng.shop_engine_fleet
    spare_engines = mng.spare_engine_fleet

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


def minimal_report(mng):
    """
    Generate a minimal report summarizing the Manager's state, configuration, 
    global parameters, and a per-aircraft summary.

    Parameters
    ----------
    mng : Manager
        The Manager object containing the simulation's state and data.

    Returns
    -------
    None
    """
    report_lines = []

    # First block: Config file content
    report_lines.append("Configuration Parameters:")
    for section in mng.config.sections():
        report_lines.append(f"[{section}]")
        for key, value in mng.config.items(section):
            report_lines.append(f"{key}: {value}")
        report_lines.append("")  # Add a blank line after each section

    # Add a blank line after the configuration block
    report_lines.append("")

    # Second block: Global parameters
    num_aircraft = len(mng.aircraft_fleet)

    # Count all shop visits by looking for "EngineExchange" in event calendars
    total_shop_visits = sum(
        1 for ac in mng.aircraft_fleet for event in ac.event_calendar
        if getattr(event, 'workscope', None) == "EngineExchange"
    )

    # Fleet averages
    avg_fc_per_year = sum(ac.fc_counter for ac in mng.aircraft_fleet) / num_aircraft
    avg_fh_per_fc = (
            sum(ac.fh_counter.total_seconds() / 3600 for ac in mng.aircraft_fleet) /
            sum(ac.fc_counter for ac in mng.aircraft_fleet)
    )

    report_lines.append("Global Parameters:")
    report_lines.append(f"Number of needed spares: {mng.num_needed_spares}")
    report_lines.append(f"Number of Aircraft on Ground (AOG) events: {mng.aog_events}")
    report_lines.append(f"Total number of engine shop visits: {total_shop_visits}")
    report_lines.append(f"Fleet average FC/year: {avg_fc_per_year:.2f}")
    report_lines.append(f"Fleet average FH/FC: {avg_fh_per_fc:.2f}")
    report_lines.append("")

    # Third block: Per-aircraft summary
    report_lines.append("Per-Aircraft Summary:")
    report_lines.append(f"{'Aircraft ID':<15}{'AOG Events':<15}{'Shop Visits':<15}"
                        f"{'Avg FC/Year':<15}{'Avg FH/FC':<15}")

    for ac in mng.aircraft_fleet:
        # Count shop visits for this aircraft
        shop_visits = sum(
            1 for event in ac.event_calendar
            if getattr(event, 'workscope', None) == "EngineExchange"
        )

        # Calculate per-aircraft averages
        avg_fc_per_year = ac.fc_counter / (ac.age.total_seconds() / (3600 * 24 * 365))
        avg_fh_per_fc = ac.fh_counter.total_seconds() / (3600 * ac.fc_counter)

        report_lines.append(f"{ac.uid:<15}{ac.aog_events:<15}{shop_visits:<15}"
                            f"{avg_fc_per_year:<15.2f}{avg_fh_per_fc:<15.2f}")

    # Output report to console
    for line in report_lines:
        print(line)

    # Save report to a .txt file
    with open("minimal_report.txt", "w") as file:
        file.write("\n".join(report_lines))

