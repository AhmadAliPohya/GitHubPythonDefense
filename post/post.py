import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def visualize_health(mng):
    # Extract all engines from the management object
    active_engines = [Aircraft.Engines for Aircraft in mng.aircraft_active]
    shop_engines = mng.engines_in_shop
    spare_engines = mng.engines_in_pool

    # Combine all engine fleets into a single list
    engine_fleet = active_engines + spare_engines + shop_engines

    # Sort engines by their UID
    engine_fleet = sorted(engine_fleet, key=lambda engine: engine.uid)

    # Assign a unique color to each engine
    colors = px.colors.qualitative.Plotly
    engine_colors = {engine.uid: colors[i % len(colors)] for i, engine in enumerate(engine_fleet)}

    # Define rows and columns
    metrics = ['EGTM', 'LLP'] #, 'SOH']  # Rows
    dimensions = ['TIME', 'EFCs', 'EFHs']  # Columns

    # Create a 3x3 subplot grid with shared x and y axes
    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,  # Reduced horizontal spacing
        vertical_spacing=0.02  # Reduced vertical spacing
    )

    # Populate the subplots
    for engine in engine_fleet:
        history = engine.history

        # Extract and prepare x-axis data
        x_time = history['TIME']  # datetime.datetime objects
        x_efcs = history['EFCs']  # Numeric
        x_efhs = [t.total_seconds() / 3600 for t in history['EFHs']]  # Convert timedelta to hours

        x_data = {
            'TIME': x_time,
            'EFCs': x_efcs,
            'EFHs': x_efhs
        }

        y_data = {
            'EGTM': history['EGTM'],
            'LLP': history['LLP'],
            # 'SOH': history['SOH']
        }

        color = engine_colors[engine.uid]

        for row_idx, metric in enumerate(metrics, 1):
            for col_idx, dimension in enumerate(dimensions, 1):
                # Determine if this is the first trace for the engine to show in the legend
                show_legend = (row_idx == 1 and col_idx == 1)

                fig.add_trace(
                    go.Scatter(
                        x=x_data[dimension],
                        y=y_data[metric],
                        mode='lines',
                        name=f'Engine {engine.uid}' if show_legend else None,
                        line=dict(color=color),
                        showlegend=show_legend,
                        legendgroup=f"Engine {engine.uid}"  # Group traces by engine UID
                    ),
                    row=row_idx,
                    col=col_idx
                )

    # Update layout for aesthetics
    fig.update_layout(
        height=900,  # Adjust height as needed
        width=1850,  # Adjust width to fit 16:9 aspect ratio
        hovermode='closest',
        legend=dict(
            title="Engines",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1  # Move legend to the far right
        ),
        margin=dict(l=10, r=10, t=25, b=10),  # Reduce white space
    )

    # Update individual axis labels for shared axes
    for row_idx, metric in enumerate(metrics, 1):
        fig.update_yaxes(title_text=metric, row=row_idx, col=1)  # Set y-axis label for first column
    for col_idx, dimension in enumerate(dimensions, 1):
        fig.update_xaxes(title_text=dimension, row=3, col=col_idx)  # Set x-axis label for last row

    # Display the figure in the browser
    fig.show(renderer='browser')











def postprocessingSOH(mng):

    active_engines = [Aircraft.Engines for Aircraft in mng.aircraft_active]
    shop_engines = mng.engines_in_shop
    spare_engines = mng.engines_in_pool

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
        xaxis_title="TIME",
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
    num_aircraft = len(mng.aircraft_active)

    # Count all shop visits by looking for "EngineExchange" in event calendars
    total_shop_visits = sum(
        1 for ac in mng.aircraft_active for event in ac.event_calendar
        if getattr(event, 'workscope', None) == "EngineExchange"
    )

    # Fleet averages
    avg_fc_per_year = sum(ac.fc_counter for ac in mng.aircraft_active) / num_aircraft
    avg_fh_per_fc = (
            sum(ac.fh_counter.total_seconds() / 3600 for ac in mng.aircraft_active) /
            sum(ac.fc_counter for ac in mng.aircraft_active)
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
    report_lines.append(f"{'Aircraft ID':<15}{'ASK':<15}{'Shop Visits':<15}"
                        f"{'Avg FC/Year':<15}{'Avg FH/FC':<15}")

    total_ask = 0


    for ac in mng.aircraft_active:
        # Count shop visits for this aircraft
        shop_visits = sum(
            1 for event in ac.event_calendar
            if getattr(event, 'workscope', None) == "EngineExchange"
        )

        # Calculate per-aircraft averages
        avg_fc_per_year = ac.fc_counter / (ac.age.total_seconds() / (3600 * 24 * 365))
        avg_fh_per_fc = ac.fh_counter.total_seconds() / (3600 * ac.fc_counter)

        ask = round(sum([el.dist for el in ac.event_calendar if el.type == 'Flight']),0)

        total_ask += ask


        report_lines.append(f"{ac.uid:<15}{ask:<15}{shop_visits:<15}"
                            f"{avg_fc_per_year:<15.2f}{avg_fh_per_fc:<15.2f}")

    report_lines.append("")
    report_lines.append("Per-Engine Summary:")
    report_lines.append(f"{'Engine ID':<15}{'Lost LLP life':<15}{'Lost EGTM °C':<15}")

    engines = []
    for ac in mng.aircraft_active:
        engines.append(ac.Engines)
    for engine in mng.engines_in_pool:
        engines.append(engine)
    for engine in mng.engines_in_shop:
        engines.append(engine)

    total_llp_lost = 0
    total_egtm_lost = 0

    for eng in engines:

        total_llp_lost += eng.llp_rul_lost
        total_egtm_lost += eng.egtm_lost

        report_lines.append(f"{eng.uid:<15}{eng.llp_rul_lost:<15}{eng.egtm_lost:<15}")



    # Output report to console
    for line in report_lines:
        print(line)

    print("\n")
    print("Total ASK: %.0f" % total_ask)
    print("Total LLP lost: %.0f" % total_llp_lost)
    print("Total EGTM lost: %.0f" % total_egtm_lost)
    print("Total No. of Ferry Flights: %d" % len(mng.ferry_flights))


    # Save report to a .txt file
    with open("minimal_report.txt", "w") as file:
        file.write("\n".join(report_lines))

