import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
import plotly.figure_factory as ff
import plotly.offline as pyo

def plot_esv_gantt(time_bounds):
    """
    Creates a Gantt chart to visualize ESV time bounds for engines.

    Parameters
    ----------
    time_bounds : list of dict
        A list where each dict contains 'UID', 'START', and 'END' for an engine's ESV time bounds.
    """
    # Prepare data for Gantt chart
    gantt_data = [
        dict(
            Task="Engine %d" % item['UID'],  # Label for the row
            Start=item['START'].strftime("%Y-%m-%d %H:%M:%S"),  # Format start time
            Finish=item['END'].strftime("%Y-%m-%d %H:%M:%S"),    # Format end time
            SOMETHiNg='uhm',
            Resource="ESV"  # Label for the bar (can customize)
        )
        for item in time_bounds
    ]

    # Create the Gantt chart
    fig = ff.create_gantt(
        gantt_data,
        index_col="Resource",  # Color code based on the "Resource" column
        show_colorbar=True,  # Display color bar
        group_tasks=True,    # Group tasks by UID
        title="Engine Shop Visit (ESV) Time Bounds",
        showgrid_x=True,  # Show gridlines for time
        showgrid_y=True   # Show gridlines for engines
    )

    # Render the chart in the browser
    fig.show()


def visualize_opportunity_windows(opp_windows, mro_windows, scopes, esv_plan, eng_uid, chosen_windows=None):
    # Convert datetime values to strings for visualization
    df = []
    # Define scope boundaries for visualization
    scope_start, scope_end = scopes
    # Add MRO windows
    for start, end in mro_windows:
        df.append(dict(Task='Maintenance Windows', Start=max(start, scope_start), Finish=min(end, scope_end),
                       Resource='MRO Window'))
    # Add opportunity windows
    for start, end in opp_windows:
        df.append(dict(Task='Opportunity Windows', Start=max(start, scope_start), Finish=min(end, scope_end),
                       Resource='Opportunity'))
    # Add chosen MRO window if provided
    if chosen_windows:
        if isinstance(chosen_windows, list):
            for start, end in chosen_windows:
                df.append(dict(Task=f'MRO {eng_uid}', Start=start,
                               Finish=end, Resource='Chosen MRO'))
        else:

            df.append(dict(Task=f'MRO {eng_uid}', Start=chosen_windows[0],
                           Finish=chosen_windows[1], Resource='Chosen MRO'))
    # Add ESV plan times for all engines within the scope
    for engine_id, engine_plan in esv_plan.iterrows():
        df.append(dict(Task=f'ESV Plan {engine_id}', Start=engine_plan['TP_MIN'],
                       Finish=engine_plan['TP_MAX'], Resource='Predicted Maintenance'))
        df.append(dict(Task=f'ESV Plan {engine_id}', Start=engine_plan['TP_EST'],
                       Finish=engine_plan['T_END'],
                       Resource='Maintenance Period'))
    # Define colors for the different elements
    colors = {
        'Opportunity': 'rgb(220, 220, 220, 0.5)',
        'MRO Window': 'rgb(135, 206, 250, 0.5)',
        'Chosen MRO': 'rgb(255, 69, 0)',
        'Predicted Maintenance': 'rgb(60, 179, 113)',
        'Maintenance Period': 'rgb(255, 215, 0)'
    }
    # Create Gantt chart
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    # Show chart in browser
    pyo.plot(fig, filename='maintenance_schedule.html', auto_open=True)

# chosen_windows = [mro_windows[idx] for idx in list_chosen_mro_window_indices[lowest_dev_idx]]



def plot_esv_gantt_with_overlaps(time_bounds, overlapping_groups, save_as_html=False, filename="esv_gantt_with_overlaps.html"):
    """
    Creates a Gantt chart to visualize ESV time bounds, with hover text displaying overlapping UIDs.

    Parameters
    ----------
    time_bounds : list of dict
        A list where each dict contains 'UID', 'START', and 'END' for an engine's ESV time bounds.
    overlapping_groups : list of list
        A list of groups where each group contains UIDs of overlapping engines.
    save_as_html : bool, optional
        If True, saves the chart as an HTML file.
    filename : str, optional
        The name of the HTML file to save the chart.
    """
    # Create a dictionary to map UIDs to their overlapping UIDs
    overlap_map = {uid: [] for bounds in time_bounds for uid in [bounds['UID']]}
    for group in overlapping_groups:
        for uid in group:
            overlap_map[uid].extend([other for other in group if other != uid])

    # Prepare data for Gantt chart
    df = pd.DataFrame([
        {
            'UID': item['UID'],
            'Task': f"Engine {item['UID']}",
            'Start': item['START'],
            'End': item['END'],
            'Overlaps': ", ".join(map(str, overlap_map[item['UID']]))  # Convert overlapping UIDs to a string
        }
        for item in time_bounds
    ])

    # Create the Gantt chart
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Task",
        title="Engine Shop Visit (ESV) Time Bounds with Overlaps",
        labels={'Task': 'Engine'},
        hover_data={"Start": True, "End": True, "Overlaps": True},  # Include overlapping UIDs in hover text
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Engines", showlegend=False)

    # Show the chart in the browser
    fig.show()

    # Optionally save the chart as an HTML file
    if save_as_html:
        fig.write_html(filename)
        print("Chart saved as %s" % filename)


def visualize_gantt_with_scope_and_windows(time_bounds, windows, scope_start, scope_end, mng=None):
    """
    Visualizes the merged ESV time bounds, scope, and opportunity windows using a Gantt chart.

    Parameters
    ----------
    time_bounds : list of dict
        A list of dictionaries containing UID, START, and END times for ESVs.
    windows : list of tuple
        A list of tuples, where each tuple represents an opportunity window (start, end).
    scope_start : datetime
        The start of the scope.
    scope_end : datetime
        The end of the scope.
    """
    # Prepare the data for visualization
    gantt_data = []

    # Add merged ESV time bounds to the data
    for bound in time_bounds:
        gantt_data.append({
            "UID": f"Engine {bound['UID']}",
            "Start": bound['START'],
            "End": bound['END'],
            "Event": "ESV Time"
        })

    # Add a single row for opportunity windows
    for start, end in windows:
        gantt_data.append({
            "UID": "Opportunity Windows",
            "Start": start,
            "End": end,
            "Event": "Available Window"
        })

    # Add a row for the scope
    gantt_data.append({
        "UID": "Scope",
        "Start": scope_start,
        "End": scope_end,
        "Event": "Scope"
    })

    # Convert the data into a DataFrame for Plotly
    df = pd.DataFrame(gantt_data)

    if mng is None:
        title="Engine Shop Visit (ESV) Schedule, Scope, and Opportunity Windows"
    else:
        title=str(mng.SimTime)

    # Create the Gantt chart
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="UID",
        color="Event",  # Different colors for ESVs, windows, and scope
        title=title,
        labels={"UID": "Engine/Gap", "Event": "Event Type"},
    )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(title="Time", type="date"),  # Treat x-axis as datetime
        yaxis_title="Engines and Events",
        hovermode="x unified",
        height=600,
    )

    # Show the chart
    fig.show()