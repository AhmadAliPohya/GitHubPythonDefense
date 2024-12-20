class Event:
    def __init__(self, event_type, time=None):
        self.event_type = event_type
        self.time = time


class FlightEvent:
    def __init__(self, name, orig, dest, dist, t_beg, t_dur):

        self.name = name
        self.orig = orig
        self.dest = dest
        self.dist = dist
        self.t_beg = t_beg
        self.t_dur = t_dur
        self.t_end = t_beg + t_dur
        self.location = self.dest
        self.type = 'Flight'

    def __repr__(self):
        total_seconds = int(self.t_dur.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        # Using old-style formatting:
        # Note: %02d ensures that minutes are always two digits.
        return ("Flight %s-%s, Dep: %s, Dur: %d:%02d hrs, Arr: %s" %
                (self.orig,
                 self.dest,
                 self.t_beg.strftime('%Y-%m-%d %H:%M'),
                 hours,
                 minutes,
                 self.t_end.strftime('%Y-%m-%d %H:%M')))


class TurnaroundEvent:
    def __init__(self, location, t_beg, t_dur):

        self.location = location
        self.t_beg = t_beg
        self.t_dur = t_dur
        self.t_end = t_beg + t_dur
        self.type = 'Turnaround'

    def __repr__(self):
        return f"TurnaroundEvent(duration={self.t_dur})"


class MaintenanceEvent:
    def __init__(self, Engines, location, t_beg, t_dur, workscope):
        self.location = location
        self.t_beg = t_beg
        self.t_dur = t_dur
        self.t_end = t_beg + t_dur
        self.type = 'Maintenance'
