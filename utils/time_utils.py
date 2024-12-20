from datetime import datetime

def parse_time(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

def format_time(datetime_obj):
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
