import configparser


class ConfigReader:
    def __init__(self, config_file="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_simulation_settings(self):
        return {
            "initial_time": self.config.get("Simulation", "initial_time"),
            "number_of_aircraft": self.config.getint("Simulation", "number_of_aircraft")
        }

    def get_aircraft_settings(self):
        return {
            "min_age": self.config.getint("Aircraft", "min_age"),
            "max_age": self.config.getint("Aircraft", "max_age")
        }

    def get_engine_settings(self):
        return {
            "health_initial": self.config.getint("Engine", "health_initial")
        }
