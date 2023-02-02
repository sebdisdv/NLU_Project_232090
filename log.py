from pprint import pformat


class LogFile:
    def __init__(self, name):
        self.name = f"logs/{name}.txt"
        with open(self.name, "w") as _:
            pass

    def write(self, data_str: str):
        with open(self.name, "a") as logFile:
            logFile.write(data_str)

    def write_dict(self, dict):
        with open(self.name, "a") as logFile:
            logFile.write(pformat(dict))

    def write_sep_line(self):
        with open(self.name, "a") as logFile:
            logFile.write("/" * 30)
            logFile.write("\n")
