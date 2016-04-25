from mrjob.job import MRJob
from mrjob import protocol
import re

class TwoYearVisitors(MRJob):
    OUTPUT_PROTOCOL = protocol.TextValueProtocol

    def get_name(self, line):
        fields = line.split(',')
        ## Ignore header
        if fields[0] == "NAMELAST":
            return None, None
        name = " ".join(filter(lambda x: x != "", [fields[1], fields[2], fields[0]]))
        
        ## Ignore missing dates
        if fields[11].strip() == "":
            return None, None
        date_form = re.compile(r"[0-9]{4}")
        date = date_form.findall(fields[11])
        ## No year found
        if len(date) != 1:
            return None, None

        return name.title(), date[0]

    def mapper(self, _, line):
        name, year = self.get_name(line)
        if name != None:
            yield name, year

    def combiner(self, name, years):
        visits = list(years)
        for year in set(visits):
            yield name, year

    def reducer(self, name, years):
        visits = list(years)
        if len(set(visits)) > 1:
            yield None, name

if __name__ == "__main__":
    TwoYearVisitors.run()