from mrjob.job import MRJob
from mrjob import protocol

class FrequentVisitors(MRJob):
    OUTPUT_PROTOCOL = protocol.TextValueProtocol

    def get_name(self, line):
        fields = line.split(',')
        ## Ignore header
        if fields[0] == "NAMELAST":
            return None
        name = " ".join(filter(lambda x: x != "", [fields[1], fields[2], fields[0]]))
        return name.title()

    def mapper(self, _, line):
        name = self.get_name(line)
        if name != None:
            yield name, 1

    def combiner(self, name, counts):
        yield name, sum(counts)

    def reducer(self, name, counts):
        visits = sum(counts)
        if visits >= 10:
            yield None, name

if __name__ == "__main__":
    FrequentVisitors.run()