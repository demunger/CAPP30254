from mrjob.job import MRJob
from mrjob import protocol

class VisitorVisitee(MRJob):
    OUTPUT_PROTOCOL = protocol.TextValueProtocol

    def get_names(self, line):
        fields = line.split(',')
        ## Ignore header
        if fields[0] == "NAMELAST":
            return None, None
        
        visitor = " ".join([fields[1], fields[0]]).title()
        visitee = " ".join([fields[20], fields[19]]).title()

        return visitor, visitee

    def mapper(self, _, line):
        visitor, visitee = self.get_names(line)
        if None not in [visitor, visitee]:
            yield visitor, "visitor"
            yield visitee, "visitee"
    
    def combiner(self, name, category):
        categories = list(category)
        for value in set(categories):
            yield name, value
    
    def reducer(self, name, category):
        categories = list(category)
        if len(set(categories)) > 1:
            yield None, name

if __name__ == "__main__":
    VisitorVisitee.run()