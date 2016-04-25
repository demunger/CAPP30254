from mrjob.job import MRJob
from mrjob import protocol
import heapq

class FrequentVisitees(MRJob):
    OUTPUT_PROTOCOL = protocol.TextValueProtocol

    def get_name(self, line):
        fields = line.split(',')
        ## Ignore header
        if fields[19] == "visitee_namelast":
            return None
        ## Ignore single-name anomalies
        if fields[19] == "" or fields[20] == "":
            return None
        ## Ignore Visitor's Office anomaly
        if fields[19].lower() == "office" and fields[20].lower().strip() == "visitors":
            return None

        return " ".join([fields[20], fields[19]]).title()

    def mapper(self, _, line):
        name = self.get_name(line)
        if name != None:
            yield name, 1

    def combiner(self, name, counts):
        yield name, sum(counts)

    def reducer_init(self):
        self.heap = [(0, "")] * 10

    def reducer(self, name, counts):
        visitors = sum(counts)
        min_visitors, min_name = self.heap[0]
        if visitors > min_visitors:
            heapq.heapreplace(self.heap, (visitors, name))

    def reducer_final(self):
        self.heap.sort(reverse = True)
        for x in range(10):
            yield self.heap[x]

if __name__ == "__main__":
    FrequentVisitees.run()