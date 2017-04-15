def flatten(lists):
    return [item for list in lists for item in list]

def avg(items):
    return sum(items) / float(len(items))
