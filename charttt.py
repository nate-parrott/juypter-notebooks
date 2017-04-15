import requests
from hashlib import sha256

def firebase_write(path, data, type='PUT'):
    requests.request(type, 'https://charttt-39854.firebaseio.com/{}.json'.format(path), json=data)

class Board(object):
    def __init__(self, key):
        self.key = key
    
    def chart(self, title):
        return Chart(self, title)

class Chart(object):
    def __init__(self, group, title):
        self.group = group
        self.title = title
        self.key = sha256(title).hexdigest()
        # write the title:
        firebase_write('{}/{}/title'.format(self.group.key, self.key), title)
    
    def write(self, x, **kwargs):
        d = {'x': x}
        for k, v in kwargs.iteritems(): d[k] = v
        firebase_write('{}/{}/values/'.format(self.group.key, self.key), d, type='POST')

# b = Board('test2')
# b.chart('Model 1').write(10, accuracy=5)
