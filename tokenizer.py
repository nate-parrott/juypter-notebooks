import re

puncts = '!?,.'

def tokenize(text):
	for p in puncts:
		text = text.replace(p, ' ' + p + ' ')
	return re.split(r"\s+", text)
