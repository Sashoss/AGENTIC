
from duckduckgo_search import DDGS

class Websearch:
    def __init__(self, query):
        self.query = query 

    def textsearch(self,  region = 'wt-wt', safesearch = 'off' , timelimit = 'y', max_results = 10):
        result = DDGS().text(keywords = self.query, region=region, safesearch=safesearch, timelimit=timelimit, max_results=max_results)
        return result
    