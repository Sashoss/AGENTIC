import ollama
import numpy
from ollama import chat
from ollama import ChatResponse
from agentic_ai.tools import Websearch

class WebsearchAgent:
    def __init__(self,  model= "llama3.2"):
        self.model = model 

    def websearch_summarizer(self, query):
        WBS = Websearch(query)
        websearch_results = WBS.textsearch()

        response = chat(
            model=self.model, 
            messages=[
                        {
                            'role': 'system',
                            'content': "You are an expert in finance and web search summarization. \n"
                                        "Instructions:\n"
                                        "Your job is to\n" 
                                        "Step 1. Evaluate the web search results given as list of dictionaries as shown below,\n"
                                        "where each dictionary is a query search output from different web link.\n\n"
                                        f"{websearch_results}\n\n"
                                        "Step 2. Find relevant information in the above results to answer users query in a research report format with appropriate context and heading."

                        },
                        {
                            'role': 'user',
                            'content': query,
                        }
                        
                    ]
            
        )

        return response["message"]["content"] 




