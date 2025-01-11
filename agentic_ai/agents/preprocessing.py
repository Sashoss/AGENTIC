import ollama
import numpy
from ollama import chat
from ollama import ChatResponse


class ProcessingAgent:
    def __init__(self,  model= "llama3.2"):
        self.model = model 


    def Minime(self, query):
        response = chat(
            model=self.model, 
            messages=[
                        {
                            'role': 'system',
                            'content': 'You are an expert in writing songs. Respond everything with song. Do not use normal sentences. You must always respond in songs.',
                        },
                        {
                            'role': 'user',
                            'content': query,
                        }
                        
                    ]
            
        )
        return response["message"]["content"] 




