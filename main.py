from agentic_ai import ProcessingAgent
from agentic_ai import WebsearchAgent

wb_search = WebsearchAgent("Who is the strongest avenger?")
result = wb_search.get_final_answer()
print(result)
