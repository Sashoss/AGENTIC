
from agentic_ai.tools import Websearch
import json
from ollama import Client

class WebsearchAgent:
    def __init__(self, query, model= "llama3.2"):
        self.model = model
        self.USER_QUERY = query
        self.AGENT_CLIENT = Client()

        EVIDENCES = list()

        websearch_results = self.websearch_summarizer(self.USER_QUERY)
        evidence_report_1 = self.response_organizer(self.USER_QUERY, websearch_results)

        print(evidence_report_1)

        try:
            evidence_obj = json.loads(evidence_report_1)['evidence']
        except:
            evidence_obj = json.loads(evidence_report_1.lstrip("```json\n").rstrip("```"))['evidence'] # if ```json returned by llm json return

        EVIDENCES = EVIDENCES + evidence_obj
        subqueries_json = self.subquery_generator(self.USER_QUERY, websearch_results)


        subqueries_dict = json.loads(subqueries_json)


        subquery_results = dict()

        for key, subquery_obj in subqueries_dict.items():
            sub_result = self.websearch_summarizer(subquery_obj)
            sub_result_report = self.response_organizer(subquery_obj, sub_result)
            try:
                EVIDENCES = EVIDENCES + json.loads(sub_result_report)['evidence']
                subquery_results[key] = sub_result
            except:
                continue

        self.final_inference = self.inference_generator(
            main_query=self.USER_QUERY,
            evidence_report=EVIDENCES,
            subqueries=subqueries_json,
            sub_result=subquery_results
        )



    def websearch_summarizer(self, query):
        print("Running websearch for query\n")
        print(f"{query}")
        WBS = Websearch(query)
        websearch_results = WBS.textsearch()

        system_prompt = (
                        "You are an expert in web search result summarization. \n"
                        "Instructions:\n"
                        "Your job is to\n" 
                        "Step 1. Evaluate the web search results given as list of dictionaries as shown below,\n"
                        "where each dictionary is a query search output from different web link.\n\n"
                        f"{websearch_results}\n\n"
                        "Step 2. Find relevant information in the above results to answer users query in a research report format with appropriate context and heading."
                        "You must only use information available in web search json results. If there is no results in web search results then say I dont know. Do not make up any information."
        )

        user_prompt = (
                       query
        )

        response = self.AGENT_CLIENT.chat(
            model=self.model, 
            messages=[
                        {
                            'role': 'system',
                            'content': system_prompt,
                        },
                        {
                            'role': 'user',
                            'content': user_prompt,
                        }
                        
                    ]
            
        )

        return response["message"]["content"]

    def subquery_generator(self, query, web_context):
        system_prompt = (
                        "You are a domain-savvy assistant specialized in analyzing user queries in life sciences.\n\n"
                        "Your task:\n"
                        "1) Identify all unique entities in the evidence_list.\n"
                        "2) For each entity, generate subqueries exploring its role or function, especially in the context of the user query.\n"
                        "3) Generate additional bridging subqueries that explore potential interactions among the entities.\n"
                        "4) High priority should be given to generating subqueries to find relationship between entities from different evidences.\n\n"
                        "We can produce 10 or more subqueries as needed.\n"
                        "Your final output must be valid JSON with keys: subquery1, subquery2, ..., subqueryN.\n"
                        "Do not include any text outside the JSON.\n\n"
                        "For example, if we have an evidence list with:\n"
                        "[\n"
                        "  {\"entity1\": \"A\", \"relationship\": \"inhibits\", \"entity2\": \"B\"},\n"
                        "  {\"entity1\": \"C\", \"relationship\": \"transcription factor for\", \"entity2\": \"D\"}\n"
                        "]\n"
                        "Then:\n"
                        " - We consider the set of entities: {A, B, C, D}.\n"
                        " - We propose subqueries about each entity's role in the user query.\n"
                        " - We also propose bridging subqueries about A↔B, A↔C, A↔D, B↔C, B↔D, and C↔D if relevant.\n"
                        " - High priority should be given to generating subqueries to find relationship between entities from different evidences.\n\n"
                        "Example format for the final inference (JSON only, no extra text, character, note, newline, or spaces):\n\n"
                        "{\n"
                        "  \"subquery1\": \"...\",\n"
                        "  \"subquery2\": \"...\",\n"
                        "  ...\n"
                        "  \"subquery10\": \"...\"\n"
                        "}\n\n"
                    )

        user_prompt = (
                        "Generate a comprehensive set of subqueries to perform web search to obtain deeper understanding to answer original user query, \n"
                        f"{query}\n\n"
                        "Below are the round 1 web results obtained from user query:\n"
                        f"{web_context}\n\n"
        )

        response = self.AGENT_CLIENT.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': system_prompt,
                        },
                        {
                            'role': 'user',
                            'content': user_prompt,
                        }
                    ]
        )

        print("Subqueries generated\n")
        print(response["message"]["content"])

        return response["message"]["content"]

    def response_organizer(self, user_query, query_response):

        print("Organizing Agent query response\n")

        system_prompt = (
                        "You are an expert in biomedical research.\n\n"

                        "Your task:\n"
                            "1. Read the user's main query.\n"
                            "2. Review the 'main_result' (the output for the main query).\n"
                            "3. Identify relationship between entities within each sentence."
                            "4. Collect evidence to reference partial or direct relationship."
                            "5. If disclaimers exist (e.g., 'no direct evidence'), still see if partial data suggests an indirect relationship between the entities in main query by infering relationship between entities within main result.\n"
                            "6. Mention contradictions or missing data if relevant.\n"
                            "7. Output must be valid JSON. No extra text outside the JSON structure.\n\n"

                        "Important: If you see disclaimers about 'no direct evidence' but also see partial or stepwise data, "
                        "you can mention that an indirect chain of regulation is likely. Spell out the chain.\n\n"

                        "Example format for the final inference (JSON only, no extra text, character, note, newline, or spaces):\n\n"
                        "{\n"
                        "  \"inference\": \"...logic-based conclusion...\",\n"
                        "  \"evidence\": \"...reference partial or direct relationships...\",\n"
                        "  \"contradictions\": \"...if any...\",\n"
                        "  \"missing_data\": \"...if any...\"\n"
                        "}\n\n"
        )

        user_prompt = (
                    f"**Main Query:** {user_query}\n\n"
                    f"**Main Result:** {query_response}\n\n"
                    "Produce a final output in valid JSON. "
                    "If partial relationships suggest an indirect chain, connect those dots explicitly.\n\n"
                    "Example format for the final inference (JSON only, no extra text):\n\n"
                    "{\n"
                    "  \"inference\": \"...logic-based conclusion...\",\n"
                    "  \"evidence\": \"...reference factual evidence...\",\n"
                    "  \"contradictions\": \"...if any...\",\n"
                    "  \"missing_data\": \"...if any...\"\n"
                    "}\n\n"
        )

        try:
            response = self.AGENT_CLIENT.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}
            )
            final_inference = response["message"]["content"]


            return final_inference.strip()
        except Exception as e:
            return (
                "{\n"
                "  \"inference\": \"An error occurred while merging the results.\",\n"
                "  \"evidence\": \"\",\n"
                "  \"contradictions\": \"\",\n"
                "  \"missing_data\": \"\"\n"
                "}"
            )

    def inference_generator(self, main_query, evidence_report, subqueries, sub_result):
        print("Generating final inference\n")

        system_prompt = (
            "You are an expert in finding relationship between multiple sentences and their entities.\n\n"

            "Your task:\n"
                "1. Read the user's main query.\n"
                "2. Review the 'main_result' (the output for the main query).\n"
                "3. Review the 'subqueries' JSON (the extra steps) AND the 'sub_result' dict (answers to those subqueries).\n"
                "4. Identify relationship between entities within each sentence."
                "5. Identify relationship between entities from different sentences using shared information."
                "6. If the 'sub_result' describes indirect, partial or inferential relationships (like A→B, B→C, C-D leading to A-D), please connect them to see if A→D is implied.\n"
                "7. If disclaimers exist (e.g., 'no direct evidence'), still see if partial data suggests an indirect relationship between the enties in main query by infering relationship between entities within each sentences.\n"
                "8. Mention contradictions or missing data if relevant.\n"
                "9. Output must be valid JSON. No extra text outside the JSON structure.\n\n"

            "Important: If you see disclaimers about 'no direct evidence' but also see partial or stepwise data, "
            "you can mention that an indirect chain of relationship is likely. Spell out the chain.\n\n"

            "Example format for the final inference (JSON only, no extra text):\n\n"
                "{\n"
                "  \"inference\": \"...logic-based conclusion...\",\n"
                "  \"evidence\": \"...reference partial relationships...\",\n"
                "  \"contradictions\": \"...if any...\",\n"
                "  \"missing_data\": \"...if any...\"\n"
                "}\n\n"
        )

        user_prompt = (
            f"**Main Query:** {main_query}\n\n"
            f"**Evidence Report:** {evidence_report}\n\n"
            f"**Subqueries (JSON):** {subqueries}\n\n"
            "Below is a dictionary of subquery results:\n"
            "(keys = 'subquery1', 'subquery2', etc., values = the textual answer)\n\n"
            f"{sub_result}\n\n"
            "Please produce a final inference in valid JSON. "
            "If partial relationships suggest an indirect chain, connect those dots explicitly.\n\n"
            "Example format for the final inference (JSON only, no extra text):\n\n"
            "{\n"
            "  \"inference\": \"...logic-based conclusion...\",\n"
            "  \"evidence\": \"...reference partial relationships...\",\n"
            "  \"contradictions\": \"...if any...\",\n"
            "  \"missing_data\": \"...if any...\"\n"
            "}\n\n"
        )

        try:
            response = self.AGENT_CLIENT.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 1.0}
            )
            final_inference = response["message"]["content"]
            return final_inference.strip()
        except Exception as e:
            # Return a minimal valid JSON structure if error
            return (
                "{\n"
                "  \"inference\": \"An error occurred while merging the results.\",\n"
                "  \"evidence\": \"\",\n"
                "  \"contradictions\": \"\",\n"
                "  \"missing_data\": \"\"\n"
                "}"
            )


    def get_final_answer(self):
        return self.final_inference






