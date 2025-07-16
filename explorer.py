"""
This module provides the Explorer class to handle interaction with the Azure OpenAI API for exploring CSV files, including generating SQL queries, Python code, and executing tasks based on user input.
"""


import os
import json
import random
from typing import Optional, List, Dict, Union

from openai import OpenAI
from config import CHEAP_MODEL, SMART_MODEL, NUM_DATA_EXAMPLES, API_KEY


class Explorer():
    """
    AI assistant for exloring csv files
    """

    def __init__(self, guidance: bool, csv: str) -> None:

        self.csv = csv
        assert self.csv is not None, "CSV file path must be provided in the config if using Explorer."

        self.guidance = guidance
        self.output_dir = "output"
        self.client = OpenAI(api_key=API_KEY)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

 

    def llm_complete(self, prompt: str, model=SMART_MODEL) -> str:
        """ Uses the Azure OpenAI API to query an LLM. """
        response = self.client.responses.create(
            model=model,
            instructions="You are a knowledgeable, thoughtful, and careful AI assistant specializing in Quantum computing.",
            temperature=0.7,
            input=prompt,
        )

        return response.output_text
    

    def extractObjectFromResult(self, result: str, object_type="python") -> str:
        """
        Extracts a specific object type from the result string.
        """

        if result.startswith(f"'''{object_type}"):
            result = result.split(f"'''{object_type}", maxsplit=1)[1].strip()
            if result.endswith("'''"):
                result = result.split("'''", maxsplit=1)[0].strip()
        elif result.startswith(f"```{object_type}"):
            result = result.split(f"```{object_type}", maxsplit=1)[1].strip()
            if result.endswith("```"):
                result = result.split("```", maxsplit=1)[0].strip()
        return result


    def split_prompt(self, user_input: str) -> Dict[str, str]:
        """
        Splits the user input into two queries: one for data retrieval and one for task processing.
        """

        prompt = f"""You are the planning agent for a quantum error correction assistant. The goal is to use the data at your disposal to answer the user's question. \
As the planner, your job is to determine what task the user is asking for and what data will be needed to answer the query. \
Given the user's input, create two new queries one that decribes the task the user would like to be completed and one that describes the data needed to complete the task. \
These queries that you return will be sent to other agents, one to retrieve data from a database and one to process the data completing the task. \
The query for the data retriever should be relatively minimalistic and only contain the necessary keywords to retrieve relevant data. \
The query for the task should be more detailed and contain all the necessary information to complete the task. \
It is okay to expand on what the user asked for (for either agent) if you think it will help the agents complete their tasks or the user's input is ambiguous this includes expanding common \
acronyms or abbreviations or adding acronyms or abbreviations so that both the acronym and the full form are included in the query. \
For example if the user's input is: "What are the 5 ranks with the highest rank cost? Plot them in a bar chart." you shoud return a json with the following 2 queries:
data_query : "5 ranks with the highest rank cost"
task_query : "Plot the 5 ranks with the highest rank cost in a bar chart and save the plot to the output directory."
Another example is if the user's input is: "What is the average pcm?" you should return a json with the following 2 queries:
data_query : "pcm (Parity Check Matrix)"
task_query : "Calculate the average pcm (Parity Check Matrix) and return the result."
Here is the user's input:
{user_input}
Now output the two queries for the the other agents carefully considering the scope of each agent and making sure you are confident in your answer. \
Your output must be a json object with the keys 'data_query' and 'task_query'. The value of 'data_query' should be a short query to retrieve relevant data from a database. The value of 'task_query' should be a detailed query to complete the user's task. Include no other text or explanation.
Output: """

        res = self.llm_complete(prompt, model=SMART_MODEL)
        res = self.extractObjectFromResult(res, object_type="json")
        res = json.loads(res)
        res["user_input"] = user_input
        self.user_input = user_input
        return res
 

    def chatify_answer(self, response: str) -> str:
        """
        Formats the response from the assistant into a user-friendly format.
        """

        full_prompt = f"""You are the user interface component of an quantum error correction assistant, you are responsible for formatting the response from the assistant into a user-friendly format. \
Make sure to answer the user's original query in a clear and concise manner. \
The user's original query was:
{self.user_input}
The response from the quantum error correction assistant is (the directory {self.output_dir} contains any plots or files generated by the assistant):
{response}
Now format the response in a user-friendly way, don't use markdown, simply provide text. Do not include any unnecessary explanation or punctuation that is not asked for. \        
Response: """

        if self.guidance:
            guidance_prompt = f"""Finally, in addition to answering the user's question, you should also suggest next steps to the user based on their recent queries. \
You should suggest between 1 and 3 independent next steps to the user. \
These next steps should be independent of each other and in the form of possible queries the user can make to the quantum error correction assistant. \
Also make sure not to suggest the same query multiple times or suggest queries that are too similar to those the user has already asked. \
Assume each of the users previous queries as been appropriately answered, that answer is just not shown here. \
Remember you should suggest next steps to the user do not provide any justification or explanation just a numbered list of next steps in the form of queries. Include at least one but no more than three. \
Preface your suggest next steps with 'Potential next steps could be:'. \
Okay, now you're ready to answer the user's question and suggest next steps.
Response: """

            full_prompt = full_prompt[:-10] + guidance_prompt

        res = self.llm_complete(full_prompt, model=CHEAP_MODEL)
        return res


    def getSQL(self, data_query: str, col_info: str) -> str:
        """
        Generates a SQL query based on the user's data query and column information.
        """

        prompt = f"""You are the SQL generation agent for an quantum error correction assistant. \
Your job is to generate a SQL query that retrieves the data needed to answer the user's question. \
You must output a valid SQL query that can be executed on the database and nothing else. \
Make sure to include all the necessary columns and filtering in the query. \
If it is unclear or you are unsure it is better to include more data than less. \
If the user is asking a calculation it is okay to perform the calculation in the SQL query. \
However, it is not necessary to perform any calculations in the SQL query as there is a separate agent that will process the data and perform any calculations needed. \
If their is a compound or complex calculation do not perform it in the SQL query and instead just retrieve the data needed to perform the calculation. \
The table name is debug_table and the table contains the following columns (with some example values):
{col_info}
The user's query is:
{data_query}
Now generate a SQL query that retrieves the data needed to answer the user's question. \
Do not include any other text, explanation, or punctuation, just the SQL query. \
Do not include any comment or ``` marks or anything else, just the SQL query. \
SQL Query: """

        res = self.llm_complete(prompt, model=CHEAP_MODEL)
        return res


    def format_code(self, code: str, data: dict) -> str:
        """
        Formats the generated Python code to include the data dictionary and a main function.
        """

        output = f"def main():\n\n\tdata = {str(data)}\n"
        for line in code.splitlines():
            output += f"\t{line}\n"
        output += f"\treturn result\n\nresult = main()"
        return output

 
    def sample_data(self, data: dict, k: Optional[int]=NUM_DATA_EXAMPLES) -> str:
        """
        Samples a subset of the data dictionary for use in code generation.
        """

        output = {}
        for key, values in data.items():
            if isinstance(values, list):
                output[key] = random.sample(values, min(k, len(values)))
            else:
                output[key] = values
        return str(output)


    def getPythonCode(self, task_query: str, data: dict) -> str:
        """
        Generates Python code to complete the task described in the user's query using the provided data.
        """

        sample_data = self.sample_data(data)
        prompt = f"""You are given a task query and some data. Your task is to generate python code that will complete the task described in the query using the data provided. \
The task query is:
{task_query}
The data is in a dictionary format like the one below, assume the same structure and keys as the one below however there may be a longer list in the valules. \
Your code should be able to handle this and not hardcode any dictionary values instead assume the dictionary exists with the same name, structure, and keys as this one and query it appropriately. \
Do not include the dictionary in your code, just assume it exists and is named 'data'. \
The data is:
{sample_data}
You should generate python code that will complete the task described in the query using the data provided in a thoughtful and careful manner. \
Make sure to understand the task and the data before generating the code. \
Do not include any other text, explanation, or punctuation, just the python code. \
If you need to generate a plot, it should be saved to the directory {self.output_dir} as they can not be passed back to the user, only return the filepath to where they have be written. \
Do not save text to files, only plots. Also do not generate plots or files unless the user has specifically asked for it. \
There is no need to include return statments instead after your code is done executing there should be a variable named result that contains the result of the code execution. \
Okay, now it is time to generate the code, remember to be thoughtful and careful in your code generation and omit any extra explanation, punctuation, ``` marks, or text. \
Python code: """

        res = self.llm_complete(prompt, model=SMART_MODEL)
        res = self.extractObjectFromResult(res, object_type="python")
        return self.format_code(res, data)


    def execute_python_code(self, python_code: str) -> Union[str, List[str]]:
        """
        Executes the generated Python code and returns the result.
        """

        ldict = {}
        try:
            exec(python_code, globals(), ldict)
            result = ldict.get('result', None)
        except Exception as e:
            raise RuntimeError(f"Failed to execute Python code: {e}")
        return result

    
    def compute(self, task_query: str, data: dict) -> Union[str, List[str]]:
        """
        Computes the result for the task described in the task_query using the provided data.
        """

        python_code = self.getPythonCode(task_query, data)        
        result = self.execute_python_code(python_code)
        return result