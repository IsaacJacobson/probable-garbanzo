"""
Main script to run the AI assistant for data exploration.
"""

"""
Example command is: python run.py -csv sample1.csv -q "Plot the relationship between capacitance and resistance"
"""


import os
import argparse
import time

from explorer import Explorer
from file_memory import FileMemory


def parse_args() -> None:
    """
    Parse command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Use an AI assistant to explore your data.")
    parser.add_argument('-q', '--query', type=str, help="Query in natural language.")
    parser.add_argument('-g', '--guidance', action='store_true', help="Assistant will return guidance. This is an experimental feature.")
    parser.add_argument('-csv', type=str, help="CSV file to be explored.")
    args = parser.parse_args()

    query = args.query
    if query is None:
        raise ValueError("Please specify a query using the -q or --query argument.")

    guidance = args.guidance
    csv = args.csv

    if csv is None:
        raise ValueError("Please specify a CSV file using the -csv argument.")

    if not os.path.exists(csv):
        raise FileNotFoundError(f"CSV file '{csv}' does not exist.")

    return csv, query, guidance

 

def main() -> None:
    """
    Main function to run the AI assistant for data exploration.
    """

    start_time = time.time()
    csv, query, guidance = parse_args()

    # Initialize the Explorer and FileMemory instances
    explorer = Explorer(guidance=guidance, csv=csv)
    memory = FileMemory(csv=csv)

    # Split the prompt into data query and task query
    prompts = explorer.split_prompt(query)

    # Get the needed data
    col_info = memory.sample_cols(prompts["data_query"])
    sql_statement = explorer.getSQL(prompts["data_query"], col_info)
    data = memory.execute_sql_statement(sql_statement)

    # Compute the result using the Explorer instance to write and execute python code
    result = explorer.compute(prompts["task_query"], data)

    # Get the response from the Explorer instance in a user-friendly format
    response = explorer.chatify_answer(result)

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    print(f"\nResponse: {response}")

 

if __name__ == "__main__":
    main()