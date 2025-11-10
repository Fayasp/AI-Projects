
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()
import os

@tool
def calculator(a:float, b:float,operation: str) -> str:
        """ Useful for performing  arithmetic operations. """
        operation = operation.lower()

        
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"{a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"{a} * {b} = {a * b}"
        elif operation == "divide":
            return f"{a} / {b} = {a / b}"

@tool
def say_hello(name: str) -> str:
    """ A simple function for greeting users. """
    return f"Hello, { name}! i hope you are well."

def main():
    model          = ChatOpenAI(temperature=0,openai_api_key = os.getenv("OPEN_API_KEY"), streaming=True)
    tool           = [calculator,say_hello]
    agent_executor = create_react_agent(model, tool)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            break

        print("\n Assistant: ", end="")

        for chunk in agent_executor.stream(
            {"messages":[HumanMessage(content = user_input)]}
            ):

            if "agent" in chunk and  "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
