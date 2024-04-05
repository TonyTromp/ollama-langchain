from langchain_community.llms import Ollama
#from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_openai import OpenAI

wikipedia = WikipediaAPIWrapper()
python_repl = PythonREPL()
search = DuckDuckGoSearchRun()

# wikipedia.run('Langchain')
# python_repl.run("print(17*2)")
# search.run("Tesla stock price?")


#llm = Ollama(model="llama2")
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

tools = [
    Tool(
        name = "python repl",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input python code"
    )
]
wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(duckduckgo_tool)
tools.append(wikipedia_tool)

from langchain.agents import initialize_agent

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

# print(zero_shot_agent.agent.llm_chain.prompt.template)
#zero_shot_agent.run("When was Barak Obama born?")
#zero_shot_agent.run("What is 17*6?")
zero_shot_agent.run('what is the current price of btc')
# zero_shot_agent.run('Is 11 a prime number?')
# zero_shot_agent.run('Write a function to check if 11 a prime number and test it')
zero_shot_agent.run('what is the most profitable crypto currency this month?')
