from langchain_community.llms import Ollama
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun

wikipedia = WikipediaAPIWrapper()
wikipedia.run('Langchain')

python_repl = PythonREPL()
python_repl.run("print(17*2)")

search = DuckDuckGoSearchRun()
search.run("Tesla stock price?")

exit

from langchain.agents import Tool

llm = Ollama(model="llama2")

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

zero_shot_agent.run("When was Barak Obama born?")
zero_shot_agent.run("What is 17*6?")

print(zero_shot_agent.agent.llm_chain.prompt.template)
zero_shot_agent.run("Tell me about LangChain")
zero_shot_agent.run("Tell me about Singapore")
zero_shot_agent.run('what is the current price of btc')
zero_shot_agent.run('Is 11 a prime number?')
zero_shot_agent.run('Write a function to check if 11 a prime number and test it')
