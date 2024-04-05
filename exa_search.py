from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_exa import ExaSearchRetriever, TextContentsOptions
import os


os.environ["EXA_API_KEY"] = ""

retriever = ExaSearchRetriever(k=5, text_contents_options=TextContentsOptions(max_length=200))
prompt = PromptTemplate.from_template(
    """Answer the following query based on the following context:
    query: {query}
    <context>
    {context}
    </context>"""
)
llm = Ollama(model="tinydolphin")
chain = (RunnableParallel({"context": retriever, "query": RunnablePassthrough()}) | prompt | llm)
result = chain.invoke("What is the latest news from CNN.com?")

print(result)

