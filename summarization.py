from langchain import PromptTemplate, LLMChain
from langchain_community.llms import Ollama

def summarize_text(text):
    llm = Ollama(model="llama3")

    summarization_template = """
    Summarize the following text:

    {text}
    """

    prompt = PromptTemplate(
        input_variables=["text"],
        template=summarization_template
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True,
    )

    summary = llm_chain.run(text)
    return summary