from langchain import PromptTemplate, LLMChain
from langchain_community.llms import Ollama

def classify_query(prompt):
    llm = Ollama(model="llama3")

    classification_template = """
    Classify the user's query into one of the following categories:
    1. Question Answering
    2. Word Explanation
    3. Summarization

    User's query: {prompt}

    Based on the user's query, I believe this is a:
    """

    prompt = PromptTemplate(
        input_variables=["prompt"],
        template=classification_template
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True,
    )

    classification = llm_chain.run({"prompt": prompt})

    if "Summarization" in classification:
        return "Summarization"
    elif "Question Answering" in classification:
        return "Question Answering"
    elif "Word Explanation" in classification:
        return "Word Explanation"
    else:
        return "Unknown"
