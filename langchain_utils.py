from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama



# Modify the retrieve_relevant_page function to include the URL in the response
def retrieve_relevant_page(query, collection):
    results = collection.query(
        query_texts=[query],
        n_results=1,
    )
    if results and results['documents']:
        document = results['documents'][0]
        # print("document", document)
        metadata = results['metadatas'][0]
        # print("metadata", metadata)
        return document, metadata[0].get('url', '')
    else:
        return "", ""

def classify_query(prompt):
    llm = Ollama(model="llama3.1")

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
