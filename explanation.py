from langchain import PromptTemplate, LLMChain
from langchain_community.llms import Ollama

def explain_word(word, context):
    llm = Ollama(model="llama3")

    explanation_templates = [     
       f"""
        What is "{word}"

        {context}
        """,
        f"""
        Explain the meaning of the word "{word}" in the context of the following text:

        {context}
        """,
        # ... (rest of the templates)
    ]

    prompt = PromptTemplate(
        input_variables=["word", "context"],
        template="\n\n".join(explanation_templates)
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True,
    )

    explanation = llm_chain.run({"word": word, "context": context})
    return explanation