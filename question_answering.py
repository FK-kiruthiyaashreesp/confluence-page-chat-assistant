import random
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import Ollama

def answer_question(question, context):
    llm = Ollama(model="llama3")

    qa_templates = [
        """
        Given the following context:
        {context}

        Answer the question: {question}
        """,

        """
        Here is the context:
        {context}

        What is the answer to: {question}
        """,

        """
        Provide the answer based on the context:
        {context}

        Question: {question}
        """,

        """
        Given this information:
        {context}

        Please answer the question: {question}
        """,

        """
        Answer this question using the following context:
        {context}

        Question: {question}
        """,

        """
        Context:
        {context}

        What is the answer to: {question}
        """
    ]

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=random.choice(qa_templates)
    )

    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True,
    )

    answer = llm_chain.run({"context": context, "question": question})
    return answer