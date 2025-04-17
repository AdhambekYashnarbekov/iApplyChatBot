import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from vector import retriever

# Set up the model
model = OllamaLLM(model="llama3")

# Prompt template for university consultant
template = """
You are an expert advisor specialized in iApply.org — a global university application platform that connects students with universities offering Bachelor's, Master's, and PhD programs worldwide.

Your role is to provide accurate, concise, and helpful responses to users asking about universities, application requirements, deadlines, scholarships, program details, visa processes, and other related queries.

Here is some relevant information: {information}
Student's question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# --- Streamlit UI ---
st.set_page_config(page_title="University Consultant", page_icon="🎓")
st.title("🎓 University Application Consultant")
st.markdown("Ask me anything about universities, programs, fees, requirements, and more.")

# User input
question = st.text_input("What would you like to know?", placeholder="E.g. What are the requirements for LLB at Victoria University of Wellington?")

if question:
    with st.spinner("Thinking..."):
        reviews = retriever.invoke(question)
        result = chain.invoke({"information": reviews, "question": question})
        st.markdown("### 📌 Answer")
        st.markdown(result)
