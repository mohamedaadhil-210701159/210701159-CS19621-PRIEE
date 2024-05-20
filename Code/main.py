from dotenv import load_dotenv
import os
import asyncio
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Ensure an event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-chatbot")

async def find_match(question):
    input_em = model.encode(question).tolist()
    response = index.query(vector=input_em, top_k=2, includeMetadata=True, include_values=True)
    return response['matches'][0]['metadata']['text']

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

async def call_rag(question):
    template = """Use the following pieces of context to answer the question in a concise way at the end.
    Make sure the answer should be appropriate and within the context.
    Try to complete the answer at the end if it is not complete.
    If the user's question is hi or hello give "Hi, How can I assist you ?" as response.
    If the user's question is Thank you give "You're always welcome, Feel free to ask questions." as response.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": find_match, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    res = await rag_chain.ainvoke(question)
    return res

def run_async_function(async_func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_func(*args))

# Streamlit app

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

st.title("Government Schemes Information Chatbot")
st.subheader("Schemes Simplified, Lives Amplified.")

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Generating response..."):
            response = run_async_function(call_rag, query)
        st.session_state.requests.append(query)
        st.session_state.responses.append(str(response))

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i), avatar_style="bottts")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user',avatar_style="avataaars")
