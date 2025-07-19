import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType


from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2
)

#db_user = "root"
#db_password = "1234"
#db_host = "localhost"
#db_name = "atliq_tshirts"
#db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost/atliq_tshirts")

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

question = "How many Nike white small size t-shirts are available?"
response = agent_executor.run(question)

print("Answer:", response)


while True:
    question = input("Question: ")
    if question.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    try:
        response = agent_executor.run(question)
        print("Answer:", response)
    except Exception as e:
        print("Error:", e)

import streamlit as st

st.title("ARVI TEES: DATABASE Q&A: ")

question = st.text_input("Question: ")
if question.lower() in ["exit", "quit"]:
    response = agent_executor.run(question)
    st.header("Answer: ")
    st.write(response)



# conda activate gemini
# streamlit run main2.py
