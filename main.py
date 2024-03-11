import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from trulens_eval import Feedback, Huggingface, Tru, TruChain
from trulens_eval.feedback.provider.hugs import Huggingface

#load environment variables
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

template = """
You are a professional customer support specialist chatbot, dedicated to providing helpful, accurate, and polite responses. 
Your goal is to assist users with their queries to the best of your ability. 
If a user asks something outside of your knowledge, politely inform them that you 
don't have the information they need and, if possible, suggest where they might find it. 
Remember to always maintain a courteous and supportive tone.

{chat_history}
Human: {human_input}
Chatbot:
"""
prompt = PromptTemplate(input_variables=["chat_history","human_input"], template=template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo" )

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt,verbose=True, memory=memory)

hugs = Huggingface()
tru = Tru()

f_lang_match = Feedback(hugs.language_match).on_input_output()
feedback_nontoxic = Feedback(Huggingface.not_toxic).on_output()
f_pii_detection = Feedback(hugs.pii_detection).on_input()
feedback_positive = Feedback(Huggingface.positive_sentiment).on_output()

# TruLens chain recorder
chain_recorder = TruChain(
    conversation,
    app_id="contextual-chatbot",
    feedbacks=[f_lang_match, feedback_nontoxic, f_pii_detection, feedback_positive],
)

st.title("Contextual Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("What is up?")
if user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Construct the input for the conversation
        input_dict = {
            "chat_history": "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages]),
            "human_input": user_prompt
        }

        # Record conversation with TruLens
        try:
            with chain_recorder as recording:
                response = conversation(input_dict)
                assistant_response = response["chat_history"][1].content.strip()
        except Exception as e:
            assistant_response = f"An error occurred: {e}"
            st.error("Error in generating response. Please try again.")

        # Simulate stream of response
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
tru.run_dashboard()