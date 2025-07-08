import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit page config
st.set_page_config(page_title="LangChain Chatbot", layout="centered")
st.title("🧠 LangChain Chatbot (Falcon)")

# Cache the model + pipeline to avoid reloading
@st.cache_resource
def load_chain():
    model_id = "tiiuae/falcon-7b-instruct"  # You may replace with a smaller model for CPU

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

    # HF pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        return_full_text=False,
    )

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=text_pipeline)

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the following question:\n\n{question}"
    )

    # Chain
    return LLMChain(llm=llm, prompt=prompt)

# Load once
chain = load_chain()

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask something:", key="input")

if user_input:
    st.session_state.chat_history.append(("🧍 You", user_input))

    with st.spinner("Thinking..."):
        try:
            reply = chain.run(user_input)
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

    st.session_state.chat_history.append(("🤖 Bot", reply))

# Display chat
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}**: {message}")
