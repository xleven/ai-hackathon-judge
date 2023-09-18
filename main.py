import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from judge import get_judge


DEFAULT_TITLE = "Streamlit LLM Hackathon"

DEFAULT_INTRO = """Build an innovative LLM-based Streamlit app that incorporates at least one of the following LLM technologies: LangChain, AssemblyAI, Weaviate, LlamaIndex, or Clarifai.
There are five “Most Innovative Use” prize categories, one for each partner listed above.
In each category, there will be two lucky app winners. You can submit your app alone, or as a team of two.
Winners will be announced by October 5.
Join the #llm-hackathon channel on Discord to get inspiration, ask questions, and participate in mini-giveaways.
Representatives from all partners will be available to guide you as you build your LLM-based apps."""

DEFAULT_JUDGING = """1. Inventive
Your app offers new features not found in other Streamlit apps. The more unique, the better.
2. Error-Free
Your app doesn’t produce any errors during testing.
3. Public GitHub Repository
Your app’s source code is public—using secrets management to protect your API keys and credentials.
4. Hosted on Community Cloud
Your app must be publicly accessible on Streamlit's Community Cloud.
5. Tools Used
Your app uses at least one partner: LangChain, LlamaIndex, Weaviate, AssemblyAI, or Clarifai.
6. LLM Pain Points
You’ll get bonus points if your app addresses common LLM pain points like transparency, trust, accuracy, privacy, cost reduction, or ethics."""

DEFAULT_APPS = """# Support GitHub repos. One repo each line in the form of `user/repo`, e.g.
xleven/ai-hackathon-judge
mmz-001/knowledge_gpt
"""


st.set_page_config(page_title="AI Hackathon Judge", page_icon="🤖")
st.title("🏅 AI Hackathon Judge 🏃🏻")


ss = st.session_state

with st.sidebar:
    ss.openai_api_key = st.text_input("Your OpenAI API key", placeholder="sk-xxxx", type="password")
    ss.model = st.selectbox("Model", (
        "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k", "gpt-4-16k",
    ))
    ss.temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, format="%.1f")
    ss.show_intermediate_steps = st.checkbox("Show intermediate steps")


st.info("""
[AI Hackathon Judge](https://github.com/xleven/ai-hackathon-judge) is an AI judge for hackathons.
Built by [xleven](https://github.com/xleven) with [LangChain](https://github.com/langchain-ai/langchain) and [Streamlit](https://streamlit.io).
""", icon="ℹ️")

with st.form("hackathon_info"):
    
    st.subheader("Hackathon Info")
    title = st.text_input("Title", DEFAULT_TITLE)
    intro = st.text_area("Intro", DEFAULT_INTRO)
    judging = st.text_area("Judging Criteria", DEFAULT_JUDGING)

    st.divider()

    st.subheader("Submit Apps")
    # apps = st.text_area("Repos", placeholder=DEFAULT_APPS)
    apps = st.text_input("Repo", placeholder="xleven/ai-hackathon-judge")

    submit_button = st.form_submit_button(label="Submit")


if submit_button:
    st.subheader("Judging Result")
    hackathon_info = {"title": title, "intro": intro, "judging": judging}
    model_config = {
        "openai_api_key": ss.openai_api_key,
        "model": ss.model,
        "temperature": ss.temperature,
    }
    try:
        judge = get_judge(hackathon_info, model_config)
        if ss.show_intermediate_steps:
            handler = StreamlitCallbackHandler(st.container(), max_thought_containers=10)
            result = judge.run(apps, callbacks=[handler])
        else:
            with st.status("Judging..."):
                result = judge.run(apps)
                st.write("Judging finished.")
    except Exception as err:
        st.error(err)
    else:
        st.success(result, icon="✅")