import streamlit as st

state = st.session_state

def show_sidebar(): 
    # Render sidebar 
    st.sidebar.header("Cluster Options")
    st.sidebar.selectbox(
        'Select a HuggingFace model:',
        ('5CD-AI/Vietnamese-Sentiment-visobert', 
        'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'finiteautomata/bertweet-base-sentiment-analysis'), 
        key="model_name"
    )
    # st.sidebar.latex(r'''
    #         \text{tendency} = \frac{k \cdot \text{neg} + (1 - k) \cdot \text{pos} + \left(1 - |2k - 1|\right) \cdot \text{neu}}{k + (1 - k) + \left(1 - |2k - 1|\right)}
    #         ''')

    
    st.sidebar.write("From 0 for Positive and 1 for Negative")
    st.sidebar.slider(
        'Select a degree of negativity (k)',
        0.01, 0.99, 0.20, 0.01, 
        key="k"
    )
    st.sidebar.slider(
        'Select a threshold that consider Tendency(T) is valid',
        0.1, 0.9, 0.5, 0.1,
        key ="threshold"
    )
    st.sidebar.header("Comment Options")
    st.sidebar.number_input("Insert number of batch", value=1 , key="num_batch")
    st.sidebar.slider(
        'Select number of comments per batch',
        20, 100, 20, 5, 
        key="num_cmt_per_batch"
    )
    if state.num_batch and state.num_cmt_per_batch: 
        st.sidebar.write("Maximum number of comment analzying:", state.num_batch * state.num_cmt_per_batch, "comment(s)")
        st.sidebar.write("Estimated waiting time:", round(((state.num_batch * state.num_cmt_per_batch)/60), 3), "minute(s)")
    st.sidebar.header("Agent Options")
    st.sidebar.write("Use our AI Agent to help you discover insights from the video comment's sentiment and gives feedback. The Agent used RAG to optimized OpenAI gpt-4o-mini LLM through the book: The Youtube Formula by Derral Eves.")
    st.sidebar.toggle("Use AI agency", value=False, key='ai_toggle')


