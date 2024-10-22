import streamlit as st 
import pandas as pd 
import numpy as np 
from src.plot_table import plot_table
from src.show_sidebar import show_sidebar
import plotly.express as px
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI 
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os 
from llama_index.core import PromptTemplate
from src.comment import CommentInteractor
from src.video import YouTubeVideo
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import streamlit as st
from llama_index.core import SimpleDirectoryReader
state = st.session_state
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if "messages" not in st.session_state:
    state.messages = []

if "init_chatbot" not in st.session_state: 
    state.init_chatbot = False
# Initialize session state
state = st.session_state
if 'analyzed' not in state: 
    state.analyzed = False

st.set_page_config(
    layout="wide", page_title="SENTAL", page_icon="🤠"
)

show_sidebar()

if "analyzed_comments" not in state: 
    state.analyzed_comments = []

# Render Homepage 
st.title("SENTAL: YouTube's Videos Sentiment Analysis")
st.subheader("An interactive sentiment analyzing dashboard for content creators")
st.markdown("App by [Hoang Quan](https://github.com/quannguyen2306-ctrl) | This webpage lets you analyze your audience comments on your YouTube channel using different pre-trained models from Hugging Face (huggingface.com).")


def predict_sentiment(model_name, comment: str, k: float) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cpu")
    
    with torch.no_grad():
        encoded_input = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoded_input['input_ids'].to("cpu")
        out = model(input_ids)
        scores = out.logits.softmax(dim=-1).cpu().numpy()[0]
        neg, pos, neu = scores
        
        tendency = (k * neg + (1 - k) * pos + (1 - abs(2 * k - 1)) * neu) / (k + (1 - k) + (1 - abs(2 * k - 1)))
        
        return (tendency, neg, pos, neu)


def init_chatbot(): 
    print(not state.init_chatbot)
    print("Initializing Chat-bot........ ")
    ANSWER_TEMPLATE = ( 
        "According to the book: The YouTube Formula, and the comment from the video {comment}, and the information of the video, title: {title}, author: {author}. If these information is empty, then ask the user to input the youtube video url and analyze"
        "What is the best solution for this problem:"
        "{question_str}?"
        "Answer like the author of the book, reference something one the book but not act as a query engine, also referent to the comment and the title"
    )

    Settings.llm = OpenAI(model = "gpt-4o-mini", temperature=0)

    # Ingestion data
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    documents = SimpleDirectoryReader(input_dir=data_path).load_data()
    # store ingrestion
    PERSIST_DIR = "index_cache"

    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        state.index = VectorStoreIndex.from_documents(documents)
        # store it for later
        state.index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        state.storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        state.index = load_index_from_storage(state.storage_context)

    state.qa_template = PromptTemplate(ANSWER_TEMPLATE)
    state.init_chatbot = True

# Create a form container
form = st.container()
with form:
    st.header("Video Analysis")
    input_url, btn_url = st.columns(2, vertical_alignment="top")
    
    with input_url: 
        st.text_input("URL:", key="video_url", placeholder="e.g: https://youtu.be/id")
    
    with btn_url:
        # Adding a callback to analyze the video
        if st.button("Analyze", key="video_url_submit"):

            state.analyzed = False  # Reset for new analysis
            print("Testing\n")
            print("Analyzing...")
            state = st.session_state
            print(state.analyzed)
            video_url = state.video_url
            k = state.k

            if video_url:
                try:
                    video_obj = YouTubeVideo(video_url)
                    video_info = video_obj.get_info()
                    state.video_title = video_info['title']
                    state.video_author = video_info['author']
                    if state.video_author and state.video_title: state.flag_video_info = True

                    comment_obj = CommentInteractor(video_url)
                    comment_gen = comment_obj.get_comments(maxResults=state.num_cmt_per_batch)
                    
                    sentiment_dict = {
                        "NEG": 0,
                        "POS": 0,
                        "NEU": 0
                    }
                    counter_cmt = 1
                    
                    for comment_list in comment_gen:
                        analyzed_comments = []
                        n = len(comment_list)
                        if counter_cmt < state.num_batch + 1:
                            # Create a placeholder for the progress bar
                            bar = st.empty()  # Create an empty placeholder
                            notice = f"Analyzing comments in batch {counter_cmt}/{state.num_batch}"
                            progress_bar = bar.progress(0, text=notice)
                            
                            for index in range(n):
                                cmt = comment_list[index]['text']
                                tendency, neg, pos, neu = predict_sentiment(state.model_name, cmt, k)
                                sentiments = {"NEG": neg, "POS": pos, "NEU": neu}
                                label = max(sentiments, key=sentiments.get)
                                sentiment_dict[label] += 1
                                
                                # Update progress bar
                                progress_bar.progress(((index + 1) / n), text=notice)  
                                
                                if tendency > state.threshold:
                                    analyzed_comments.append({
                                        'comment': cmt,
                                        'neg': round(float(neg), 4),
                                        'pos': round(float(pos), 4),
                                        'neu': round(float(neu), 4),
                                        'tendency': round(float(tendency), 4)
                                    })
                        
                            # Clear the progress bar after completion
                            bar.empty()

                            # Update session variable 
                            state.sentiment_dict = sentiment_dict
                            state.analyzed_comments = sorted(analyzed_comments, key=lambda x: x['tendency'], reverse=True)
                            print(state.analyzed_comments)
                            state.analyzed = True  

                            counter_cmt += 1
                        else:
                            break
                    init_chatbot()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("An error occurred during URL processing!")

    # Display progress bar if needed
    if state.analyzed:
        st.success("Analysis complete!")

# AI Toggle
# ai_toggle = st.checkbox("Show AI Chatbot", value=False)

# Create a layout for the chatbot and the tabs side by side
col1, col2 = st.columns(2)  # Adjust the column widths as needed

# Tabs for Info and Comment
with col1:
    InfoTab, CommentTab = st.tabs(["Info", "Comment"])

    with CommentTab: 
        if state.analyzed:
            st.subheader("Comments with most tendency")
            plot_table(state.analyzed_comments)

    with InfoTab: 
        st.subheader("Summarize key insights")
        if not state.analyzed: 
            st.markdown("Enter URL to analyze")
        elif state.flag_video_info and state.analyzed: 
            st.write(f"Title: {state.video_title}")
            st.write(f"Author: {state.video_author}")

            df = pd.DataFrame(list(state.sentiment_dict.items()), columns=["Sentiment", "Value"])

            fig = px.bar(df, x="Sentiment", y="Value", text="Value", labels={"index": "Sentiment", "Value": "Count"})
            fig.update_traces(textposition='outside')
            fig.update_layout(
                title="Sentiment Analysis",
                xaxis_title="Sentiment",
                yaxis_title="Value",
            )

            st.plotly_chart(fig)



# Chatbot section
if state.ai_toggle:
    if not state.init_chatbot:
        init_chatbot()
    with col2:
        if state.init_chatbot:
            st.subheader("Ask the AI about YouTube strategies, answer will be derived from book: The Youtube Formula by DERRAL EVES")
            # user_question = st.text_input("Ask a question about YouTube strategies:")
            if "analyzed_comments" not in state: 
                comment = []
            else: 
                comment = state.analyzed_comments
            if "video_author" not in state: 
                author = ""
            else: 
                author = state.video_author
            if "video_title" not in state: 
                title = ""
            else: 
                title = state.video_title
            
            # if user_question:
            #     streaming_response = chat_engine.stream_chat(prompt)
            #     response_placeholder = st.empty()  
            #     bot_response = "Bot: "

            #     for token in streaming_response.response_gen:
            #         bot_response += token
            #         response_placeholder.write(bot_response)
            if prompt := st.chat_input("Ask a question about YouTube strategies"):
                
                prompt_t = state.qa_template.format(question_str = prompt, comment=comment, author=author, title=title)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    def stream(): 
                        chat_engine = state.index.as_chat_engine()
                        streaming_response = chat_engine.stream_chat(prompt)
                        for token in streaming_response.response_gen: 
                            yield token
                    streaming = stream()
                    response = st.write_stream(streaming)
                state.messages.append({"role": "assistant", "content": response})


