import streamlit as st
import pandas as pd 

def plot_table(analyzed_comments):
    # Convert the list of comments to a DataFrame
    df = pd.DataFrame(analyzed_comments)

    # Draw table with all comments
    st.dataframe(df)

    # Optionally, you can display additional information if needed
    st.write(f"Total comments: {len(analyzed_comments)}")
