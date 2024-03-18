import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to load data and answer queries
def answer_query(data, query):
    # Convert DataFrame to string
    data_str = data.astype(str)
    # Create TAPAS pipeline
    answerer = pipeline("table-question-answering", model='google/tapas-base-finetuned-wtq')
    # Get answer based on the query
    answer = answerer(table=data_str, query=query)["answer"]
    return answer

# Main function
def main():
    st.title("Table-based Question Answering")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV data:")
        st.write(data)

        # Allow user to input query
        query = st.text_input("Enter your question from the data:")

        if st.button("Get Answer"):
            # Get answer based on the query
            if query:
                answer = answer_query(data, query)
                st.write("Answer:", answer)
            else:
                st.write("Please enter a query.")

if __name__ == "__main__":
    main()
