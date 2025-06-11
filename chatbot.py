## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


groq_api_key=st.secrets["GROQ_API_KEY"]
valid_users = st.secrets["auth"]["users"]


def verify_login ():   
        if username in valid_users and password == valid_users[username]:
            st.session_state["authenticated"] = True
            st.success("Login successful!")
        else:   
            st.error("Please enter correct credentials to proceed")

## set up Streamlit 
st.title("PDF Q/A Chatbot")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

## Input the UserName and Password 
if st.session_state["authenticated"] == False :
    username = st.text_input("Enter your Username")
    password = st.text_input("Enter your Password", type="password")
    if st.button("Login"):
        ## Check if credentials are entered
        if username and password :
            verify_login()
            
            
if st.session_state["authenticated"] == True :
        st.write("Welcome to PDF Chatbot, where you can upload pdf and ask questions on it")
                
        llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")

        ## chat interface
        session_id= st.text_input("Enter your session id")

        ## statefully manage chat history
        if 'store' not in st.session_state:
            st.session_state.store={}

        uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
        print("file uploaded succesfully")
                
        ## Process uploaded  PDF's
        if uploaded_files :
            if "finalDocuments" not in st.session_state.store:
                documents=[]
                for uploaded_file in uploaded_files:
                    temppdf=f"./temp.pdf"
                    with open(temppdf,"wb") as file:
                        file.write(uploaded_file.getvalue())
                        file_name=uploaded_file.name

                    loader=PyPDFLoader(temppdf)
                    docs=loader.load()
                    documents.extend(docs)

                st.session_state.store["finalDocuments"] = documents
                print("documents ingested succesfully.")

            # Split and create embeddings for the documents
            if "chunks" not in st.session_state.store:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                    st.session_state.store["chunks"] = text_splitter.split_documents(st.session_state.store["finalDocuments"])
                    print("chunking done succesfully")
                        
            if "vectorstore" not in st.session_state.store:
                    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.store["vectorstore"] = FAISS.from_documents(documents=st.session_state.store["chunks"], embedding=embeddings) 
                    print("vector store ready succesfully")
                    
            retriever = st.session_state.store["vectorstore"].as_retriever() 

            history_prompt_message=(
                        "Rephrase the latest user question into a standalone one using the chat history."
                        " Do not answer it—only rewrite if needed"
                    )
            history_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", history_prompt_message),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
                    
            history_aware_retriever=create_history_aware_retriever(llm,retriever,history_prompt)

            ## Answer question
            qa_message = (
                            "Answer the question using the retrieved context. "
                            "If unsure, say you don't know. Keep your response concise, max three sentences"
                            "{context}"
                        )
            qa_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", qa_message),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
                    
            question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
            rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

            def get_session_history(session:str)->BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id]=ChatMessageHistory()
                    return st.session_state.store[session_id]
                    
            conversational_rag_chain=RunnableWithMessageHistory(
                        rag_chain,get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer"
                    )

            user_input = st.text_input("Ask your question:")
            if user_input:
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={
                                "configurable": {"session_id":session_id}
                            },  # constructs a key "abc123" in `store`.
                        )
                #st.write(st.session_state.store)
                st.write("Assistant:", response['answer'])
                #st.write("Chat History:", session_history.messages)










