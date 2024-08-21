import streamlit as st
import tiktoken # 텍스트를 여러 개의 청크로 나눌 때 token 개수를 세기 위한 라이브러리
from loguru import logger # 구동한 것이 log로 남도록 하기 위한 라이브러리

# 대화 메모리를 가지고 있는 chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# 데이터 로드
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader # 워드파일
from langchain.document_loaders import UnstructuredPowerPointLoader

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 임베딩
from langchain.embeddings import HuggingFaceEmbeddings

# 몇 개까지의 대화를 메모리에 넣어줄 것인가? - ConversationBufferMomory(ConversationalRetrievalChain과 관련)
from langchain.memory import ConversationBufferMemory

# vector store, FAISS : 임시로 벡터 저장
from langchain.vectorstores import FAISS

# 메모리를 구현하기 위해서 추가적으로 필요한 라이브러리 호출
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title = "MedicalChat",
    page_icon = ":hospital:")
    
    st.title("_Medical :red[QA Chat]_ :hospital:")
    
    # conversation과 chat_history를 session state에서 활용하기 위해 미리 설정
    if "conversation" not in st.session_state :
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state :
        st.session_state.chat_history = None
     
    # 상위 구문에 하위 구문을 넣어야 할 때 with    
    with st.sidebar :
        uploaded_files = st.file_uploader("Upload your file", type = ['pdf', 'docx', 'pptx'], accept_multiple_files = True)
        openai_api_key = st.text_input("OpenAI API Key", key = "chatbot_api_key", type = "password")
        process = st.button("Process")
        
    if process :
        if not openai_api_key :
            st.info("Please aid your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files) # 업로드 파일을 텍스트로 변환
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        
    if 'messages' not in st.session_state :
            st.session_state['messages'] = [{"role" : "assistant",
                                             "content" : "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    
    # 계속 반복        
    for message in st.session_state.messages :
            with st.chat_message(message["role"]) :
                st.markdown(message["content"])
    
    # 메모리를 가지고 컨텍스트를 고려해서 답변하기 위해 
    history = StreamlitChatMessageHistory(key = "chat_messages")
    
    # Chat logic
    
    # 사용자가 입력을 하면
    if query := st.chat_input("질문을 입력해주세요.") :
        
        # 사용자가 질문한 것을 기록
        st.session_state.messages.append({"role" : "user", "content" : query})
        
        with st.chat_message("user") :
            # 쿼리라고 인식
            st.markdown(query)
            
        with st.chat_message("assistant") :
            
            chain = st.session_state.conversation
            
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
        
        # 어시스턴트가 답변한 것도 함께 기록
        st.session_state.messages.append({"role": "assistant", "content": response})

# 토큰 개수를 기준으로 텍스트를 splitting        
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens) 

def get_text(docs):

    # 여러 개의 파일
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
            
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
        
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

if __name__ == '__main__':
    main()      
