import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# load env variables
load_dotenv()
my_openai_key = os.getenv("OPENAI_API_KEY")
my_pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = "rag-chatbot"

# connect to pinecone
pc = Pinecone(api_key=my_pinecone_key)

# check if index is there, otherwise create a new one
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# read data from text file
with open("data.txt", "r") as file:
    raw_text = file.read()

# split text into small pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

my_document = Document(page_content=raw_text)
my_split_docs = splitter.split_documents([my_document])

# setup embeddings to convert text to numbers
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=my_openai_key
)

# save all pieces to pinecone database
vector_db = PineconeVectorStore(
    index_name=index_name,
    embedding=embed_model,
    pinecone_api_key=my_pinecone_key
)
vector_db.add_documents(my_split_docs)

# setup retriever to search for answers
my_retriever = vector_db.as_retriever(search_kwargs={"k": 3}) #top 3 relevent chunks 

# setup chat model
my_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=my_openai_key
)

# system prompt config
my_prompt_text = """
You are a helpful Study Assistant. Answer the student's question only from the given study notes context.
Explain the concepts clearly and simply.
If the answer is not in the study notes, say: "Sorry, ye topic mere notes mein nahi hai."

Context:
{context}

Question:
{input}
"""
prompt = ChatPromptTemplate.from_template(my_prompt_text)

# join the chains together
doc_chain = create_stuff_documents_chain(llm=my_llm, prompt=prompt)
my_rag_chain = create_retrieval_chain(retriever=my_retriever, combine_docs_chain=doc_chain)

# chat loop starts here
while True:
    user_input = input("you: ")

    if user_input.lower() == "exit":
        print("chat ended.")
        break

    # get answer from our chain
    input_dict = {"input": user_input}
    ans = my_rag_chain.invoke(input_dict)
    
    #  final output
    print("bot:", ans["answer"])