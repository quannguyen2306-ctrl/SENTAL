from llama_index.core import Settings
from llama_index.llms.openai import OpenAI 
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os 
from llama_index.core import PromptTemplate

ANSWER_TEMPLATE = ( 
    "According to the book: The YouTube Formula,"
    "What is the best solution for this problem:"
    "{question_str}?"
    " Answer like the author of the book."
)

Settings.llm = OpenAI(model = "gpt-4o-mini", temperature=0)

# Ingestion data
STORAGE_PATH = "docx"
documents = LlamaParse(result_type="markdown").load_data(
    "./data/pdfcoffee.com-the-youtube-formula-by-derral-eve.pdf"
)


# store ingrestion
PERSIST_DIR = "index_cache"

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

qa_template = PromptTemplate(ANSWER_TEMPLATE)
