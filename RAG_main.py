from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openllm import OpenLLM
from llama_index.llms.google_genai import GoogleGenAI
import os

# 通过dotenv加载模型api-key
from dotenv import load_dotenv
load_dotenv()

# 获取DeepSeek API密钥
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# 检查API密钥是否存在
if not deepseek_api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

llm_deepseek = OpenLLM(
    model="deepseek-reasoner", api_base="https://api.deepseek.com/v1", api_key=deepseek_api_key
)

google_api_key = os.getenv("GOOGLE_API_KEY")
llm_google = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,  # uses GOOGLE_API_KEY env var by default
)

# 初始化 Ollama 嵌入模型
ollama_embedding = OllamaEmbedding(
    model_name="bge-m3",  # 替换为实际的模型名称，如 mxbai-embed-large
    base_url="http://localhost:11434",  # Ollama 服务地址
    ollama_additional_kwargs={"mirostat": 0},  # 可选的额外参数
)



def setup_rag_system(pdf_directory="./pdfs"):
    Settings.llm = llm_google
    Settings.embed_model = ollama_embedding
    
    # 加载PDF文档
    documents = SimpleDirectoryReader(pdf_directory).load_data()
    
    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 创建查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    return query_engine

def query_pdf(query_engine, question):
    # 执行查询
    response = query_engine.query(question)
    return response

def main():
    # 创建存储PDF的目录
    if not os.path.exists("./pdfs"):
        os.makedirs("./pdfs")
        print("请将PDF文件放入./pdfs目录中")
        return
    
    # 初始化RAG系统
    print("正在初始化RAG系统...")
    query_engine = setup_rag_system()
    
    # 示例问答循环
    while True:
        question = input("\n请输入您的问题（输入'退出'结束）：")
        if question.strip().lower() == "退出":
            break
            
        response = query_pdf(query_engine, question)
        print("\n回答：", response)

if __name__ == "__main__":
    main()