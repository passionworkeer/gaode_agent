
"""
RAG Engine 基础实现：
- 支持文档入库（embedding + FAISS 向量存储）
- 支持问题检索与召回
- 可与主Agent集成
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

# 定义本地模型路径
LOCAL_EMBEDDING_MODEL_PATH = r"D:\local_models\sentence-transformers_all-MiniLM-L6-v2"

class RAGEngine:
    def __init__(self, embedding_model_path=LOCAL_EMBEDDING_MODEL_PATH, persist_path=None):
        # 使用本地模型路径初始化 HuggingFaceEmbeddings
        # 注意：HuggingFaceEmbeddings 通常使用 model_name 参数
        # 为了使用本地路径，需要传递路径字符串
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.persist_path = persist_path or os.path.join(os.path.dirname(__file__), 'faiss_index')
        self.vectorstore = None

    def build_index(self, docs: list[str]):
        """将文档列表分割、embedding、构建FAISS索引"""
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        doc_objs = [Document(page_content=chunk) for doc in docs for chunk in splitter.split_text(doc)]
        self.vectorstore = FAISS.from_documents(doc_objs, self.embedder)
        self.vectorstore.save_local(self.persist_path)

    def load_index(self):
        """加载已保存的FAISS索引"""
        # 确保 embedder 在加载索引时也使用本地模型
        self.vectorstore = FAISS.load_local(self.persist_path, self.embedder)

    def query(self, question: str, top_k=3) -> str:
        """检索相关文档片段，并美化输出"""
        if self.vectorstore is None:
            self.load_index()
        # 兼容 langchain 新版参数名为 top_k
        try:
            docs = self.vectorstore.similarity_search(question, top_k=top_k) # type: ignore
        except TypeError:
            docs = self.vectorstore.similarity_search(question, k=top_k) # type: ignore
        if not docs:
            return "未检索到相关内容。"
        # 美化输出：编号+分段
        return "\n\n".join([f"【片段{i+1}】\n{d.page_content.strip()}" for i, d in enumerate(docs)])

