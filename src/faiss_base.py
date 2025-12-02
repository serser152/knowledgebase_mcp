import os
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PDFMinerLoader



class KnowledgeBase:
    def __init__(self, embedding_model_name = "all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=100):
        self.index = None
        self.txt = []
        self.embeddings = None
        self.model = SentenceTransformer(embedding_model_name)

        # define text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""])

    def set_Embedding_Model(self, model_name):
        if self.model is None:
            self.model = SentenceTransformer(model_name)
        else:
            print('Model already set')
        return self


    def get_text_chunks_from_file(self, file_path):
        """Загрузка текста из файла"""
        # define loader
        loader = None
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PDFMinerLoader(file_path)
        else:
            return []
        if loader is None:
            return []

        # load and split
        r = loader.load_and_split(self.text_splitter)

        return [{'txt': d.page_content} for d in r]

    def load_docs(self):
        """Загрузка документов из папки docs"""
        for doc_file in os.listdir('./docs'):
            if doc_file.endswith('.txt') or doc_file.endswith('.pdf'):
                self.txt.extend(self.get_text_chunks_from_file('./docs/' + doc_file))
        self.create_index()

    def create_index(self):
        """Создание индекса Faiss"""
        docs_embeddings = self.model.encode(self.txt)
        d = docs_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(docs_embeddings)  # Добавляем эмбеддинги
        
    def find_similar(self, query, k=3):
        """Поиск похожих документов"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        print(distances)
        print(indices)
        return [{'txt':self.txt[ii]['txt'], 'distance':distances[0][i]} for i,ii in enumerate(indices[0])]



#print(k.txt)
#
# txt = []
#
#
#
#     if file.endswith('.txt') or file.endswith('.pdf'):
#         txt.extend(get_text_chunks_from_file('./planner_mcp/docs/' + file))
# # загрузить все файлы из каталога docs и получить тексты
# for file in os.listdir('./docs'):
#     if file.endswith('.txt') or file.endswith('.pdf'):
#         txt.extend(get_text_chunks_from_file('./planner_mcp/docs/' + file))
#
#
# splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=100)
#
# # Модель для получения эмбеддингов
# print('Loading model...')
# model = SentenceTransformer('all-MiniLM-L6-v2')
# 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# z
# # Получаем эмбеддинги (размерность 384 у модели 'all-MiniLM-L6-v2')
# print('Encoding...')
# embeddings = model.encode(txt)
# d = embeddings.shape[1]
# print("d=",d)
# # Создаём индекс Faiss (L2 расстояние)
# print('Creating index...')
# index = faiss.IndexFlatL2(d)
#
# index.add(embeddings)  # Добавляем эмбеддинги
# i2 = faiss.IndexIVFFlat(index, d, 3)
# i2.train(embeddings)
# i2.nprobe=10
#
# # Запрос
#
# query = "Какие котики бывают?"
# query_embedding = model.encode([query])
#
# # Поиск (например, 3 ближайших)
# print('Searching...')
# k = 7
# distances, indices = index.search(query_embedding, k)
# print(distances)
# print(indices)
# # Вывод результатов
# print("Запрос:", query)
# print("Найденные тексты:")
# for t,i in enumerate(indices[0]):
#     print(f"{distances[0][t]}- {txt[i]}")
