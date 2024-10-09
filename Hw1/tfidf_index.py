import pickle # Для сохранения и загрузки индексов
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


class TFIDFIndexer:
    def __init__(self, save_path: str):
        """
        Класс для работы с TF-IDF индексом.

        :param save_path: Путь до файла, где сохраняется индекс
        """
        self.vectorizer = TfidfVectorizer()
        self.save_path = save_path
        self.index = None # Здесь будет храниться матрица TF-IDF

    def build_index(self, texts: List[str]):
        """
        Строит TF-IDF для списка текстов и сохраняет его. Сохраняет также векторайзер.

        :param texts: Список текстов для индексации
        """
        self.index = self.vectorizer.fit_transform(texts)
        with open(self.save_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.index), f)

    def load_index(self):
        """Загружает индекс и векторайзер, который был сохранён."""
        with open(self.save_path, 'rb') as f:
            self.index, self.vectorizer = pickle.load(f)

    def search(self, query: str, top_n: int = 5) -> List[int]:
        """
        Выполняет поиск по запросу и возвращает индексы наиболее релевантных документов.

        :param query: Текст запроса
        :param top_n: Количество топ результатов
        :return: Список индексов документов
        """
        query_tfidf = self.vectorizer.transform([query])
        scores = (self.index * query_tfidf.T).toarray().flatten()
        ranked_indices = scores.argsort()[::-1][:top_n]
        return ranked_indices
