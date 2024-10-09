import re
import string
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Инициализация лемматизатора
morph = pymorphy2.MorphAnalyzer()

# Загрузка стоп-слов для русского языка
stop_words = set(stopwords.words('russian'))

def preprocess_for_tfidf(text: str) -> str:
    """
    Предобрабатывает текст для индексации с помощью TF-IDF:
    приводит к нижнему регистру, удаляет пунктуацию, стоп-слова, выполняет лемматизацию.
    Происходят именно эти шаги, поскольку:
    стоп-слова — достаточно частотные вещи, они в принципе часто встречаются, будут присутствовать
    почти в каждом документе. Нам же нужно выделить ключевые слова.
    пунктуация — мы считаем слова, пунктуация нам ни к чему.
    лемматизация — мы считаем вхождение слова => форма должна быть одинаковой для подсчёта.

    :param text: Исходный текст
    :return: Предобработанный текст
    """
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление пунктуации
    text = re.sub(f'[{string.punctuation}]', ' ', text)

    # Токенизация текста
    words = word_tokenize(text, language='russian')

    # Лемматизация и удаление стоп-слов
    words = [morph.parse(word)[0].normal_form for word in words if word not in stop_words]

    return ' '.join(words)


def preprocess_for_bert(text: str) -> str:
    """
    Минимальная предобработка для BERT — убирает лишние пробелы. Для berta важна пунктуация и слова такие,
    какие они есть, поскольку он обучается на контексте. Соответственно, стоп-слова также могут полезны для обучения.

    :param text: Исходный текст
    :return: Предобработанный текст
    """
    return text.strip()  # Просто убираем лишние пробелы