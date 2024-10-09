import time
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import pandas as pd
import random
from typing import List, Dict

# Инициализируем сессию и фейковый User-Agent для краулера
ua = UserAgent()
session = requests.session()

def get_user_agent():
    """Получает корректный User-Agent."""
    user_agent = ua.random
    # Проверка на корректность
    if user_agent and not any(c in user_agent for c in ['\r', '\n', ' ']):
        return user_agent
    else:
        # Используем статический User-Agent в случае проблемы. Это решение я нашла на просторах интернета.
        # В какой-то момент при выкачивании песен появилась проблема с неверным user-agent
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'


def parse_one_song(link: str) -> Dict[str, str]:
    """
       Парсит страницу с песней, извлекает автора, название и текст песни. Вся информация, что есть на этом сайте

       :param link: Ссылка на страницу с песней.
       :return: Словарь с автором, названием и полным текстом песни.
    """
    block = {}

    # Запрос страницы
    url_one = link
    req = session.get(url_one, headers={'User-Agent':  get_user_agent()})
    page = req.text
    soup = BeautifulSoup(page, 'html.parser')

    # Извлечение автора песни
    author_all = soup.find_all('a', {'itemprop': 'item'})
    author = author_all[1].find_all('span', {'itemprop': 'name'})[0].text
    block['author'] = author

    # Извлечение названия песни
    title_all = soup.find_all('div', {'class': 'container'})
    title = title_all[2].find('h1').text
    block['title'] = title.split(' — ')[0]

    # Извлечение текста песни
    lyrics_with_tags = soup.find_all('div', {'class': 'main_text-slova'})
    lyrics_text = lyrics_with_tags[0].find_all('p')
    lyrics = ''
    for element in lyrics_text:
        # Удаление HTML-тегов
        element = str(element).replace('<p>', '').replace('</p>', '')
        if element.lower() != '': # Проверка на пустую строку
            lyrics += element + '\n'
    block['full_text'] = lyrics

    return block


def get_nth_page(page_number: int) -> List[Dict[str, str]]:
    """
        Парсит список песен с указанной страницы, собирает ссылки на песни и осуществляется переход на  парсинг
        одной страницы

        :param page_number: Номер страницы, которую необходимо спарсить.
        :return: Список словарей с данными о песнях (автор, название, текст).
    """
    all_href = []
    url = f'https://text-pesenok.ru/?page={page_number}/'

    # Запрос страницы со списком песен
    req = session.get(url, headers={'User-Agent':  get_user_agent()})
    page = req.text
    soup = BeautifulSoup(page, 'html.parser')

    # Извлечение всех ссылок на песни
    elements = soup.find_all('div', class_='item')
    for item in elements:
      links = item.find_all('a')
      for link in links:
          href = link.get('href')
          all_href.append(href)

    blocks = []
    for n in tqdm(all_href):
        try:
            # Парсинг каждой песни
            blocks.append(parse_one_song('https://text-pesenok.ru'+ n))
        except Exception as e:
            # Перечень ошибок
            with open('Errors.txt', 'a', encoding='utf-8') as f:
                    error = ' '.join([str(e), 'Ссылка на страницу:', url, '\n'])
                    f.write(error)
        time.sleep(random.uniform(1.1, 10.2))
    return blocks


def run_all(n_pages: int) -> List[Dict[str, str]]:
    """
        Запускает процесс парсинга указанного количества страниц с песнями.

        :param n_pages: Количество страниц для парсинга.
        :return: Список словарей с данными о всех песнях.
    """
    blocks_run = []
    for i in tqdm(range(n_pages)):
        # Парсинг страниц по порядку
        blocks_run.extend(get_nth_page(i + 1))
        time.sleep(random.uniform(1.1, 10.2))
    return blocks_run


# Запустим, переведём в дф и сохраним пока в csv
blocks = run_all(30)
df = pd.DataFrame(blocks)
df.to_csv('corpus.csv', index=False)
