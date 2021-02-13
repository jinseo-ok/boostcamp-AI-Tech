import os
import sys

import requests
import json
from bs4 import BeautifulSoup
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from PIL import Image

# -------- 네이버 웹툰 카테고리 목록 크롤링 -------- #
def getWebtoons():
    url = 'https://comic.naver.com/webtoon/genre.nhn?genre=episode'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.select('div.snb > ul.spot > li')
    categories = [elem.text.strip() for elem in categories]

    return categories


# -------- 네이버 웹툰 카테고리별 웹툰 대표 썸네일 크롤링 -------- #
def getRepThumb(category):
    
    url = f'https://comic.naver.com/webtoon/genre.nhn?genre={category}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.select('div.list_area > ul > li')
    
    category_endpoints = []
    for item in items:
        endpoint = 'https://comic.naver.com'
        web_url = endpoint + item.select_one('div.thumb > a')['href']
        web_title = item.select_one('div.thumb > a')['title']
        web_thumb = item.select_one('div.thumb > a > img')['src']
        
        DATA_PATH = 'thumbnails'
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        
        FILE_PATH = os.path.join(DATA_PATH, category)
        if not os.path.exists(FILE_PATH):
            os.makedirs(FILE_PATH)
        
        FILE_NAME = FILE_PATH + f'/{web_title}.jpg'
        urlretrieve(web_thumb, FILE_NAME)
        category_endpoints.append([category, web_title, web_url])
        
    return category_endpoints


# -------- 네이버 웹툰별 최근 20화 썸네일 크롤링 -------- #
def getdetailThumbs(url):
    thumb_list = []
    for i in range(1, 3): # 1, 2페이지까지 (최대 20화)
        try:
            url = url + f"&page={i}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.select('tr > td > a > img')
        
        
            for item in items:
                if item['alt'] == 'AD 배너':
                    continue

                thumb_list.append(item['src'])
        except:
            pass
            
    return thumb_list


if __name__ == "__main__":
    print("Webtoon category crawling....")
    categories = getWebtoons()
    
    print("Webtoon Thumbnail crawling....")
    web_urls = []
    for category in tqdm(categories):
        category_urls = getRepThumb(category)
        web_urls.extend(category_urls)
    
    
    webtoon = pd.DataFrame(web_urls, columns = ['category', 'title', 'url'])
    
    print("Webtoon detail Thumbnail crawling....")
    for category, title, url in tqdm(webtoon.values):
    
        thumb_list = getdetailThumbs(url)

        for i, thumb in enumerate(thumb_list, start = 1):
            FILE_NAME = os.path.join('thumbnails', category, f'{title}_{i}.jpg')
            urlretrieve(thumb, FILE_NAME)
    print("Complete Webtoon detail Thumbnail crawling!")
    
    
    file_num = 0
    categories = os.listdir('./thumbnails')

    for category in categories:
        FILE_PATH = os.path.join('thumbnails', category)
        file_num += len(os.listdir(FILE_PATH))
        
    print(f'Total Webtoon categories: {len(categories)}')
#     print(f'Total Webtoons: {webtoon['title'].nunique()}')
    print(f'Total Data: {file_num}')