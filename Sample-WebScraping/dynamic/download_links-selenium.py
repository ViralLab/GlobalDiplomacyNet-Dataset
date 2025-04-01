import time
import random
from bs4 import BeautifulSoup as bs
import requests

import os, sys, glob, re
import json
from pprint import pprint

import pandas as pd
import numpy as np
import uuid

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

from config import LINK_LIST_PATH

ENCODING = "utf-8"

options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)

def save_link(url, page):

    id_str = uuid.uuid3(uuid.NAMESPACE_URL, url).hex
    with open(LINK_LIST_PATH, "a", encoding=ENCODING) as f:
        f.write("\t".join([id_str, url, str(page)]) + "\n")


def download_links_from_index():

    if not os.path.exists(LINK_LIST_PATH):
        with open(LINK_LIST_PATH, "w", encoding=ENCODING) as f:
            f.write("\t".join(["id", "url", "page"]) + "\n")
        downloaded_url_list = []
        page_list = []

    # If some links have already been downloaded,
    # get the downloaded links and start page
    else:
        # Get the page to start from
        data = pd.read_csv(LINK_LIST_PATH, sep="\t")
        if data.shape[0] == 0:
            downloaded_url_list = []
        else:
            page_list = data["page"].to_list()
            downloaded_url_list = data["url"].to_list()

    rootURL= 'https://mofa.gov.qa/en/all-mofa-news'
    driver.get(rootURL)
    pid=1

    while True:

        try:    
            print("Page", pid)

            soup = bs(driver.page_source, 'lxml')
            articles=soup.find_all("div", {'class':'col-12 col-md-6 news-list-item'})[-6:]

            for idx, item in enumerate(articles, 1):
                collected_url= 'https://mofa.gov.qa' + item.find('a')['href']

                if collected_url not in downloaded_url_list:
                    print(f"\t{pid}.{idx}\t", collected_url, flush=True)
                    save_link(collected_url, pid)
                    downloaded_url_list.append(collected_url)

            load_more_button = driver.find_element(By.ID, "load-more-inner")
            load_more_button.click()
            time.sleep(1)
            pid+=1


        except KeyboardInterrupt:
            return
        
        except Exception as e:
            print(f"Error {pid:3}:\t{e}")



if __name__ == "__main__":
    download_links_from_index()
    driver.quit()