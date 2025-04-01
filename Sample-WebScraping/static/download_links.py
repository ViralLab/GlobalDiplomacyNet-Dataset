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


from config import LINK_LIST_PATH

ENCODING = "utf-8"


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

    rootURL= 'https://www.pio.gov.cy/en/press-releases/?keyword=&startdate=&enddate=&category=President+of+the+Republic+of+Cyprus&submitbtn=Search&page={}'
    
    for pid in range(1, 155): 
        if pid in page_list:
            continue

        pageURL= rootURL.format(pid)
        print("Page", pid)
        try:    

            resp = requests.get(pageURL, headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'})           
            time.sleep(random.uniform(0.3,1.2))

            soup = bs(resp.text, 'lxml')
            articles=soup.find_all("div", {'class':'row press_release_list_item'})


            for idx, item in enumerate(articles, 1):
                collected_url= "https://www.pio.gov.cy/en/"+item.find('a')['href']

                if collected_url not in downloaded_url_list:
                    print(f"\t{pid}.{idx}\t", collected_url, flush=True)
                    save_link(collected_url, pid)
                    downloaded_url_list.append(collected_url)
            
        except KeyboardInterrupt:
            return
        
        except Exception as e:
            print(f"Error {pid:3}:\t{e}")

    



if __name__ == "__main__":
    download_links_from_index()