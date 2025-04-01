import os, sys, glob, re
import json
from pprint import pprint

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np

import time

from config import LINK_LIST_PATH, RAW_HTML_DIR


# Encoding for writing the page html files
# Do not change unless you are getting a UnicodeEncodeError
ENCODING = "utf-8"




def save_html_pages():
   
    # Step 1: Read URL/Link list file from LINK_LIST_PATH
    #         to get the urls that need to be saved
    url_df = pd.read_csv(LINK_LIST_PATH, sep="\t")

    # Step 2: Checking the downloaded html page IDs
    html_list = os.listdir(RAW_HTML_DIR)
    id_list = list(map(lambda x: x[:-5], html_list))


    # Step 3: Iterating through the URL list
    for idx, row in url_df.iterrows():
        page_id = row["id"]
        page_url = row["url"]

        # Skip page if already downloaded
        if page_id in id_list:
            continue

        # Step 4: Loading page html
        while True:
            try:
                resp=requests.get(page_url)
                page_html = resp.text

                # Step 5: Saving page html
                save_path = os.path.join(RAW_HTML_DIR, f"{page_id}.html")
                    
                with open(save_path, "w", encoding=ENCODING) as f:
                    f.write(page_html)
                    percentage="%.3f"%((idx+1.0)/url_df.shape[0]*100)
                print(f"Saved page {page_id} ({idx+1} / {url_df.shape[0]}) \t{percentage}%")
            
            except KeyboardInterrupt:
                return
                
            except Exception:
                # Pages that were not collected are saved as empty strings
                page_html = ""
                print(f"Error getting page {page_id}")
            else:
                break



            


        

if __name__ == "__main__":
    save_html_pages()
