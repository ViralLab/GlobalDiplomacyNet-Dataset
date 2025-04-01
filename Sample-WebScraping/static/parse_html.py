import os, sys, glob, re
import json
from pprint import pprint

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from config import RAW_HTML_DIR, PARSED_HTML_PATH, LINK_LIST_PATH

from datetime import datetime
import ast

# Encoding for writing the parsed data to JSONS file
# Do not change unless you are getting a UnicodeEncodeError
ENCODING = "utf-8"



def extract_content_from_page(file_path):

    parsed_data = {}

    ##################################
    soup = bs(open(file_path, 'r').read(),'html')
    
    parsed_data['date']=soup.find("div",{"class":"col-sm-6"}).text.split()[0]
    parsed_data['title']=soup.find('p', {"class":"press_release_title_inner"}).text.strip()
    parsed_data['content'] = soup.find("p",{"class":"press_release_title_inner"}).find_next('div').text
    
    ##################################
    

    return parsed_data


def parse_html_pages():
    # Load the parsed pages
    url_df = pd.read_csv(LINK_LIST_PATH, sep="\t")

    parsed_id_list = []
    if os.path.exists(PARSED_HTML_PATH):
        with open(PARSED_HTML_PATH, "r", encoding=ENCODING) as f:
            for line in f:
                parsed_id_list.append(json.loads(line)["id"])
    else:
        with open(PARSED_HTML_PATH, "w", encoding=ENCODING) as f:    
            pass



    # Iterating through html files
    for i, file_name in enumerate(os.listdir(RAW_HTML_DIR),start=1):
        page_id = file_name[:-5]

        # Skip if already parsed
        if page_id in parsed_id_list:
            continue

        # Path to the html file
        file_path = os.path.join(RAW_HTML_DIR, file_name)

        try:
            parsed_data = extract_content_from_page(file_path)
            parsed_data["id"] = page_id
            percentage="%.2f"%((i)/len(os.listdir(RAW_HTML_DIR))*100)

            print(f"Parsed page {page_id}\t{percentage}%")
            
            # Saving the parsed data
            with open(PARSED_HTML_PATH, "a", encoding=ENCODING) as f:
                f.write("{}\n".format(json.dumps(parsed_data, ensure_ascii=False)))

        except KeyboardInterrupt:
            return 
        
        except Exception as e:
            print(f"Failed to parse page {page_id}: {e}")
            pass
        

        
        


if __name__ == "__main__":
    parse_html_pages()
