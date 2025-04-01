import os, sys, glob, re
import json
from pprint import pprint

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from config import RAW_HTML_DIR, PARSED_HTML_PATH


# Encoding for writing the parsed data to JSONS file
# Do not change unless you are getting a UnicodeEncodeError
ENCODING = "utf-8"


def extract_content_from_page(file_path):
    """
    This function takes as input the path to one html file
    and returns a dictionary "parsed_data" with the following information:

    parsed_data = {
        "date": Date of the news on the html page
        "title": Title of the news on the html page
        "content": The text content of the html page
    }

    This function is used in the parse_html_pages() function.
    You do not need to modify anything in that function.
    """
    parsed_data = {}

    ##################################
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = bs(file.read(), 'html.parser')
    
    # Get the date from <time> tag
    time_tag = soup.find("time", class_=["entry-date"])
    parsed_data["date"] = time_tag["datetime"] if time_tag else ""
    
    # Get the title from the <h1> tag
    title_tag = soup.find("h1", class_="title-post entry-title")
    if title_tag:
        parsed_data["title"] = title_tag.get_text(strip=True).split('&#8211;')[0].strip()
    else:
        parsed_data["title"] = ""
    
    # Get the content
    content_div = soup.find("div", class_="entry-content")
    if content_div:
        content_text = content_div.get_text(separator=" ", strip=True)
        # Clean the content text to remove unwanted JSON-like strings
        cleaned_content = re.sub(r'\{.*?\}', '', content_text)
        parsed_data["content"] = cleaned_content.strip()
    else:
        parsed_data["content"] = ""
    ##################################

    return parsed_data

def parse_html_pages():
    # Load the parsed pages
    parsed_id_list = []
    if os.path.exists(PARSED_HTML_PATH):
        with open(PARSED_HTML_PATH, "r", encoding=ENCODING) as f:
            # Saving the parsed ids to avoid reparsing them
            for line in f:
                data = json.loads(line.strip())
                id_str = data["id"]
                parsed_id_list.append(id_str)
    else:
        with open(PARSED_HTML_PATH, "w", encoding=ENCODING) as f:
            pass

    # Iterating through html files
    for file_name in os.listdir(RAW_HTML_DIR):
        
        page_id = file_name[:-5]

        # Skip if already parsed
        if page_id in parsed_id_list:
            continue

        # Read the html file and extract the required information

        # Path to the html file
        file_path = os.path.join(RAW_HTML_DIR, file_name)

        try:
            parsed_data = extract_content_from_page(file_path)
            parsed_data["id"] = page_id
            print(f"Parsed page {page_id}")

            # Saving the parsed data
            with open(PARSED_HTML_PATH, "a", encoding=ENCODING) as f:
                f.write("{}\n".format(json.dumps(parsed_data)))

        except Exception as e:
            print(f"Failed to parse page {page_id}: {e}")

if __name__ == "__main__":
    parse_html_pages()
    