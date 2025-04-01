import os
import uuid
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

from zenrows import ZenRowsClient

from config import LINK_LIST_PATH, ZENROWS_KEY


# Encoding for writing the URLs to the .txt file
# Do not change unless you are getting a UnicodeEncodeError
ENCODING = "utf-8"


def initialize_link_list():
    """
    Initialize the link list file with headers if it does not exist or is empty.
    """
    if not os.path.exists(LINK_LIST_PATH) or os.path.getsize(LINK_LIST_PATH) == 0:
        with open(LINK_LIST_PATH, "w", encoding=ENCODING) as f:
            f.write("\t".join(["id", "url", "page"]) + "\n")

def save_link(url, page):
    """
    Save collected link/url and page to the .txt file in LINK_LIST_PATH
    """
    id_str = uuid.uuid3(uuid.NAMESPACE_URL, url).hex
    with open(LINK_LIST_PATH, "a", encoding=ENCODING) as f:
        f.write("\t".join([id_str, url, str(page)]) + "\n")

def download_links_from_index():
    """
    This function should go to the defined "url" and download the news page links from all
    pages and save them into a .txt file.
    """
    # Initialize the link list file if needed
    initialize_link_list()

    # Get the page to start from
    data = pd.read_csv(LINK_LIST_PATH, sep="\t")
    if data.shape[0] == 0:
        start_page = 1
        downloaded_url_list = []
    else:
        start_page = data["page"].astype("int").max() + 1
        downloaded_url_list = data["url"].to_list()

    # Initialize the proxy client
    rootURL = "https://mfa.gov.gh/index.php/category/news/page/{}/"
    client = ZenRowsClient(ZENROWS_KEY)

    # Start downloading from the page "start_page"
    for pid in range(start_page, 32):

        pageURL = rootURL.format(pid)
        print(pageURL)

        try:

            resp = client.get(pageURL)
            soup = bs(resp.text, "html.parser")
            h2_tags = soup.find_all("h2", class_="title-post entry-title")

            for idx, h2 in enumerate(h2_tags,1):

                collected_url = h2.find("a")["href"]

                if collected_url not in downloaded_url_list:
                    print(f"\t{pid}.{idx}\t{collected_url}")
                    save_link(collected_url, pid)
                    

        except requests.RequestException as e:
            print(f"Error fetching page {pageURL}: {e}")
            continue
        except Exception as e:
            print(f"Error processing page {pageURL}: {e}")
            continue

if __name__ == "__main__":
    download_links_from_index()
