
# GlobalDiplomacyNet 

## Overview
This repository contains the reproduction code for the paper [Quantifying Global Foreign Affairs with a Multimodal Dataset of Diplomatic Websites](https://www.globaldiplomacy.net) by Nihat Mugurtay, Kaan Guray Sirin , Mehrdad Heshmat Najafabad, Ahmet Taha Kahya, Fazli Goktug Yilmaz , Yasser Zouzou, Batuhan Bahceci , Ayca Demir , Dogukan Tosun, Meltem Müftüler-Baç, Onur Varol.

## Reference to Dataset
Detailed description about the data can be found on [Harvard Dataverse](https://doi.org/10.7910/DVN/OFN15B). Please refer to the dataset’s README or the journal paper for any details regarding data fields, folder structures or the content.

## Repository Structure
This repository contains three main folders:
- **`figures/`**: Contains the code and Jupyter Notebooks for reproducing the figures presented in the paper. Figure subfolders may also contain external data.
- **`Sample-WebScraping/`**:  Includes sample scraper and parser scripts for demonstrating the data collection process of the dataset. The samples cover three approaches: _dynamic_ webpages, _static_ webpages, and webpages requiring a _proxy_.
- **`statistics/`**: Contains summary statistics of the dataset and the code used to generate it.

## Setup Instructions
Required Python libraries are listed in `requirements.txt`. Since the dependencies are very standard, most users’ existing Python environments should already have the required packages installed.

To set up a virtual environment and install the required libraries:
```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
pip install -r requirements.txt
```

## Contributors
_(in no particular order)_
- Nihat Mugurtay
- Kaan Guray Sirin 
- Mehrdad Heshmat Najafabad
- Ahmet Taha Kahya
- Fazli Goktug Yilmaz 
- Onur Varol

## Citation
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  url={}
}
```

## Acknowledgments
This work is supported by TUBITAK under the grant agreement 223K173. We also thank TUBITAK 121C220 for their partial support.
