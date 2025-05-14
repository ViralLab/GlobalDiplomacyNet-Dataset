import os, re
import json, copy
from collections import Counter

from datetime import datetime
from langcodes import get as langcode_get
from typing import List, Dict, Any
from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec 
from matplotlib.colors import Colormap

import warnings

warnings.filterwarnings('ignore') # always good practice to do this :)

font_dict = {'size':12}
title_font= {'size':14}
plot_style= {'color':'grey'}

# 66 most diplomatically active countries according to https://globaldiplomacyindex.lowyinstitute.org
lowy_66 = ['CHN', 'USA', 'TUR', 'JPN', 'FRA', 'RUS', 'GBR', 'DEU', 'ITA', 'BRA', 'IND', 'ESP', 'KOR', 'MEX', 'CAN', 'ARG', 'NLD', 'HUN', 'POL', 'GRC', 'IDN', 'SAU', 'PRT', 'AUS', 'PAK', 'CHL', 'CZE', 'COL', 'ZAF', 'BEL', 'TWN', 'ISR', 'MYS', 'AUT', 'SWE', 'IRL', 'THA', 'VNM', 'PHL', 'NOR', 'FIN', 'DNK', 'BGD', 'NZL', 'LTU', 'SVN', 'CRI', 'MNG', 'SGP', 'LVA', 'EST', 'MMR', 'LUX', 'PRK', 'KHM', 'BRN', 'NPL', 'LAO', 'TLS', 'ISL', 'PNG', 'BTN', 'SUI', 'SVK', 'LKA']


### function I use to get absolute paths for country files
def get_countries(data_directory:str, include:List[str]|None=None, exclude:List[str]|None=None)->List[str]:
    # for replication no need to use the include and exclude parameters
    all_countries = list(map(lambda x: f'{data_directory}/{x}',os.listdir(data_directory)))
    if include: 
        all_countries = [i for i in all_countries if any(sub in i for sub in include)]
    if exclude: 
        all_countries = [i for i in all_countries if not any(sub in i for sub in exclude)]

    return all_countries

### getting the counts by dates
def get_country_date_series(country_directory:str)->List[str]:
    """Parses the country's jsonl file into a list of dates"""
    country_date_list=[json.loads(i)['date'] for i in open(f'{country_directory}/news.jsonl','r').read().split('\n')[:-1] if json.loads(i)['date']]
    return country_date_list

def group_date_series(dates_list:List[str])->Dict[str,int]:
    """Groups the date list into years"""
    return dict(sorted(Counter(map(lambda x: datetime.strptime(x,'%Y/%m/%d').strftime('%Y'), dates_list)).items()))


def fill_year_dict(count_dict:dict, start:int, end:int=2024)->Dict[str,int]:
    for y in range(start,end+1):
        if str(y) not in count_dict:
            count_dict[str(y)]=0
    # removing outside the range
    count_dict= {k:v for k,v in count_dict.items() if str(start)<=k and k<=str(end)}

    return dict(sorted(count_dict.items()))

def merge_and_sum(dict1:Dict[str,Any], dict2:Dict[str,Any])->Dict[str,Any]:
    return dict(sorted({key: dict1.get(key, 0) + dict2.get(key, 0) for key in dict1.keys() | dict2.keys()}.items()))


def create_combined_dict(count_dicts:Dict[str,Any], level:int=1)->Dict[str,Any]:
    """
    Merges the countries in the counts_dicts
    
    if level == combines _mofa and _exec

    else combines _2 _3
    """
    stem= 7 if level==0 else 3 

    merged_dict= dict()
    
    for k, d in tqdm(count_dicts.items()):
        k_stem = k.split('/')[-1][:stem]
        if k_stem in merged_dict:
            merged_dict[k_stem] = merge_and_sum(merged_dict[k_stem],d)
        else:
            merged_dict[k_stem] = d
    
    return dict(sorted(merged_dict.items()))


def get_country_dicts(countries:List[str]):
    """From the list of input paths, countries, returns the year counts"""

    countries_year_counts = dict()

    for country in tqdm(countries):
        # if country in processed_countries: continue
        country_date_series= get_country_date_series(country)
        countries_year_counts[country]=group_date_series(country_date_series)

    return countries_year_counts


def get_image_counts(countries:List[str])->List[int]:
    image_counts=[]

    for country in countries:
        # news-id s related to the images
        # some news dont have any images and therefore
        # do not have a record in the images.jsonl
        image_ids = list(map(lambda x:json.loads(x)['news-id'], open(f'{country}/images.jsonl','r').read().splitlines()))
        # we will identify those ids and infer they have 0 images
        every_news_id = list(map(lambda x:json.loads(x)['id'], open(f'{country}/news.jsonl','r').read().splitlines()))
        zero_img_news= len(set(every_news_id).difference(set(image_ids)))
        image_counts.extend([0]*zero_img_news)
        image_counts.extend(list(Counter(image_ids).values()))
    return image_counts


def get_language_counts(countries:List[str])->Dict[str,int]:
    languages=[]
    for country in countries:
        languages.extend(
            list(map(lambda x:json.loads(x)['lang'], open(f'{country}/news.jsonl','r').read().splitlines()))
        )
    languages= dict(sorted(Counter(languages).items(),key= lambda x:x[1], reverse=True))
    languages.pop('low_conf')
    languages= {langcode_get(k).display_name():v for k,v in list(languages.items())[:10]}
    return languages


def get_content_length_distribution(countries:List[str])->List[int]:
    length_distribution= []
    for country in countries:
        length_distribution.extend(
            list(map(lambda x:len(json.loads(x)['content']), open(f'{country}/news.jsonl','r').read().splitlines()))
        )
    return length_distribution

### Plotting functions

# compressed version of greys colormap
grays = plt.cm.Greys(np.linspace(0, 1, 256))
compressed_x = np.hstack([np.linspace(0, 0.375, 48), np.linspace(0.375, 1, 208)])
new_colors = plt.cm.Greys(compressed_x)
custom_grays = mcolors.ListedColormap(new_colors, name="CompressedGrays")

def plot_heatmap(country_dict:dict, begin:int, country_limitations:list|None=lowy_66, ax:Axes=None):

    country_dicts = copy.deepcopy(country_dict)
    # if selects a subset
    if country_limitations:
        country_dicts = {k:v for k,v in country_dicts.items() if k in country_limitations}

    # Adding the boundaries
    country_dicts = {k: fill_year_dict(v, begin) for k,v in country_dicts.items()}
    # Converting counts to frequencies, i.e range [0,1]
    country_totals= {k: sum(v.values()) for k,v in country_dicts.items()}
    country_dicts = {k: {i:j/country_totals[k] for i,j in v.items()}  for k,v in country_dicts.items()}

    matrix = np.array([list(i.values()) for i in country_dicts.values()])
    labels = list(country_dicts.keys())

    # fig, ax = plt.subplots(figsize=(20, 4), dpi=100)
    sns.heatmap(data=matrix.T[::-1], xticklabels=labels, 
            # square=True,
            yticklabels=np.arange(begin, 2025)[::-1], cmap=custom_grays,
            vmax=0.5, ax=ax, cbar=False )# will show the colorbar later
    ax.set_yticklabels(np.arange(begin, 2025)[::-1], **font_dict)
    ax.set_xticklabels(ax.get_xticklabels(), **font_dict)


def plot_languages(lang_dict, ax):
    # raise NotImplemented('languages')
    ax.bar(x=lang_dict.keys(), height=lang_dict.values(), **plot_style)
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('# of Articles', **font_dict)
    ax.set_xticklabels(lang_dict.keys(), rotation=90)
    ax.set_title('Language Distribution', **title_font)
    ax.set_xticklabels(ax.get_xticklabels(), **font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), **font_dict)
    

def plot_start_years(country_dicts:str, ax:Axes, begin:int=1990):

    start_years= [ str(min([int(i) for i in v.keys() if int(i)>=begin])) for v in copy.deepcopy(country_dicts).values() if v]
    start_years= Counter(start_years)
    start_years= dict(sorted(fill_year_dict(start_years,begin).items()))

    ax.bar(x=start_years.keys(), height= start_years.values(), **plot_style)
    ax.set_xticklabels(list(map(lambda x: f'\'{x[-2:]}',list(start_years.keys())[::4])),rotation=0)

    ax.set_xticks([str(y) for y in range(begin, 2025, 4)]) 
    ax.set_xticks([str(y) for y in range(begin, 2025, 1)], minor=True)

    ax.set_title('First Published Article Year', **title_font)
    ax.set_ylabel('# of Countries', **font_dict)
    ax.set_xticklabels(ax.get_xticklabels(), **font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), **font_dict)

def plot_content_length_dist(length_distribution:list, ax:Axes):
    # raise NotImplemented('length dist')
    ax.hist(x=np.log10(list(map(lambda x: x+1,length_distribution))), bins=100, **plot_style )
    ax.set_yscale('log')
    ax.set_xticklabels([f'$10^{int(t)}$' for t in ax.get_xticks()], **font_dict)
    ax.set_title('Content Length', **title_font)
    ax.set_ylabel('# of Articles', **font_dict)
    ax.set_xlabel('# of Characters', **title_font)

def plot_image_counts(image_count_dist:list, ax:Axes):
    ax.hist(np.log10(list(map(lambda x:x+1,image_count_dist))),bins=120, **plot_style)
    ax.set_yscale('log')
    ax.set_title('Number of Images per Article',size =14)
    ax.set_xlabel('# of Images + 1', **title_font)
    ax.set_xticks(ax.get_xticks()[1::2])
    ax.set_xticklabels([f'$10^{int(t)}$' for t in ax.get_xticks()], **font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), **font_dict)
    ax.set_ylabel('# of Articles', **font_dict)
