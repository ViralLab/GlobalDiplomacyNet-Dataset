import json, os
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
import pickle
from tqdm import tqdm
font_dict = {'size':12}
tick_dict = {'size':10}

g20_name_lookup = {'ARG': 'Argentina', 'AUS': 'Australia', 'BRA': 'Brazil', 'CAN': 'Canada', 'CHN': 'China', 'FRA': 'France', 'DEU': 'Germany', 'IND': 'India', 'IDN': 'Indonesia', 'ITA': 'Italy', 'JPN': 'Japan', 'MEX': 'Mexico', 'RUS': 'Russia', 'SAU': 'Saudi Arabia', 'ZAF': 'South Africa', 'KOR': 'South Korea', 'TUR': 'Türkiye', 'GBR': 'United Kingdom', 'USA': 'United States'}

### Helper functions
def get_countries(data_directory:str, include:List[str]|None=None, exclude:List[str]|None=None)->List[str]:
    all_countries = list(map(lambda x: f'{data_directory}/{x}',os.listdir(data_directory)))
    if include: 
        all_countries = [i for i in all_countries if any(sub in i for sub in include)]
    if exclude: 
        all_countries = [i for i in all_countries if not any(sub in i for sub in exclude)]

    return all_countries

def initialize_period_counter(start_year:int, end_year:int, month_frequency:int) -> List[str]:
    # month_frequency will never change but whatever
    assert 12 % month_frequency == 0, "Month frequency must be a factor of 12"

    period_count = dict()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13, month_frequency):
            period = f'{year}-{month//month_frequency}'
            period_count[period] = 0
    return Counter(period_count)

def get_entity_mentions(
        entities:List[int], 
        countries_to_include:List[str],
        month_frequency:int, 
        start_year:int, 
        end_year:int)->Dict[int,Any]:
    """
    This function should be given a list of wikipedia id s
    It'll group the occurences of the entity from the countries list
    and from the start and end year with the provided frequency
    And return the period counter
    
    countries_to_include should be the output of get_countries, another function
    same as in fig2
    """

    entity_mentions_counter = defaultdict(lambda: initialize_period_counter(start_year, end_year, month_frequency))

    for country in countries_to_include:

        # news= list(
        #     map(json.loads, open(f'{country}/news.jsonl','r').read().splitlines())
        #     )
        news = list(
            filter(
                lambda x: x['date'] and start_year <= int(x['date'][:4]) <= end_year, #news
                    map(json.loads, open(f'{country}/news.jsonl','r').read().splitlines())
            )
        )
        
        for entity in entities:
            entity_mentions= filter(
                lambda x: entity in sum(x['wikidata_qids'].values(),[]), news
            )
            entity_mention_periods = list(map(
                lambda x: f"{x['date'][:4]}-{(int(x['date'][5:7])-1)//month_frequency}", entity_mentions
            ))

            entity_mentions_counter[entity].update(entity_mention_periods)
    
    entity_mentions_counter = {k: dict(sorted(v.items())) for k, v in entity_mentions_counter.items()}

    return entity_mentions_counter
    

def plot_trend( entities:List[Tuple],
                countries_to_include:List[str], ax:Axes=None,
                month_frequency:int=6, start_year:int=2000, end_year:int=2024):

    entity_mentions= get_entity_mentions(list(map(lambda x:x[0],entities)),
                                        countries_to_include=countries_to_include, month_frequency=month_frequency,
                                        start_year=start_year, end_year=end_year)

    for idx, val in enumerate(entity_mentions.values()):
        label, color = entities[idx][1:]
        ax.plot(list(val.values()),label = label, c=color, marker='o')
        # I want to get the periods as well for the x ticks
        # Doing it inside the loop because as we are already calling the values
        # I cant do it outside the for loop in a clean way as there is no slicing for dictionary
        # Dont have to do it at every iter. but its okay
        x_ticks = list(val.keys())

    x_tick_labels= [f'{t[:4] if t.endswith("0") else ""}' for t in x_ticks]
    ax.set_xticks(ticks=range(0,  len(x_ticks)), labels=x_tick_labels, rotation=90, **font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), **font_dict)
    
    ax.set_ylabel('Mention Count', **font_dict)
    ax.grid(which='both',linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')


def get_org_mentions(countries:List[str], orgs:List[tuple],data_directory:str):
    """countries should be list of ISO3 codes,
    orgs should be: [(label, qid), ]"""
    org_labels, org_qids = list(map(lambda x:x[0],orgs)), list(map(lambda x:x[1],orgs))
    qid_to_label = dict(zip(org_qids,org_labels))
    country_org_mentions = defaultdict(lambda: Counter())

    for c in tqdm(countries):
        country_org_mentions[c].update({k:0 for k in org_labels}) # initialization
        
        for country_dirs in get_countries(data_directory, c):

            country_jsonl = list(map(json.loads,open(f'{country_dirs}/news.jsonl','r').read().splitlines()))

            for j in country_jsonl:
                news_orgs = list(map(lambda x:qid_to_label[x],set(j['wikidata_qids']['organizations']).intersection(org_qids)))
                country_org_mentions[c].update(news_orgs)
        
    return {isocode:{org_name:round(v/sum(values.values())*100,2) for org_name,v in values.items()} for isocode, values in country_org_mentions.items()}


def plot_org_heatmap(countries:List[tuple], orgs:List[tuple], ax:Axes, data_directory:str):
    """
    countries should be just ISO3 codes,

    orgs should be: [(label, qid), ]
    
    """
    org_labels, org_qids = list(map(lambda x:x[0],orgs)), list(map(lambda x:x[1],orgs))
    heatmap_values = get_org_mentions(countries, orgs,data_directory)

    org_order = sorted(org_labels)
    # sorting for better look
    heatmap_values = dict(sorted(map(lambda x:(x[0],dict(sorted(x[1].items()))),heatmap_values.items())))
    data = np.array(list(map(lambda x:list(x.values()),heatmap_values.values())))
    mask = data==0
    # TODO change labels to full names with a harcoded dict
    labels = list(map(lambda x:g20_name_lookup[x],heatmap_values.keys()))
    sns.heatmap(data,yticklabels=labels, xticklabels= org_order, cmap='YlGnBu',square=True,fmt='.2f',annot=True,mask=mask, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

