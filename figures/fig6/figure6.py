import json, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

import requests
from PIL import Image, ImageDraw
from io import BytesIO

import pandas as pd
from pycountry import countries as pyc
from copy import deepcopy

from typing import List, Dict, Any
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')


tick_text= {'size':12}
label_text= {'size':14}
title_text= {'size':14}


freedom_colors = {
    'F': (0, 0, 1, 1),    # Blue for Free
    'PF': (1, 1, 0, 1),   # Yellow for Partly Free
    'NF': (1, 0, 0, 1),    # Red for Not Free
    'No Data': (0.5, 0.5, 0.5, 1)  # Default gray for No Data
}

categories = ['F', 'PF', 'NF']
category_labels = {'F': 'Free', 'PF': 'Partly Free', 'NF': 'Not Free'}


### Helper functions
def get_countries(data_directory:str, include:List[str]|None=None, exclude:List[str]|None=None)->List[str]:
    all_countries = list(map(lambda x: f'{data_directory}/{x}',os.listdir(data_directory)))
    if include: 
        all_countries = [i for i in all_countries if any(sub in i for sub in include)]
    if exclude: 
        all_countries = [i for i in all_countries if not any(sub in i for sub in exclude)]
    return all_countries

def get_image_per_country(countries: List[str])->Dict[str,int]:
    country_avg_img = dict()
    for country in countries:
        num_news = len(open(f'{country}/news.jsonl','r').read().splitlines())
        num_images = len(open(f'{country}/images.jsonl','r').read().splitlines())
        country_avg_img[country.split('/')[-1]]=num_images/num_news
    return dict(sorted(country_avg_img.items(), key=lambda x:x[1],reverse=True))

def fetch_flag_image(country_code):
    url = f"https://flagcdn.com/w80/{country_code.lower()}.png"
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    # Standardize flag size (maintaining 5:3 aspect ratio)
    standard_size = (160, 96)
    img = img.resize(standard_size, Image.Resampling.LANCZOS)

    return img

# TODO: maybe give up and use a dataframe
def get_female_ratios(countries:List[str],min_total_people=100,min_total_images=100):
    country_gender_ratio= dict()
    for country in countries:

        country_name=country.split('/')[-1][:3]

        image_file=open(f'{country}/images.jsonl','r').read().splitlines()
        total_images=len(image_file)

        image_json_list = list(map(lambda x:json.loads(x), image_file))
        male_count = sum(map(lambda x:x['male-count'],image_json_list))
        female_count = sum(map(lambda x:x['female-count'],image_json_list))

        # poor mans dataframe
        if country_name in country_gender_ratio:
            current_values = country_gender_ratio[country_name]
            country_gender_ratio[country_name] = {'male-count':current_values['male-count'] + male_count,
                                                'female-count':current_values['female-count'] + female_count, 
                                                'total-images':current_values['total-images'] + total_images}
        else:
            country_gender_ratio[country_name] = {'male-count':male_count, 
                                                  'female-count':female_count, 
                                                  'total-images':total_images}

    country_gender_ratio= {k:v['female-count']/(v['female-count']+v['male-count'])*100 for k,v in country_gender_ratio.items()
                            if v['female-count']+v['male-count']>= min_total_people and v['total-images']>=min_total_images
                            }

    return dict(sorted(country_gender_ratio.items(), key=lambda x:x[1],reverse=True))


def get_freedom_house_classification(country_codes:List[str])->Dict[str,Any]:
    freedom_df = pd.read_excel("./freedomhousewithccodes.xlsx", sheet_name='Sheet1')
    freedom_df = freedom_df[freedom_df['Edition'] == 2024]  # Filter for 2024 edition
    freedom_dict = dict(zip(freedom_df['ccode'], freedom_df['Status']))
    freedom_dict = {c:freedom_dict[c] if c in freedom_dict else 'No Data' for c in country_codes}
    return freedom_dict

def get_alpha2_code(alpha3_code):
    """Convert 3-letter country code to 2-letter code using pycountry"""
    try:
        return pyc.get(alpha_3=alpha3_code).alpha_2.lower()
    except Exception as e:
        if alpha3_code=='XKX': return 'xk'
        else:
            print(f'Couldn\'t convert {alpha3_code}: {e}')
            return None
        
### Plotting Functions

#### a) plot

def plot_hist(country_avg_img, ax):

    country_avg_img = {k:v for k,v in deepcopy(country_avg_img).items() if v>0}
    values = [np.log10(i) for i in country_avg_img.values() if i < 100]

    n, bins, patches = plt.hist(values, bins=40, color='#A3A3A3',orientation='horizontal')

    non_zero_indices = [i for i, count in enumerate(n) if count > 0]
    min_bin_idx = min(non_zero_indices)
    max_bin_idx = max(non_zero_indices)

    # Calculate the center of each bin
    min_bin_center = (bins[min_bin_idx] + bins[min_bin_idx + 1]) / 2
    max_bin_center = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2

    # Get the height (count) of each bin
    min_bin_count = n[min_bin_idx]
    max_bin_count = n[max_bin_idx]

    for p, (c, v) in enumerate(list(country_avg_img.items())[:3]):
        if p==0:
            ax.annotate(f'$1^{{st}}$  {c}: {v:.2f}',
                        xy=(max_bin_count, max_bin_center),  # Position at the max bar
                        xytext=(max_bin_count + 13, max_bin_center-0.01),  # Text to the right
                        arrowprops=dict(arrowstyle='<-',facecolor='black',connectionstyle="angle,angleA=0,angleB=90"),**tick_text)
        else:
            if p==1: up='nd'
            elif p==2:up='rd'
            ax.annotate(f'${p+1}^{{{up}}}$  {c}: {v:.2f}',
                        xy=(min_bin_count+6, min_bin_center),  # Position at the min bar
                        xytext=(max_bin_count + 13, max_bin_center-0.01-p/5),  # Text to the right
                        **tick_text)

    for p, (c, v) in enumerate(list(country_avg_img.items())[-3:][::-1]):
        if p==0:
            ax.annotate(f'$Last$  {c}: {v:.5f}',
                        xy=(min_bin_count, min_bin_center),  # Position at the min bar
                        xytext=(min_bin_count + 3, min_bin_center),  # Text to the righ
                        arrowprops=dict(arrowstyle='<-',facecolor='black',connectionstyle="angle,angleA=0,angleB=90" ),**tick_text)
        else:
            ax.annotate(f'{"":9}{c}: {v:.5f}',
                        xy=(min_bin_count+6, min_bin_center),  # Position at the min bar
                        xytext=(min_bin_count + 3, min_bin_center+p/5),  # Text to the righ
                        **tick_text)

    ax.set_xticklabels(ax.get_xticklabels(),**tick_text)
    ax.set_yticklabels([f'10$^{{{i.get_text()}}}$' for i in ax.get_yticklabels()],**tick_text)

    sns.despine(ax=ax)

    ax.set_ylabel('Avg. # of Images per News',**label_text)
    ax.set_xlabel('# of websites',**label_text,labelpad=8)



#### b) plot

def invert_dict(original:Dict[Any,Any])->Dict[Any,List]:
    inverted = defaultdict(list)
    for key, value in original.items():
        inverted[value].append(key)
    return dict(inverted)

def find_closest_value(target, values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.argmin(np.abs(values - target))]

def assign_categories(ratios, classifications, midpoints):
    category_data = defaultdict(list)

    for cat, v in invert_dict(classifications).items():
        for country in v:
            ratio = ratios[country]
            closest_value = find_closest_value(ratio, midpoints)
            iso2_code = get_alpha2_code(country)
            category_data[cat].append((closest_value,iso2_code))

    return dict(category_data)

def fetch_flag_image_with_border(country_code, border_color):
    """Fetch flag image using the country code"""
    iso2_code = get_alpha2_code(country_code)
    
    if not iso2_code:
        raise ValueError(f"Could not convert country code: {country_code}")
        
    url = f"https://flagcdn.com/w80/{iso2_code}.png"
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGBA")

    standard_size = (160, 96)
    img = img.resize(standard_size, Image.Resampling.LANCZOS)

    if isinstance(border_color, tuple):
        border_rgba = tuple(int(c * 255) for c in border_color[:3]) + (255,)
    else:
        border_rgba = (128, 128, 128, 255)  # Default gray

    border_size = 15
    bordered_img = Image.new("RGBA", (standard_size[0] + 2 * border_size, standard_size[1] + 2 * border_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bordered_img)
    draw.rectangle(
        [(0, 0), (bordered_img.width - 1, bordered_img.height - 1)], fill=border_rgba, outline=border_rgba
    )
    bordered_img.paste(img, (border_size, border_size), img)
    return bordered_img

# I'm trying to make this function general in most places, but it will never change
def plot_female_percentage(countries:List[str], ax_list:list[Axes], fig:Figure, zoom:float=0.25)->None:
    countries_female_ratio = get_female_ratios(countries)
    freedom_classifications = get_freedom_house_classification(countries_female_ratio.keys())

    min_ratio = np.floor(min(countries_female_ratio.values()))  # Round down to nearest integer
    max_ratio = np.ceil(max(countries_female_ratio.values()))   # Round up to nearest integer
    step_size = (max_ratio - min_ratio) / 10
    intervals = [round(min_ratio + i * step_size, 2) for i in range(11)]
    midpoints = [(intervals[i] + intervals[i+1]) / 2 for i in range(10)]

    category_data = assign_categories(countries_female_ratio,freedom_classifications,midpoints)

    for i, category in enumerate(categories):
        ax= ax_list[i]
        # ax = fig.add_subplot(ax_list[i])
        # print(type(ax))
        ax.set_ylabel(category_labels[category], **label_text,labelpad=10)
        ax.yaxis.set_label_position("left")
        ax.set_xlim(intervals[0] - 1, intervals[-1] + 1)
        ax.set_xticks(intervals)
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='gray', linewidth=0.8)
        stacked_positions = {}
        # this is trial and errored value
        flag_height = 0.1 * 96 / fig.dpi  # Reduced flag height

        for x_val, country_code_iso2 in category_data[category]:
            stack_position = stacked_positions.get(x_val, 0)
            y_pos = stack_position * flag_height + flag_height / 2

            try:
                flag_img = fetch_flag_image(country_code_iso2)
                imagebox = OffsetImage(flag_img, zoom=zoom)  # Reduced zoom factor
                ab = AnnotationBbox(imagebox, (x_val, y_pos), box_alignment=(0.5, 0.5), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error fetching flag for {country_code_iso2}, {category}: {e}")

            stacked_positions[x_val] = stack_position + 1

        max_stack = max(stacked_positions.values()) if stacked_positions else 1
        ax.set_ylim(-flag_height / 2, max_stack * flag_height + flag_height / 2)
        ax.set_yticks([])

        if i != len(categories)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{label:.1f}" for label in intervals], **tick_text)
            ax.set_xlabel("Percentage of Female Representation", **label_text,labelpad=8)#, labelpad=15)
    ax_list[0].text(-0.05, 1.12, 'b)', transform=ax_list[0].transAxes, 
    size=16, va='top', ha='right')

#### c)
# TODO: do the c) figure