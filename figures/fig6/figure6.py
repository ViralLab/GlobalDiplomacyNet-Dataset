############################################################
# There are some redundant functions that are not being used
# They are special visualizations woooOOOooo 👻 ooohooo
############################################################

import json, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
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

categories = ['F', 'PF', 'NF']
category_labels = {'F': 'Free', 'PF': 'Partly Free', 'NF': 'Not Free'}

freedom_colors = {
    'F': (0, 0, 1, 1),    # Blue for Free
    'PF': (1, 1, 0, 1),   # Yellow for Partly Free
    'NF': (1, 0, 0, 1),    # Red for Not Free
    'No Data': (0.5, 0.5, 0.5, 1)  # Default gray for No Data
}


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


def get_image_ratios(countries:List[str],min_total_people=100,min_total_images=100)->Dict[str,Any]:
    country_gender_ratio= dict()
    for country in countries:

        country_name=country.split('/')[-1][:3]

        image_file=open(f'{country}/images.jsonl','r').read().splitlines()
        # filtering for images with people in it only
        image_json_list = list(filter(lambda x: x['male-count'] + x['female-count'] > 0, map(lambda x:json.loads(x), image_file)))
        total_images=len(image_json_list)

        male_count = sum(map(lambda x:x['male-count'],image_json_list))
        female_count = sum(map(lambda x:x['female-count'],image_json_list))

        # refusing pandas dataframe usage
        if country_name in country_gender_ratio:
            current_values = country_gender_ratio[country_name]
            country_gender_ratio[country_name] = {'male-count':current_values['male-count'] + male_count,
                                                'female-count':current_values['female-count'] + female_count, 
                                                'total-images':current_values['total-images'] + total_images}
        else:
            country_gender_ratio[country_name] = {'male-count':male_count, 
                                                  'female-count':female_count, 
                                                  'total-images':total_images}

    country_gender_ratio= {k:v for k,v in country_gender_ratio.items()
                            if v['female-count']+v['male-count']>= min_total_people and v['total-images']>=min_total_images}

    return country_gender_ratio


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

def fetch_flag_image_with_border(country_code, border_color, draw_box:bool=False):
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
    if draw_box:
        draw.rectangle(
            [(0, 0), (bordered_img.width - 1, bordered_img.height - 1)], fill=border_rgba, outline=border_rgba
        )
    bordered_img.paste(img, (border_size, border_size), img)
    return bordered_img

# I'm trying to make this function general in most places, but it will never change
def plot_female_percentage(countries:List[str], ax_list:list[Axes], fig:Figure, zoom:float=0.25)->None:
    countries_female_ratio = {k:v['female-count']/(v['female-count']+v['male-count'])*100 for k,v in get_image_ratios(countries).items()}
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

#### c)

def get_gender_ratios(country_list:List[str])->Dict[str,Any]:
    
    country_gender_ratio= defaultdict(lambda: defaultdict(list))

    for country in country_list:

        country_name=country.split('/')[-1][:3]

        image_file=list(map(json.loads,open(f'{country}/images.jsonl','r').read().splitlines()))

        male_ratio = [x['male-count'] / (x['male-count'] + x['female-count']) if (x['male-count'] + x['female-count']) != 0 else 0.0 for x in image_file]
        female_ratio = [x['female-count'] / (x['male-count'] + x['female-count']) if (x['male-count'] + x['female-count']) != 0 else 0.0 for x in image_file]
        

        country_gender_ratio[country_name]['male_ratio'].extend(male_ratio)
        country_gender_ratio[country_name]['female_ratio'].extend(female_ratio)


    return country_gender_ratio



def assign_grids(country_stats:Dict[str,Any])->List[Any]:

    # Determining max value for axes
    max_avg_value = max(max(v[0] for v in country_stats.values()), max(v[1] for v in country_stats.values()))

    # Create an 8x8 grid
    num_cells = 8
    cell_size = max_avg_value / num_cells

    # Assigning countries to grid cells with limit of 30 countries per cell
    grid = [[[] for _ in range(num_cells)] for _ in range(num_cells)]

    # Assigning to grids
    for country_code, stats in country_stats.items():
        avg_female = stats[0]
        avg_male = stats[1]
        col = min(int(avg_female // cell_size), num_cells - 1)
        row = min(int(avg_male // cell_size), num_cells - 1)
        grid[row][col].append((country_code, stats))

    for row_idx in range(num_cells):
        for col_idx in range(num_cells):
            cell = grid[row_idx][col_idx]
            # Sort countries in descending order by image_count
            cell_sorted = sorted(cell, key=lambda x: x[1][-1], reverse=True)
            # Limit to top 30 countries
            grid[row_idx][col_idx] = cell_sorted[:30]
    
    return grid


def plot_male_female_per_image(countries:List[str], ax:Axes, flags_per_row:int = 6, flags_per_col:int =5, draw_box:bool=False, zoom:float=0.12 )->None:

    # Calculate avg. males and females on images
    country_stats={k:(
                        v['female-count']/v['total-images'],
                        v['male-count']/v['total-images'], 
                        v['total-images']
                    ) for k,v in get_image_ratios(countries).items()}
    # Freedom score for bounding boxes
    freedom_status =get_freedom_house_classification(country_stats.keys())
    # Grouping them on a grid
    grid = assign_grids(country_stats)


    #################
    # Drawing grid lines
    for i in range(8):
        ax.axhline(y=i, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=i, color='gray', linestyle='--', linewidth=1)

    # Setting axes limits and labels
    ax.set_xlim(0, 2.5)
    ax.set_ylim(1, 8)
    ax.set_aspect('auto')
    ax.set_xticks([0, 1, 2, 2.5])
    ax.set_yticks(range(1, 9))
    ax.set_xticklabels(['0', '1', '2', '2.5'])
    ax.set_facecolor('white')  # Set white background

    # x=y line for female=male
    ax.plot([1, 2.5], [1, 2.5], 'r--', linewidth=1.5)


    #################
    # Adding flags to each grid cell
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if col_idx >= 3:  # Skip columns beyond index 2
                continue    # there aren't any anyhow
                
            start_x = col_idx
            start_y = row_idx

            # flags_per_row = 6
            # flags_per_col = 5
            max_flags_per_cell = flags_per_row * flags_per_col

            num_flags = min(len(cell), max_flags_per_cell)

            # Calculate spacing
            h_padding = 0.1
            h_usable_space = 1 - (2 * h_padding)
            h_spacing = h_usable_space / (flags_per_row - 1) if flags_per_row > 1 else 0

            v_padding = 0.15
            v_usable_space = 1 - (2 * v_padding)
            v_spacing = v_usable_space / (flags_per_col - 1) if flags_per_col > 1 else 0

            x_positions = [start_x + h_padding + (i * h_spacing) for i in range(flags_per_row)]
            y_positions = [start_y + (1 - v_padding) - (i * v_spacing) for i in range(flags_per_col)]

            for i, (country_code, stats) in enumerate(cell[:num_flags]):
                avg_female = stats[0]
                avg_male = stats[1]
                
                if i >= max_flags_per_cell:
                    break

                current_row = i // flags_per_row
                current_col = i % flags_per_row

                x_pos = x_positions[current_col]
                y_pos = y_positions[current_row]

                try:
                    clean_code = country_code.replace("_", " ").strip().upper()
                    status = freedom_status.get(clean_code, 'Unknown')
                    border_color = freedom_colors.get(status, (0.5, 0.5, 0.5, 1))

                    flag_img = fetch_flag_image_with_border(country_code, border_color, draw_box)
                    imagebox = OffsetImage(flag_img, zoom=zoom)
                    ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
                    ax.add_artist(ab)
                except:
                    print(f"Error processing {country_code}:")
                    # print(f"Error details: {str(e)}")
                    continue


    #################
    # Final plot tweaks
    ax.set_xlabel("Avg. # of Females per Image", **label_text, labelpad=8)
    ax.set_ylabel("Avg. # of Males per Image", **label_text, labelpad=8)
    ax.tick_params(axis='both', which='major', **tick_text)
    ax.grid(False)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.set_axisbelow(False)  # This ensures axes are drawn on top of grid

    ax.set_xticklabels(ax.get_xticklabels(),**tick_text)
    ax.set_yticklabels(ax.get_yticklabels(),**tick_text)
    
    if draw_box:
        legend_elements = [
            Patch(facecolor=freedom_colors['F'], label='Free'),
            Patch(facecolor=freedom_colors['PF'], label='Partly Free'),
            Patch(facecolor=freedom_colors['NF'], label='Not Free'),
            # Patch(facecolor=freedom_colors['Unknown'], label='No Data')
        ]

        plt.legend(handles=legend_elements, ncol=1, 
                bbox_to_anchor=(1, 0.00),  # x=0.98 for slight padding from right, y=0.15 to be above x-axis
                loc='lower right', 
                fontsize=12,
                markerfirst=False)


def plot_country_genders_ratios(country_iso_codes:List[str], ax_list:List[Axes], fig, data_dir:str ,max_people:int=np.inf):
    
    assert len(country_iso_codes)==len(ax_list), 'length of the countries and axes are not matching'

    for idx, iso3 in enumerate(country_iso_codes):

        gender_dists = get_gender_ratios(get_countries(data_dir, include=[f'{iso3}_']))
        male_dist = list(map(lambda x:min(x,max_people), gender_dists[iso3]['male_ratio']))
        female_dist = list(map(lambda x:min(x,max_people), gender_dists[iso3]['female_ratio']))

        # male_counts = dict(sorted(Counter(male_dist).items()))
        # female_counts = dict(sorted(Counter(female_dist).items()))

        ax = ax_list[idx]
        # sns.histplot(male_dist, color="#51a1de", label='Male Ratio',binwidth=0.05, ax=ax, alpha=0.5)
        sns.histplot(female_dist, color='#f5a9b8', label='Female Ratio', binwidth=0.05, ax=ax, alpha=0.7)

        # xticks= ax.get_xticks()
        # ax.set_xticklabels([int(i) if i==int(i) else '' for i in xticks ])
        # from matplotlib.ticker import FuncFormatter
        # ax.xaxis.set_major_formatter(FuncFormatter(
        #     lambda x, _: f"{int(x)}" if x.is_integer() else ""
        # ))
        # from matplotlib.ticker import MaxNLocator
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.setp(ax.get_xticklabels(), **tick_text)
        plt.setp(ax.get_yticklabels(), **tick_text)

        ax.set_yscale('log')
        # ax.set_xlabel('# of People per Image',**label_text)
        ax.set_ylabel('# of Images',**label_text)
        title = ax.set_title(f'{iso3} Images Gender Distribution',**title_text)

        fig.canvas.draw()
        bbox = title.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = ax.transAxes.inverted()
        x0, y0 = inv.transform((bbox.x0, bbox.y0))


        flag_img = fetch_flag_image(get_alpha2_code(iso3))
        imagebox = OffsetImage(flag_img, zoom=0.09)

        margin = 0.01 
        ab = AnnotationBbox(imagebox,
            (x0 - margin,    y0 + 0.03),
            box_alignment=(1, 0.5),
            frameon=False,xycoords="axes fraction")
        ax.add_artist(ab)


        # print(title.get_position())
        ax.legend()
    # ax.set_xlabel('# of People per Image',**label_text)
