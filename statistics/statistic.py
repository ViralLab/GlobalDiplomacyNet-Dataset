import json, os, re, statistics
from collections import Counter
from tqdm import tqdm
import pandas as pd

data_dir = ' '
output_dir = ' '

alpha3_to_full = {i['ccode']:i['country_name'] for i in  json.loads(open('./country_codes2.json','r').read()).values()}

def make_statistic():

    rows=[]

    for country in tqdm(os.listdir(data_dir)):
        country_dict = dict()
        
        country_dict['Country Name'] = alpha3_to_full[country[:3]] if country[:3] in alpha3_to_full else ''
        country_dict['Country Code'] = country[:3]
        country_dict['Instution'] = country[4:]

        parsed_data= [json.loads(l) for l in open(f'{data_dir}/{country}/news.jsonl','r').read().splitlines()]
        country_dict['News Count']= len(parsed_data)

        years = set([x['date'][:4] for x in parsed_data if x['date']])
        if years:
            country_dict['Time Span']= f'{min(years)}-{max(years)}'
        else:
            country_dict['Time Span']= 'N/A'

        content_distrbution = [len(j['content']) for j in parsed_data if j['content']]
        country_dict['Median Content Length'] = int(statistics.median(content_distrbution))

        # Images are 100% where as the data is currently 10%
        parsed_data_ids=list(map(lambda x:x['id'], parsed_data))
        imgs = [json.loads(i) for i in open(f'{data_dir}/{country}/images.jsonl').read().splitlines()]
        imgs = [i for i in imgs if i['news-id'] in parsed_data_ids]
        country_dict['Avg. Image per News'] = round(len(imgs)/country_dict['News Count'],2)

        language_counts = Counter(map(lambda x:x['lang'], parsed_data))
        country_dict['Translated (%)'] = round((sum(language_counts.values())-language_counts['en'])/sum(language_counts.values())*100,1)


        country_dict['URL Host'] = re.search(pattern='(https?:\/\/(?!web)[^\/]+/)',
                                    string=parsed_data[1]['url']).group(1)

        rows.append(country_dict)


    df= pd.DataFrame(rows)
    df.sort_values(by=['Country Code','Instution']).to_excel(output_dir,index=False)

if __name__=='__main__':
    make_statistic()