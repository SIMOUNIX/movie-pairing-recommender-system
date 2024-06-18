import shutil
import gzip
import requests
import os
import pandas as pd

# Download the IMDB dataset from the given URL
def download_imdb_dataset(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

# Extract the .gz file
def extract_gz_file(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def load_imdb_dataset():
    # URLs of the IMDb datasets
    urls = {
        'name.basics': 'https://datasets.imdbws.com/name.basics.tsv.gz',
        'title.basics': 'https://datasets.imdbws.com/title.basics.tsv.gz',
        'title.ratings': 'https://datasets.imdbws.com/title.ratings.tsv.gz'
    }

    # Paths to save the downloaded files
    download_paths = {
        'name.basics': '../data/imdb/name.basics.tsv.gz',
        'title.basics': '../data/imdb/title.basics.tsv.gz',
        'title.ratings': '../data/imdb/title.ratings.tsv.gz'
    }

    # Paths to save the extracted files
    extracted_paths = {
        'name.basics': '../data/imdb/name.basics.tsv',
        'title.basics': '../data/imdb/title.basics.tsv',
        'title.ratings': '../data/imdb/title.ratings.tsv'
    }
    
    # Create directories if they do not exist
    for path in download_paths.values():
        print(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    for key in urls.keys():
        print(f'Downloading {key}...')
        download_imdb_dataset(urls[key], download_paths[key])
        print(f'Extracting {key}...')
        extract_gz_file(download_paths[key], extracted_paths[key])

    # Load the datasets into pandas DataFrames
    name_basics = pd.read_csv(extracted_paths['name.basics'], sep='\t', dtype=str)
    title_basics = pd.read_csv(extracted_paths['title.basics'], sep='\t', dtype=str)
    title_ratings = pd.read_csv(extracted_paths['title.ratings'], sep='\t', dtype=str)
    
    return name_basics, title_basics, title_ratings



