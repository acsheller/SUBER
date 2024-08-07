import os, gc
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
from environment.item import ItemsLoader
from environment.mind.news import News
from environment.mind.datasets.load_mind_data import load_data, news_columns, behaviors_columns, train_path, save_catagories_to_csv

from environment.mind.datasets.load_small_mind_data import load_data_small, news_columns_small, behaviors_columns_small, train_path_small, save_catagories_to_csv_small


from algorithms.logging_config  import get_logger

logger = get_logger("suber_logger")


class NewsLoader(ItemsLoader):
    '''
    Responsible for loading news articles
    '''
    _news_data = None  # Class-level attribute to store the loaded data
    _dataset = None

    def __init__(self,dataset_to_use="small_mind_dataset"):
        '''
        Similar to movies we want to pass in data.  Movies does it with a Json file
        We can do it with a dataframe to make it more efficient. IMHO.

        On the first time it is launched it creates the datafile which can be time consuming.

        '''
        if not NewsLoader._dataset:
            NewsLoader._dataset = dataset_to_use
        else:
            dataset_to_use = NewsLoader._dataset
        # Path for the preprocessed file
        self.dataset_to_use = dataset_to_use
        pp_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"datasets/",)
        if self.dataset_to_use == 'small_mind_dataset':
            data_path = os.path.join(pp_save_path, "small_news_pp.csv")
        else:
            data_path = os.path.join(pp_save_path, "news_pp.csv")
        if NewsLoader._news_data is not None and  os.path.exists(data_path):
            self.data = NewsLoader._news_data
        elif NewsLoader._news_data is None and  os.path.exists(data_path):
            # Load the preprocessed data if it exists
            logger.info(f"{data_path} exists. Loading preprocessed data.")
            NewsLoader._news_data = pd.read_csv(data_path)
            self.data = NewsLoader._news_data
            self.data.set_index('id', inplace=True)
        else:
            # Load and preprocess the data if the preprocessed file does not exist
            self.preprocess_data(pp_save_path, dataset_to_use)

        
    def preprocess_data(self, pp_save_path, dataset_to_use):
        '''
        Load the original data and preprocess it if the preprocessed file does not exist
        '''
        logger.info(f"Data not processed yet. Proceeding with data augmentation.")

        if 'small' in dataset_to_use:
            self.data = load_data(os.path.join(train_path_small, 'news.tsv'), news_columns)
            self.behaviors = load_data(os.path.join(train_path_small, 'behaviors.tsv'), behaviors_columns)

        else:
            self.data = load_data(os.path.join(train_path, 'news.tsv'), news_columns)
            self.behaviors = load_data(os.path.join(train_path, 'behaviors.tsv'), behaviors_columns)
        # I'm saving these off as I use them later

        # Create a dataframe to store click counts and impression counts
        logger.info("Augmenting News Data with Historical User Behavior and Impressions")


        # Preprocess 
        # drop duplicates and Nulls
        self.data.drop_duplicates(subset=['title'],inplace=True)
        self.data.dropna(inplace=True)
        # make impressions a list
        self.behaviors['impressions'] = self.behaviors['impressions'].str.split()

        # Dictionairy of article stats
        article_stats = {}

        # Process impressions
        for impressions_index in tqdm(range(len(self.behaviors)), desc="--- Processing Impressions"):
            for impression in self.behaviors.loc[impressions_index]['impressions']:
                article_id, click = impression.split('-')
                click = int(click)
                if article_id not in article_stats:
                    article_stats[article_id] = {'clicks': 0, 'impressions': 0}
                article_stats[article_id]['impressions'] += 1
                if click == 1:
                    article_stats[article_id]['clicks'] += 1

        # Convert the stats dictionary to a DataFrame
        df_article_stats = pd.DataFrame.from_dict(article_stats, orient='index')
        df_article_stats['click_through_rate'] = df_article_stats['clicks'] / df_article_stats['impressions']
        df_article_stats['vote_average'] =df_article_stats['click_through_rate']
        df_article_stats['vote_count'] = df_article_stats['impressions']
        
        # Deleting the dictionary to free up memory
        del article_stats
        gc.collect()

        # Flatten the history column into a single list
        logger.info("Processing User Read History")
        all_histories = self.behaviors['history'].str.split().explode().dropna()

        # Count occurrences of each article in the history
        history_counts = Counter(all_histories)

        # Convert to DataFrame
        df_history_counts = pd.DataFrame.from_dict(history_counts, orient='index', columns=['read_frequency'])

        # Merge click_through_rate and click/impression counts with news dataframe
        self.data = self.data.merge(df_article_stats, left_on='id', right_index=True, how='left')

        # Merge Read Frequency with news dataframe
        self.data = self.data.merge(df_history_counts, left_on='id', right_index=True, how='left')
        
        # Deleting the behaviors dataframe to free up memory
        del self.behaviors
        gc.collect()

        # Fill NaN values with 0 (if necessary, since some news might not have clicks or history counts)
        self.data.dropna(subset=['read_frequency'],inplace=True)
        self.data.fillna(0, inplace=True)

        self.data.index = self.data["id"]
        gc.collect()

        if 'small' in dataset_to_use:
            data_path = os.path.join(pp_save_path, "small_news_pp.csv")
            logger.info("Saving off list of categories.")
            save_catagories_to_csv_small(self.data)
        else:
            data_path = os.path.join(pp_save_path, "news_pp.csv")
            logger.info("Saving off list of categories.")
            save_catagories_to_csv(self.data)

        # Save the preprocessed data to news_pp.csv
        self.data.to_csv(data_path, index=False)
        print(f"--- Preprocessed data saved to {data_path} for later use.")


    def load_all_ids(self):
        '''
        define this abstract method
        Return a list of ids
        '''
        return self.data.index.tolist()

    def load_items(self):
        '''
        Define the abstract method like books/movies
        returns:
            news (dictionary from news ids to news): a mapping for all 
        '''
        news = {}
        for article in self.data:
            news[article.index] = News.from_dataframe(article)
        return news
    
    def load_items_from_ids(self,id_list):
        '''
        Define the abstract method
        Returns a list of news articles
        '''
        news = []
        for id in id_list:
            #print("ID is {}".format(id))
            #print("\t{}\t\n".format(self.data.loc[[id]]))
            news.append(News.from_dataframe(self.data.loc[[id]]))
        return news

        
