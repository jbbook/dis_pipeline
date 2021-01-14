"""process data (ETL pipeline)."""
import sys
import numpy as np
import pandas as pd
import copy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load two csv files, merges them on their commmon ID and returns as a DF."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')


def clean_data(df):
    """Clean data in dataframe and return cleaned data as dataframe."""
    # obtain cleaned category names from 'categories' column
    categories = df['categories'].str.split(pat=';', expand=True)
    category_colnames = categories.loc[0].str.split('-', expand=True)[0]
    categories.columns = category_colnames

    #convert category values to 0 and 1
    categories_temp = copy.copy(categories)
    for column in categories:
        #set each value to be the last character of the string
        categories_temp[column] = categories_temp[column].str.split('-', expand=True)[1]
        categories_temp[column] = categories_temp[column].astype(np.int)
        #change value of 2 to 1. occurs in 'related' category
        #could be performed outside of the loop to save comp. time. TODO
        #could be refactored to account for all numbers not (0, 1) for future data
        categories_temp[column].replace(2, 1, inplace=True)
        #only 0 and 1 -> can be stored as bool to save space.
        categories_temp[column] = categories_temp[column].astype(bool)

    # child_alone category is empty, no information
    categories_temp.drop(labels=['child_alone'], axis=1, inplace=True)

    #drop old categories column, re-merge dataframes
    df.drop(labels=['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_temp], axis=1)

    #remove duplicates
    df= df[~df.duplicated(subset=['message', 'original'])]


    return df


def save_data(df, database_filename):
    """Save dataframe into database."""
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
    pass  

def main():
    """Main function of process_data.py."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()