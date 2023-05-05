import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages =  pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    """
    This function splitd categories into separate category columns and convert category values to just numbers 0 or 1

    """
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda s:s[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda s:s[-1])
        categories[column] = categories[column].astype(int)

    df = df.drop(['categories'],axis = 1)
    df = pd.concat([df,categories],axis = 1)

    df = df.drop('child_alone',axis = 1)
    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x) 

    df = df.drop_duplicates()

    return df

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
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