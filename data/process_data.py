import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads and merges the dataframes.
    
    Args:
        messages_filepath: str , path for message file 
        categories_filepath: str , path for category file 

    Return:
        df: a dataframe combines messages and categories information
    """
    # Read the info contained in the csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)

    # Split column 'categories' into multiple rows, split and expand categories
    categories_expanded = split_expand_categories(df)
    df = df.merge(categories_expanded) # join on 'id'

    return df, categories_expanded


def split_expand_categories(df):
    """Splits and expands "categories"

    Args:
        categories: dataframe containing category information for each entries
    Return:
        categories: updated dataframe
    """
#     categories_in = df["categories"].copy()
    categories = df["categories"].str.split(";", expand=True)

    # verify if the sequence of category is consitent among all the rows
    for col in categories.columns:
        if len(categories[col].apply(lambda x: x[:-2:]).unique()) != 1:
            print("the cat sequence is not consistent among all the entries")

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2:])
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    categories['id'] = df['id']

    return categories


def clean_data(df):
    """"Removes duplicates
    
    to-dos: 
        - correct cat-values that is neither 0 nor 1 
    """
    df = df.drop_duplicates(inplace=True)
    df = df.replace(2,1)
    df = df.drop('categories', axis =1)
    # To do drop NaN files
    return df


def save_data(df, database_filename):
    """Loads the table to the database"""
    engine = create_engine("sqlite:///{:}".format(database_filename))
    df.to_sql(database_filename.split(".")[0], engine, index=False, if_exists = 'replace')

    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df, categories_expanded = load_data(messages_filepath, categories_filepath)
        df.to_csv('test.csv')

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()