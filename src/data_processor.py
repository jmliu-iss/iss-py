import argparse
import numpy as np
import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def save_data(data_path, df):
    df.to_csv(data_path.replace('.csv','_processed.csv'), index=False)
    return None

def run(data_path):
    df = load_data(data_path)
    return df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    args = argparser.parse_args()
    run(args.data_path)