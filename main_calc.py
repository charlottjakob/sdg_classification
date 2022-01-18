import pandas as pd
import json
from training_data.utils.build_basic_dict import build_basic_dict
import matplotlib.pyplot as plt
import os


# basic_df = build_basic_dict()

scholar_data = pd.read_csv("files/scholar_data.csv")
scholar_data = scholar_data[scholar_data['pdf_text'].notna()]
paper_count = scholar_data.groupby('goal').count().reset_index()


print(paper_count)