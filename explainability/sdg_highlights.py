
from utils.layer_attribution import AttentionClass
from utils import visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import visualization as viz
from training import BERTClassifier

class_names = [str(number) for number in np.arange(1, 18)]


if __name__ == '__main__':


    sr_data = pd.read_csv("data/sr_data_downloaded.csv")

    # Take random sample between certain word amount
    # sr_data = sr_data[(sr_data['page_text_word_count'] <= 517) & (sr_data['page_text_word_count'] >= 200)]
    # sr_page = sr_data.sample().reset_index().loc[0]

    # Define expamle to be checked
    sr_page = sr_data[(sr_data['_id'] == 1496216090) & (sr_data['page_number'] == 31)].reset_index().loc[0]

    # 1558445799 32
    # 1558445799 50
    # 1496216090 31
    # 1558445799 18


    text = sr_page['page_text']
    true_labels = list(sr_page[class_names])

    vis_list = []
    for sdg_idx, true_label in enumerate(true_labels[:17]):

        att = AttentionClass(sdg_idx=sdg_idx)

        all_tokens, attributions_sum, delta, scores = att.run_attention_extraction(text=text, sdg_idx=sdg_idx)

        # storing couple samples in an array for visualization purposes
        score_vis = visualization.VisualizationDataRecord(attributions_sum,
                                                torch.sigmoid(scores)[sdg_idx],
                                                torch.sigmoid(scores)[sdg_idx] > 0.5,
                                                true_label,
                                                text,
                                                attributions_sum.sum(),       
                                                all_tokens,
                                                delta,
                                                sdg_idx)

        vis_list.append(score_vis)


    print('\033[1m', 'Visualization For Score', '\033[0m')
    html_object_text = visualization.visualize_text(vis_list)

    with open('explainability/html_attr_text.html', 'w') as f:
        f.write(html_object_text.data)

    # image = visualization.visualize_pdf(vis_list, sr_id=sr_page['_id'], page_number=sr_page['page_number'])
    html_object_pdf = visualization.visualize_pdf_all_sdgs(vis_list, sr_id=sr_page['_id'], page_number=sr_page['page_number'])
    
    with open('explainability/html_attr_pdf.html', 'w') as f:
        f.write(html_object_pdf.data)
    
    print("report_id: ", sr_page['_id'], " page_number: ", sr_page['page_number'])