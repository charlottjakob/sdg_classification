
# source for initial code: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py#L809

#!/usr/bin/env python3
import warnings
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from re import finditer
import fitz
from pprint import pprint
import colorsys

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

sdg_colors_hsl = {
    1: {"h": 353, "s": 79, "l": 52},
    2: {"h": 40, "s": 71, "l": 55},
    3: {"h": 108, "s": 48, "l": 42},
    4: {"h": 353, "s": 77, "l": 44},
    5:  {"h": 7, "s": 100, "l": 56},
    6: {"h": 192, "s": 76, "l": 52},
    7: {"h": 46, "s": 98, "l": 52},
    8: {"h": 342, "s": 73, "l": 37},
    9: {"h": 19, "s": 98, "l": 57},
    10: {"h": 335, "s": 84, "l": 47},
    11: {"h": 33, "s": 98, "l": 57},
    12: {"h": 38, "s": 61, "l": 46},
    13: {"h": 125, "s": 33, "l": 37},
    14: {"h": 199, "s": 91, "l": 45},
    15: {"h": 103, "s": 63, "l": 46},
    16: {"h": 200, "s": 100, "l": 31},
    17: {"h": 205, "s": 62, "l": 26},
}

sdg_strings = [str(sdg) if sdg >= 10 else '0' + str(sdg) for sdg in range(1, 18)]
sdg_paths = [f'/Users/charlott/Documents/github_repos/sdg_classification/data/SDG_icons_PRINT/E_SDG_PRINT-{sdg_str}.jpg' for sdg_str in sdg_strings]


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_class",
        "attr_score",
        "raw_input_ids",
        "convergence_score",
        "sdg_idx"
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input_ids,
        convergence_score,
        sdg_idx
    ) -> None:
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input_ids = raw_input_ids
        self.convergence_score = convergence_score
        self.sdg_idx = sdg_idx

def _get_color(attr,sdg_idx):

    hsl = sdg_colors_hsl[sdg_idx + 1]
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0: # positiv
        hue = hsl['h']
        sat = hsl['s']
        lig = 100 - int((100 - hsl['l']) * attr) # 0 - 50 = schwarz bis farbe & 50 - 100 = farbe bis weiß
        # lig = 100 - int(50 * attr) 
    else:
        hue = 0
        sat = 75
        lig = 100 # don't show negative attention -> always white
        # lig = 100 - int(-40 * attr)

    return "hsl({}, {}%, {}%)".format(hue, sat, lig)



def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )


def format_word_importances(words, importances,sdg_idx):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance, sdg_idx)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )

    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Actual</th>"
        "<th>Predicted Label</th>"
        "<th>Word Importance</th>"
    ]

    for datarecord in datarecords:
        gray_scale = 'style="filter: grayscale(100%)"' if datarecord.true_class == 0 else ''
        rows.append(
            "".join(
                [
                    "<tr>",
                    f'<td><img src={sdg_paths[datarecord.sdg_idx]} alt="sdg_logo" width="50" ' + gray_scale +'></td>',
                    
                    '<td><text style="padding-right:1em">{0} ({1:.2f})</text></td>'.format(
                            datarecord.pred_class, datarecord.pred_prob
                    ),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions, datarecord.sdg_idx
                    ),
                    "<tr>",
                ]
            )
        )


    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html


def add_word_highlight(page, words_clean, word_attrs, word_idx, word_sdg):

    word = words_clean[word_idx]
    attr = word_attrs[word_idx]
    
    try: 
        rl = page.search_for(word)
        if len(rl) != 0:
            if len(rl) > 1:
                
                character_where_word_starts = len(" ".join(words_clean[:word_idx]))
                words_clean_joined = " ".join(words_clean).lower()
                

                occurence_wanted = False
                for match_idx, match in enumerate(finditer(word, words_clean_joined)):
                    if match.span()[0] >= character_where_word_starts:
                        occurence_wanted = match_idx
                        break

                # check how often does i occure before in words
                # occurence_wanted = words_clean[:word_idx].count(word)
                if occurence_wanted and len(rl) > occurence_wanted:
                    clip = rl[occurence_wanted]
                else:
                    clip = rl[-1]

            else:
                clip = rl[0]
            

            # extract text info now - before the redacting removes it.
            blocks = page.get_text("dict", clip=clip)["blocks"]
            span = blocks[0]["lines"][0]["spans"][0]
            # assert span["text"] == word

            sdg_hsl = sdg_colors_hsl[word_sdg]

            l_attr = 100 - int((100 - sdg_hsl['l']) * attr)
            h_max, s_max, l_max = 360, 100, 100

            r_scaled, g_scaled, b_scaled = colorsys.hls_to_rgb(sdg_hsl['h']/h_max, l_attr/l_max, sdg_hsl['s']/s_max)

            highlight = page.add_highlight_annot(clip=clip)
            highlight.set_colors(stroke=[r_scaled, g_scaled, b_scaled]) # light red color (r, g, b) (76, 159, 56) / 255 = ()
            highlight.update()
    
    except Exception as e:
        print(e)


def visualize_pdf(
    datarecords: Iterable[VisualizationDataRecord], sr_id, page_number
):
    # find pdf and open it
    sr_path = '/Users/charlott/Nextcloud/Documents/SRs/' + str(sr_id)

    # sr_path = 'https://tubcloud.tu-berlin.de/index.php/f/' + str(sr_id)

    doc = fitz.open(sr_path)
    page = doc[int(page_number)-1]

    # datarecord.raw_input_ids, datarecord.word_attributions, datarecord.sdg_idx
    for datarecord in datarecords:
        
        words = datarecord.raw_input_ids
        word_attrs = datarecord.word_attributions
        sdg = datarecord.sdg_idx + 1

        # filter for words with positive attributions & words
        word_idx_positives = [ i for i in range(len(word_attrs)) if word_attrs[i] > 0]

        # remove "##" from splitted words
        words_clean = [word.replace("##","") for word in words]

        for word_idx in word_idx_positives:
            add_word_highlight(page, words_clean, word_attrs, word_idx, sdg)
            
        rotate = int(0)
        # PDF Page is converted into a whole picture 1056*816 and then for each picture a screenshot is taken.
        # zoom = 1.33333333 -----> Image size = 1056*816
        # zoom = 2 ---> 2 * Default Resolution (text is clear, image text is hard to read)    = filesize small / Image size = 1584*1224
        # zoom = 4 ---> 4 * Default Resolution (text is clear, image text is barely readable) = filesize large
        # zoom = 8 ---> 8 * Default Resolution (text is clear, image text is readable) = filesize large
        zoom_x = 2
        zoom_y = 2
        # The zoom factor is equal to 2 in order to make text clear
        # Pre-rotate is to rotate if needed.
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        output_file = f"test_page_sdg_{str(datarecord.sdg_idx + 1)}.png"
        pix.save(output_file)

        return output_file


def visualize_pdf_all_sdgs(
    datarecords: Iterable[VisualizationDataRecord], sr_id, page_number
):
    # find pdf and open it
    sr_path = '/Users/charlott/Nextcloud/Documents/SRs/' + str(sr_id)

    # sr_path = 'https://tubcloud.tu-berlin.de/index.php/f/' + str(sr_id)

    doc = fitz.open(sr_path)
    page = doc[int(page_number) - 1]


    # create single datarrecord where each word gets the highest attr between sdgs

    # create initial matrix
    words = datarecords[0].raw_input_ids    


    attr_matrix = []
    for datarecord in datarecords:

        attr_matrix.append(datarecord.word_attributions.tolist())

    attr_matrix = np.array(attr_matrix)
    
    word_attrs = attr_matrix.max(axis=0)
    word_sdgs = attr_matrix.argmax(axis=0) + 1

    # filter for words with positive attributions & words
    word_idx_positives = [ i for i in range(len(word_attrs)) if word_attrs[i] > 0]

    # remove "##" from splitted words
    words_clean = [word.replace("##","") for word in words]

    for word_idx in word_idx_positives:
        add_word_highlight(page, words_clean, word_attrs, word_idx, word_sdgs[word_idx])
        
    rotate = int(0)
    # PDF Page is converted into a whole picture 1056*816 and then for each picture a screenshot is taken.
    # zoom = 1.33333333 -----> Image size = 1056*816
    # zoom = 2 ---> 2 * Default Resolution (text is clear, image text is hard to read)    = filesize small / Image size = 1584*1224
    # zoom = 4 ---> 4 * Default Resolution (text is clear, image text is barely readable) = filesize large
    # zoom = 8 ---> 8 * Default Resolution (text is clear, image text is readable) = filesize large
    zoom_x = 2
    zoom_y = 2
    # The zoom factor is equal to 2 in order to make text clear
    # Pre-rotate is to rotate if needed.
    mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    output_file = f"explainability/test_page_all_sdgs.png"
    pix.save(output_file)

    dom = [""]

    # add SDG Icons
    sdg_icons = []
    for datarecord in datarecords:
        gray_scale = 'style="filter: grayscale(100%)"' if datarecord.pred_class == 0 else ''
        sdg_icons.append(
            f'<img src="{sdg_paths[datarecord.sdg_idx]}" alt="sdg_logo" width="50" ' + gray_scale +'>'
        )
    dom.append("".join(sdg_icons))

    pdf_path = "/Users/charlott/Documents/github_repos/sdg_classification/explainability/test_page_all_sdgs.png"
    dom.append(
        "<img src='"+ pdf_path +"' alt='pdf_page' height='800' style='border: 1px solid black; margin: 10px;'>"
    )

    # Add PDF
    html = HTML("".join(dom))
    display(html)

    return html


