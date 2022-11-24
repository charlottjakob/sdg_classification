"""utils for sdg-data."""
import numpy as np

class_names = [str(number) for number in np.arange(1, 18)]

sdg_short_description = {
    '1': 'no poverty',
    '2': 'zero hunger',
    '3': 'good health and well-being',
    '4': 'quality education',
    '5': 'gender equality',
    '6': 'clean water and sanitation',
    '7': 'affordable and clean energy',
    '8': 'decent work and economic growth',
    '9': 'industry, innovation and infrastructure',
    '10': 'reduce inequalities',
    '11': 'sustainable cities and communities',
    '12': 'responsible consumption and production',
    '13': 'climate action',
    '14': 'life below water',
    '15': 'life on land',
    '16': 'peace, justice and strong institutions',
    '17': 'partnerships for the goals'
}

sdg_colors = {
    1: "#e5243b",
    2: '#DDA63A',
    3: "#4C9F38",
    4: "#C5192D",
    5: "#FF3A21",
    6: "#26BDE2",
    7: "#FCC30B",
    8: "#A21942",
    9: "#FD6925",
    10: "#DD1367",
    11: "#FD9D24",
    12: "#BF8B2E",
    13: "#3F7E44",
    14: "#0A97D9",
    15: "#56C02B",
    16: "#00689D",
    17: "#19486A"
}
