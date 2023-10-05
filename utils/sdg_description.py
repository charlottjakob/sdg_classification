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

sdg_long_description = {
    '1': 'End poverty in all its forms, everywhere.',
    '2': 'End hunger, achieve food security and improved nutrition, and promote sustainable agriculture.',
    '3': 'Ensure healthy lives and promote well-being for all, at all ages.',
    '4': 'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.',
    '5': 'Achieve gender equality and empower all women and girls.',
    '6': 'Ensure availability and sustainable management of water and sanitation for all.',
    '7': 'Ensure access to affordable, reliable, sustainable, and modern energy for all.',
    '8': 'Promote sustained, inclusive, and sustainable economic growth, full and productive employment, and decent work for all.',
    '9': 'Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation.',
    '10': 'Reduce inequality within and among countries.',
    '11': 'Make cities and human settlements inclusive, safe, resilient, and sustainable.',
    '12': 'Ensure sustainable consumption and production patterns.',
    '13': 'Take urgent action to combat climate change and its impacts.',
    '14': 'Conserve and sustainably use the oceans, seas, and marine resources for sustainable development.',
    '15': 'Protect, restore, and promote sustainable use of terrestrial ecosystems, manage forests, combat desertification and biodiversity loss, and halt and reverse land degradation.',
    '16': 'Promote peaceful and inclusive societies for sustainable development, provide access to justice for all, and build effective, accountable, and inclusive institutions.',
    '17': 'Strengthen the means of implementation and revitalize the global partnership for sustainable development.'
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
