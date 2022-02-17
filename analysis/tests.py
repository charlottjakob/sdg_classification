import pandas as pd
import os
import regex as re

values = [
    '{print:hello}Hallo ich bin Charlott.',
]

for value in values:
    strings = re.findall(r"[\s\,\p{L}]{10,}\s[\s\,\p{L}\'\"]*[.!?]", value)
    print(strings)