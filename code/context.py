import pathlib


def projectpath():
    return pathlib.Path(__file__).parents[1]

def pdsettings():
    import pandas as pd
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)