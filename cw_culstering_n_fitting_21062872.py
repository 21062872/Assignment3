# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:47:37 2023

@author: TharindaArachchi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import scipy.stats as stats

def read_external_files(filename):
    '''
    Read an external file and load into a dataframe, create another dataframe 
    by transposing original one

    Parameters
    ----------
    filename : external file name with extension

    Returns
    -------
    a dataframe and it's transpose

    '''
    #Look for the extension of the file 
    splitFileName = os.path.splitext(filename)
    fileExtension = splitFileName[1]
    
    #World bank data is in csv, excel formats. Based on those formats reading the files into dataframe
    if (fileExtension == '.csv'):
        df_climt_chg = pd.read_csv(filename, skiprows=4)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    elif (fileExtension == '.xls'):
        df_climt_chg = pd.read_excel(filename, skiprows=3)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    else:
        raise Exception("Invalid File Format")
    return df_climt_chg, df_climt_chg_tp

#Executing the function to load external file to dataframe     
df_climt_chg, df_climt_chg_tp = read_external_files('API_19_DS2_en_csv_v2_4773766.csv')  