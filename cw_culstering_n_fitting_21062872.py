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
import scipy.optimize as opt

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

#Select few countries representing all continets from the dataset
countries = ['United Kingdom', 'India', 'Japan', 'China', 'Korea, Rep.',
             'South Africa', 'United States', 'Korea, Rep.', 'Germany']
df_countries = df_climt_chg[df_climt_chg['Country Name'].isin(countries)]

#Select records for specific indicators
indecators = [ 'Urban population (% of total population)' \
                ,'Urban population growth (annual %)' \
                ,'Population growth (annual %)' \
                ,'Mortality rate, under-5 (per 1,000 live births)' \
                ,'School enrollment, primary and secondary (gross)' \
                ,'gender parity index (GPI)' \
                ,'Agriculture, forestry, and fishing, value added (% of GDP)' \
                ,'Marine protected areas (% of territorial waters)' \
                ,'Urban population living in areas where elevation is below 5 meters (% of total population)' \
                ,'Rural population living in areas where elevation is below 5 meters (% of total population)' \
                ,'Nitrous oxide emissions (% change from 1990)' \
                ,'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)' \
                ,'Methane emissions (% change from 1990)' \
                ,'Total greenhouse gas emissions (% change from 1990)' \
                ,'Other greenhouse gas emissions (% change from 1990)' \
                ,'CO2 emissions (kg per PPP $ of GDP)' \
                ,'CO2 emissions (metric tons per capita)' \
                ,'CO2 emissions from gaseous fuel consumption (% of total)' \
                ,'Electric power consumption (kWh per capita)' \
                ,'Renewable energy consumption (% of total final energy consumption)' \
                ,'Electricity production from renewable sources, excluding hydroelectric (% of total)' \
                ,'Renewable electricity output (% of total electricity output)' \
                ,'Electricity production from oil sources (% of total)' \
                ,'Electricity production from nuclear sources (% of total)' \
                ,'Electricity production from natural gas sources (% of total)' \
                ,'Electricity production from hydroelectric sources (% of total)' \
                ,'Electricity production from coal sources (% of total)' \
                ,'Access to electricity (% of population)' \
                ,'Foreign direct investment, net inflows (% of GDP)' \
                ,'Cereal yield (kg per hectare)' \
                ,'Average precipitation in depth (mm per year)' \
                ,'Agricultural irrigated land (% of total agricultural land)' \
                ,'Forest area (% of land area)' \
                ,'Arable land (% of land area)' \
                ,'Agricultural land (% of land area)' \
                ,'Urban population (% of total population)' \
                ,'Urban population growth (annual %)'
            ]
df_cntry_ind = df_countries[df_countries['Indicator Name'].isin(indecators)]

#Select only columns after year 1990
df_cntry_yrs = df_cntry_ind.loc[:,['Country Name', 'Indicator Name', '1990', '1991', '1992',	
                                    '1993', '1994', '1995', '1996', '1997', '1998', '1999',
                                    '2000', '2001', '2002', '2003', '2004', '2005', '2006',
                                    '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                                    '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

#Check for null values
count_nan = df_cntry_yrs.isna().sum()
#Replace null values with row mean
df_cntry_yrs['raw_avg'] = df_cntry_yrs[['1990', '1991', '1992',	'1993', '1994', '1995', '1996', 
                                        '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
                                        '2004', '2005', '2006','2007', '2008', '2009', '2010', '2011', '2012', '2013',
                                        '2014', '2015', '2016', '2017', '2018', '2019', '2020']].mean(axis=1)

cols = ['1990', '1991', '1992',	'1993', '1994', '1995', '1996', 
        '1997', '1998', '1999','2000', '2001', '2002', '2003', 
        '2004', '2005', '2006','2007', '2008', '2009', '2010',
        '2011', '2012', '2013','2014', '2015', '2016', '2017', 
        '2018', '2019', '2020']

for i in cols:
    df_cntry_yrs[i].fillna(df_cntry_yrs['raw_avg'], inplace=True)
