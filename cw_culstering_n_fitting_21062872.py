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
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn import preprocessing

def exp_growth(t, scale, growth):
    '''
    Computes exponential function with scale and growth

    '''
    f = scale * np.exp(growth * (t-1950))
    return f

def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth and time of 
        the turning point
    
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


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

def norm_df_sk(df):
    '''
    Normalize dataframe values

    Parameters
    ----------
    df : dataframe object

    Returns
    -------
    df_merged3_norm : normalized dataframe

    '''
    x = df_merged3.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_merged3_norm = pd.DataFrame(x_scaled)
    return df_merged3_norm

def norm(df):
    '''
    Normalize dataframe values

    Parameters
    ----------
    df : dataframe object

    Returns
    -------
    df_merged3_norm : normalized dataframe

    '''
    min_val = np.min(df)
    max_val = np.max(df)
    df_merged3_norm = (df-min_val) / (max_val-min_val)
    return df_merged3_norm

def norm_df(df, first=0, last=None):
    '''
    Normalize dataframe values

    Parameters
    ----------
    df : dataframe object

    Returns
    -------
    df_merged3_norm : normalized dataframe

    '''    
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df    

#Executing the function to load external file to dataframe     
df_climt_chg, df_climt_chg_tp = read_external_files('API_19_DS2_en_csv_v2_4773766.csv')  

#Select few countries representing all continets from the dataset
countries = ['United Kingdom', 'India', 'Japan', 'China', 'Korea, Rep.',
             'South Africa', 'United States', 'Korea, Rep.', 'Germany']
df_countries = df_climt_chg[df_climt_chg['Country Name'].isin(countries)]

#Select records for specific indicators
indecators = [ 'Urban population (% of total population)' \
                ,'Urban population growth (annual %)' \
                ,'Urban population' \
                ,'Population growth (annual %)' \
                ,'Population, total' \
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
                ,'Agricultural land (% of land area)'
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

#Heatmap of data before removing null values
sb.heatmap(df_cntry_yrs.isnull())

for i in cols:
    df_cntry_yrs[i].fillna(df_cntry_yrs['raw_avg'], inplace=True)


#Filtering for UK's population growth over time series
df_pop_growth = df_cntry_yrs[df_cntry_yrs['Indicator Name'] == 'Population, total']
df_pop_growth = df_pop_growth[df_pop_growth['Country Name'] == 'United Kingdom']
df_pop_growth = df_pop_growth.loc[:,['Country Name', '1990', '1991', '1992',	'1993', '1994', '1995', '1996', 
                                    '1997', '1998', '1999','2000', '2001', '2002', '2003', 
                                    '2004', '2005', '2006','2007', '2008', '2009', '2010',
                                    '2011', '2012', '2013','2014', '2015', '2016', '2017', 
                                    '2018', '2019', '2020']]
df_pop_growth_tp = df_pop_growth.set_index('Country Name').transpose()
df_pop_growth_tp['Year'] = df_pop_growth_tp.index.astype(str).astype(int)

# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, df_pop_growth_tp["Year"],
                                      df_pop_growth_tp["United Kingdom"])
print('Fit Parameter is : ', popt)

# plot first fit attempt
df_pop_growth_tp["pop_exp"] = exp_growth(df_pop_growth_tp["Year"], *popt)
plt.figure()
plt.style.use('ggplot')
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"], label="data")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["pop_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("Total Population")
plt.show()
print()

# fit exponential growth giving a smaller expo factor 
popt, covar = opt.curve_fit(exp_growth, df_pop_growth_tp["Year"],
                            df_pop_growth_tp["United Kingdom"], p0=[57247586.0, 0.01])
# much better
print("Fit parameter", popt)
df_pop_growth_tp["pop_exp"] = exp_growth(df_pop_growth_tp["Year"], *popt)
plt.figure()
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"], label="data")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Final fit exponential growth")
plt.show()
print()
print('Fit Parameter after p0 suggestion is : ', popt)
'''
#finding initial approx. of Logistict function
popt = [57247586, 0.01, 1990]
df_pop_growth_tp["pop_log"] = logistics(df_pop_growth_tp["Year"], *popt)
plt.figure()
plt.style.use("seaborn")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"], label="data")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Improved start value")
plt.show()
print('Fit Parameter (Logistic is) : ', popt)

popt, covar = opt.curve_fit(logistics, df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"],
p0=(1.17898243e+05, 1, 1990.0), maxfev=5000)
print("Fit parameter", popt)
df_pop_growth_tp["pop_log"] = logistics(df_pop_growth_tp["Year"], *popt)
plt.figure()
plt.title("logistics function")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"], label="data")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()
print('Fit Parameter (Logistic is) : ', popt)
print('Covariance is ', covar)
'''
# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err_ranges(df_pop_growth_tp["Year"], exp_growth, popt, sigma)
plt.figure()
plt.title("logistics functions")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["United Kingdom"], label="data")
plt.plot(df_pop_growth_tp["Year"], df_pop_growth_tp["pop_exp"], label="fit")
plt.fill_between(df_pop_growth_tp["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()

#Giving ranges for predictions
print("Forcasted population")
low, up = err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err_ranges(2050, exp_growth, popt, sigma)
print("2050 between ", low, "and", up)

#Clustering
indecators_class = ['Population, total', 'CO2 emissions (kg per PPP $ of GDP)', 'Electric power consumption (kWh per capita)', 'Mortality rate, under-5 (per 1,000 live births)']
df_classif_pop = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='Population, total']
df_classif_co2 = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='CO2 emissions (kg per PPP $ of GDP)']
df_classif_elc = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='Electric power consumption (kWh per capita)']
df_classif_mr = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='Mortality rate, under-5 (per 1,000 live births)']
#Filtering to get the data for year 2020
df_classif_pop = df_classif_pop.loc[:,['Country Name', '2020']]
df_classif_pop.rename({'2020': 'Population, total 2020'}, axis=1, inplace=True)
df_classif_co2 = df_classif_co2.loc[:,['Country Name', '2020']]
df_classif_co2.rename({'2020': 'CO2 emissions 2020'}, axis=1, inplace=True)
df_classif_elc = df_classif_elc.loc[:,['Country Name', '2020']]
df_classif_elc.rename({'2020': 'Electric power consumption 2020'}, axis=1, inplace=True)
df_classif_mr = df_classif_mr.loc[:,['Country Name', '2020']]
df_classif_mr.rename({'2020': 'Mortality rate, under-5 2020'}, axis=1, inplace=True)
#Merging 2 dataframes together
df_merged = pd.merge(df_classif_pop, df_classif_co2, on='Country Name', how='outer')
df_merged2 = pd.merge(df_merged, df_classif_elc, on='Country Name', how='outer')
df_merged3 = pd.merge(df_merged2, df_classif_mr, on='Country Name', how='outer')
df_merged3 = df_merged3.loc[:,['CO2 emissions 2020', 'Population, total 2020', 'Electric power consumption 2020', 'Mortality rate, under-5 2020']]

#Normalize data in dataframe
df_merged3_norm = norm_df(df_merged3)

pd.plotting.scatter_matrix(df_merged3_norm, figsize=(9.0, 9.0))
plt.tight_layout() # helps to avoid overlap of labels
plt.show()

# extract columns for fitting
df_fit = df_merged3[['CO2 emissions 2020', 'Population, total 2020']].copy()
# normalise dataframe and inspect result
df_fit = norm_df(df_fit)
print(df_fit.describe())
print()

for n in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(df_fit)

    labels = kmeans.labels_
    print (n, skmet.silhouette_score(df_fit, labels))

#Good results for 2nd cluster

#Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)   
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))



plt.scatter(df_fit['CO2 emissions 2020'],df_fit['Population, total 2020'], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("CO2 emissions 2020")
plt.ylabel("Population, total 2020")
plt.title("2 clusters")
plt.show()

#Find the clusters of below indicators
df_clss_elec = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='Electric power consumption (kWh per capita)']
df_classif_re = df_cntry_yrs[df_cntry_yrs['Indicator Name']=='Electricity production from renewable sources, excluding hydroelectric (% of total)']
#Filtering to get the data for year 2020
df_clss_elec = df_clss_elec.loc[:,['Country Name', 'raw_avg']]
df_clss_elec.rename({'raw_avg': 'Electric power consumption (kWh per capita)'}, axis=1, inplace=True)
df_classif_re = df_classif_re.loc[:,['Country Name', 'raw_avg']]
df_classif_re.rename({'raw_avg': 'Electricity production from renewable sources, excluding hydroelectric (% of total)'}, axis=1, inplace=True)
#Merging 2 dataframes together
df_merged4 = pd.merge(df_clss_elec, df_classif_re, on='Country Name', how='outer')

# extract columns for fitting
df_fit2 = df_merged4[['Electric power consumption (kWh per capita)', 'Electricity production from renewable sources, excluding hydroelectric (% of total)']].copy()
# normalise dataframe and inspect result
df_fit2 = norm_df(df_fit2)
print(df_fit2.describe())
print()

for n in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(df_fit2)

    labels = kmeans.labels_
    print (n, skmet.silhouette_score(df_fit2, labels))

#Good results for 2nd cluster

#Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)   
kmeans.fit(df_fit2)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_fit2['Electric power consumption (kWh per capita)'],df_fit2['Electricity production from renewable sources, excluding hydroelectric (% of total)'], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Electric power consumption (kWh per capita) avg per year")
plt.ylabel("Electricity production from renewable sources, excluding hydroelectric (% of total)")
plt.title("2 clusters")
plt.show()

''' Plot: 1
    Plot type: Bar chart
    Plot name: Urban Population Over Time'''
#Replace NaN values with preceding row value
df_cntry_yrs = df_cntry_yrs.fillna(method='ffill', axis=1)

#Select data in 5 year intervals 
df_cntry_ind_hdr = df_cntry_yrs.iloc[:,[0,1]]
df_yrs_five_intr = df_cntry_yrs.iloc[:,2::5]
df_fnl = pd.concat([df_cntry_ind_hdr,df_yrs_five_intr], axis=1)

#Rename countries with shortnames for better clarity of labels
df_fnl.replace("United Kingdom", "UK", inplace=True)
df_fnl.replace("United States", "USA", inplace=True)
df_fnl.replace("South Africa", "SA", inplace=True)
df_fnl.replace("Korea, Rep.", "Korea", inplace=True)
    
#Plot bar chart against indicator Urban population
df_urbn_pop = df_fnl[df_fnl['Indicator Name'] == 'Urban population']

# plotting graph
plt.figure(figsize=(8, 6), dpi=80)
plt.style.use('ggplot')
# plot grouped bar chart
df_urbn_pop.plot(x='Country Name',
                kind='bar',
                stacked=False,
                title='Urban Population Over Time')
# labeling the graph
plt.xlabel('Country')
plt.ylabel('Number of population')

plt.legend(title ="Years")
plt.show()

''' Plot: 7
    Plot type: correlation heatmap
    Plot name: correlation heatmap of United Kingdom'''
df_cor = df_cntry_yrs[df_cntry_yrs['Country Name'] == 'United Kingdom']
indecators_cls = ['Electric power consumption (kWh per capita)'
,'CO2 emissions (kg per PPP $ of GDP)'
,'Population growth (annual %)'
,'gender parity index (GPI)'
,'Agriculture, forestry, and fishing, value added (% of GDP)'
,'Electricity production from oil sources (% of total)'
,'Renewable electricity output (% of total electricity output)'
,'Agricultural land (% of land area)']
df_cor = df_cor[df_cor['Indicator Name'].isin(indecators_cls)]
df_cor2 = df_cor.loc[:,['Indicator Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
df_cor3 = df_cor2.set_index('Indicator Name').transpose()

df_cor3 = df_cor3.astype(str).astype(float)    
print(df_cor3.corr())
#print (df_cor3.dtypes)

# plotting correlation heatmap
dataplot = sb.heatmap(df_cor3.corr(), cmap="viridis", annot=True)
# displaying heatmap
plt.show()