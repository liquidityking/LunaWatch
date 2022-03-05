from cryptocmd import CmcScraper
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st


def LUNA_Price():
    # initialise scraper with time interval
    scraper = CmcScraper("LUNA", "01-01-2021", "25-10-2030")

    # get raw data as list of list
    headers, data = scraper.get_data()

    # get data in a json format
    json_data = scraper.get_data("json")

    # export the data to csv
    scraper.export("csv")

    # get dataframe for the data
    df = scraper.get_dataframe("Close")

    return (df.get("Close"))

##print(LUNA_Price())

def UST_Cap():
    # initialise scraper with time interval
    scraper = CmcScraper("UST", "01-01-2021", "25-10-2030")

    # get raw data as list of list
    headers, data = scraper.get_data()

    # get data in a json format
    json_data = scraper.get_data("json")

    # export the data to csv
    scraper.export("csv")

    # get dataframe for the data
    df = scraper.get_dataframe("Close")

    return (df.get("Market Cap"))

price = (LUNA_Price())[::-1]
log_price = np.log(LUNA_Price())[::-1]
log_cap = np.log(UST_Cap())[::-1]


results=[]
price_list=[]
std_list=[]

data_length=len(log_price)-1

i=3
ii=0


while ii <= data_length:
    m, b = np.polyfit(log_cap.iloc[0:i][:], log_price.iloc[0:i][:], 1)
    point=np.exp((m*log_cap[0:i]+b))[ii]
    results.append(point)

    std_point=np.std(log_price[0:i][:])
    std_list.append(np.exp(std_point))
    
    i=i+1
    ii=ii+1

high_price = [i *2.5 for i in results]
low_price = [i*(1/2.5) for i in results]


date_index = log_price.index

date_list = date_index.tolist()




plt.style.use('dark_background')

ax = plt.gca()
ax.set_yscale('log')



plt.plot(date_list,high_price,color='tomato',alpha=0.25,linewidth=2)
plt.plot(date_list,results,color='gold',alpha=0.25,linewidth=2)
plt.plot(date_list,low_price,color='limegreen',alpha=0.25,linewidth=2)

plt.plot(date_list,high_price,color='tomato',alpha=0.8,linewidth=0.5)
plt.plot(date_list,results,color='gold',alpha=0.8,linewidth=0.5)
plt.plot(date_list,low_price,color='limegreen',alpha=0.8,linewidth=0.5)

plt.plot(date_list,(np.exp(log_price)).values,color='skyblue',linewidth=8,alpha=0.1)
plt.plot(date_list,(np.exp(log_price)).values,color='skyblue',linewidth=5,alpha=0.25)
plt.plot(date_list,(np.exp(log_price)).values,color='skyblue',linewidth=1.5,alpha=0.5)
plt.plot(date_list,(np.exp(log_price)).values,color='skyblue',linewidth=0.5)

plt.yscale("log")


from matplotlib.ticker import FormatStrFormatter





plt.grid(visible=True, which='minor', color='b', linestyle='--',alpha=0.1)
plt.grid(axis='y')

plt.tick_params(axis='y', which='minor')


from matplotlib.ticker import FormatStrFormatter
ax = plt.gca()

ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
ax.tick_params(axis='y', size=4)

plt.grid(True, which="both", axis='y',alpha=0.15,color="white")
plt.grid(True, which="both", axis='x',alpha=0.1,color="white")

plt.xticks(fontsize=8)

ax.tick_params(axis='y', which='major', labelsize=8,pad=12)
ax.tick_params(axis='both', which='minor', labelsize=5)

##plt.title("Value Bands",fontsize=14)

from matplotlib.ticker import FuncFormatter
def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.1f' % x

scientific_formatter = FuncFormatter(scientific)
ax.yaxis.set_major_formatter(scientific_formatter)

plt.savefig('chart.png',dpi=1000)

plt.show()

from PIL import Image

image = Image.open('chart.png')
image.show()


