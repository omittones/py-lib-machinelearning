# -*- coding: utf-8 -*-

#from pandas_datareader import data as pdr
import logging
import pandas as pd
import os


def get_data_for_one_security(ticker, function, isfromfile):
        
    if(isfromfile):
        df=get_data_from_csv(ticker, function)
        return df
    else:
        df=get_data_from_external_feed(ticker, function)
        return df

    return
    
def get_data_from_external_feed(ticker, function):
    
    logging.debug('Trying to get data from AlphaVantage')
    
    try:        
        print(ticker)
        
        url_base="https://www.alphavantage.co/query?apikey=GXTGTQUF98ZBL1OX&datatype=csv&outputsize=full"
        a_from_symbol=ticker[0:3]
        a_to_symbol=ticker[3:6]
        
        a_function=function
        url=url_base+"&function="+a_function + "&from_symbol="+a_from_symbol+"&to_symbol="+a_to_symbol    
        df=pd.read_csv(url, index_col='timestamp', parse_dates=True)        
        df=df.iloc[::-1]
    except Exception:
        print("Error in data fetching from AlphaVantage for ticker: ", ticker)
        return None
        
        
    filename="./LocalDB/AlphaVantage/"+ticker+"_"+a_function+".csv"
    df.to_csv(filename, sep=',')
    
    logging.info('Received and stored data for '+ticker)
               
    return df
    
    

def get_data_from_csv(ticker, function):
    
    filepath="./LocalDB/AlphaVantage/"+ticker+"_"+function+".csv"
    
    if not os.path.exists(filepath):
        print("File does not exist: ", filepath)        
        return None
    
    
    df = pd.read_csv(filepath, sep=',', index_col='timestamp', parse_dates=True)
    return df
    

    
    
# MAIN

def main():
  print("In alphavantagedatareader main function...")  

if __name__== "__main__":
    main()

logging.basicConfig(filename='my.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

#start = datetime.datetime(1900, 1, 1)
#end = datetime.datetime(2016, 12, 16)
#feed="yahoo"
#ticker="C"
#df=get_data_for_one_security(feed, ticker, start, end)
#print(df)


