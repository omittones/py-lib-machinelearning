# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:07:11 2019

@author: curakiva
"""

import numpy as np
import mass_ts as mts
#print (np.__version__)

from matplotlib import pyplot as plt

#ts_for_search = np.loadtxt('ts_for_search.txt')
#ts_query = np.loadtxt('carpet_query.txt')

###############
import alphavantagedatareader as avdr

ticker_for_query="GBPUSD"
query_pos_start=400
query_length=100
df_for_query=avdr.get_data_for_one_security(ticker_for_query, "FX_DAILY", True)
print(df_for_query.shape)
ts_query=df_for_query['close'][query_pos_start:query_pos_start+query_length].values

ticker="EURUSD"
df=avdr.get_data_for_one_security(ticker, "FX_DAILY", True)
#ts=df['close'][1:1000].values
ts_for_search=df['close'].values
print(ts_for_search)

#########################

plt.figure(figsize=(20,5))
plt.plot(np.arange(len(ts_for_search)), ts_for_search)
plt.ylabel('Price')
plt.title('For search: '+ticker)
plt.show()


#plt.figure(figsize=(25,5))
#plt.plot(np.arange(len(ts_query)), ts_query)
#plt.ylabel('Price')
#plt.title('Query')
#plt.show()

#
#distances = mts.mass3(ts_for_search, ts_query, 256)
distances = mts.mass2(ts_for_search, ts_query)
print(len(distances))
#print(distances[0], distances[1])
found = mts.top_k_motifs(distances, 5, 25)
found = np.array(found)
print(found)

#distances=np.sort_complex(distances)
#print(distances[2])

min_idx = np.argmin(distances)
print(min_idx)


plt.figure(figsize=(20,5))
plt.plot(np.arange(len(ts_for_search)), ts_for_search)
plt.plot(np.arange(min_idx, min_idx + query_length), ts_query, c='r')
plt.ylabel('Price')
plt.title('For search: '+ticker)
plt.show()


fig, (ax1, ax3) = plt.subplots(2,1,sharex=True,figsize=(10,10))
ax1.plot(np.arange(len(ts_query)), ts_query)
#ax1.set_ylabel('Query', size=12)
ax1.set_title('Query', size=12)

# Plot our best match
ax3.plot(np.arange(query_length), ts_for_search[min_idx:min_idx+query_length])
#ax3.set_ylabel('Best Match', size=12)
ax3.set_title('Best Match', size=12)

plt.show()
