#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:14:11 2025

@author: vedant
"""
'''
Instructions on how to run this file
Step 1 - Download value factor from AQR for comparison:
    https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly
Step 2 - Change file paths where needed
Step 3 - Click Run All
Step 4 - Input your WRDS username and password
Step 5 - Done!
'''

import wrds
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%

wrds_db = wrds.Connection()

#%%
# Import data

sql_query = """
          SELECT gvkey, 
                  permno,
                  crsp_shrcd,
                  sic,
                  eom,
                  me,
                  ret,
                  ret_exc,
                  be_me,
                  book_equity AS be,
                  prc AS Price
        FROM contrib.global_factor
        WHERE common=1 
            and exch_main=1 
            and primary_sec=1 
            and obs_main=1 
            and excntry='USA' 
            and eom >= '1970-01-01'
            and eom <= '2024-12-31'
          """
data = wrds_db.raw_sql(sql_query)

# Store data
data.to_csv("/Users/vedant/Desktop/BK/Value_Factor/data.csv")

#%%

# Load data
data = pd.read_csv("/Users/vedant/Desktop/BK/Value_Factor/data.csv")

# Data preparation
data = data.dropna(subset=["permno","sic"])
data["sic"] = data["sic"].astype(int)
data = data[(data["crsp_shrcd"] == 10) | (data["crsp_shrcd"] == 11)]
data["permno"] = data["permno"].astype(int)

#%%
comp_data = pd.read_parquet("/Users/vedant/Desktop/BK/Value_Factor/book_equity_data.parquet")
comp_data["gvkey"] = comp_data["gvkey"].astype('int')

#%%
comp_data['eom'] = pd.to_datetime(comp_data['datadate'])

def fill_quarterly_values(group):
    group = group.sort_values('eom')

    group = group.set_index('eom')

    monthly_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='M')
  
    group_monthly = group.reindex(monthly_index)

    group_monthly['gvkey'] = group_monthly['gvkey'].ffill()
    group_monthly['book_equity'] = group_monthly['book_equity'].ffill()
    group_monthly = group_monthly.reset_index().rename(columns={'index': 'eom'})
    return group_monthly

comp_data= comp_data.groupby('gvkey').apply(fill_quarterly_values).reset_index(drop=True)

#%%
# Save it because the previous step takes a long time to run
comp_data.to_csv("/Users/vedant/Desktop/BK/Value_Factor/comp_data.csv")
#%%
# Read the above saved csv
comp_data = pd.read_csv("/Users/vedant/Desktop/BK/Value_Factor/comp_data.csv")

#%% Sort values
data.sort_values(by= "eom", inplace=True)
comp_data.sort_values(by = "eom", inplace=True)
#%%
comp_data["eom"] = comp_data["eom"].astype(str)
comp_data["gvkey"] = comp_data["gvkey"].astype(int)
#%%
merged = comp_data.merge(data,on=["gvkey","eom"], how="inner")
# merg["be"] = merg["book_equity"].fillna(merg["be"])
merged["be"] = merged["book_equity"]
merged.drop(columns=["book_equity","source"],inplace=True)
data = merged

#%% Preprocessing according to the paper

from dateutil.relativedelta import relativedelta

# Step 1: Set up
data['eom'] = pd.to_datetime(data['eom'])
data = data.sort_values(['permno', 'eom'])
data['tradeable'] = 1  # default to tradeable
data["isna"] = data["be"].isna()
data["isna_ret"] = data["ret"].isna()

# Step 2: Prepare for faster lookup
data.set_index(['permno', 'eom'], inplace=True)

# Step 3: Loop through permno groups and apply 6-month non-tradeable flags
for permno, group in tqdm(data.groupby(level=0), total=data.index.get_level_values(0).nunique(), desc="Processing permnos"):
    # Get the dates where be is missing
    na_dates = group[group['isna']].index.get_level_values(1)
    
    na_dates_ret = group[group['isna']].index.get_level_values(1)
    
    # For each date, generate the next 6 calendar months and mark them non-tradeable
    for date in na_dates:
        for i in range(1, 7):  # 1 to 6 months ahead
            future_date = (date + relativedelta(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            if (permno, future_date) in data.index:
                data.loc[(permno, future_date), 'tradeable'] = 0
    
    # For each date, generate the next 12 calendar months and mark them non-tradeable
    for date in na_dates_ret:
        for i in range(1, 13):  # 1 to 12 months ahead
            future_date = (date + relativedelta(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            if (permno, future_date) in data.index:
                data.loc[(permno, future_date), 'tradeable'] = 0
    
data = data.reset_index()

#%%
# Remove ADRs, REITs, foreign shares, etc based on sic
data = data[(data["sic"]>7000) | (data["sic"] < 6000)]

# Do not trade stocks with prices less than 1 at the beginning of month
data["price"] = data.groupby("permno")["price"].shift(1)
data.loc[data["price"] < 1, 'tradeable'] = 0

#%%
filtered_data = data.copy()

#%%

# We limit the remaining universe of stocks in each market to a very
# liquid set of securities that could be traded for reasonably low cost at reasonable
# trading volume size. Specifically, we rank stocks based on their beginning-ofmonth
# market capitalization in descending order and include in our universe
# the number of stocks that account cumulatively for 90% of the total market
# capitalization of the entire stock market
filtered_data["me"] = filtered_data.groupby("permno")["me"].shift(1)

# Make sure data is sorted by date and me
filtered_data = filtered_data.sort_values(['eom', 'me'], ascending=[True, False])

#%%
# Code to select the liquid stocks
# List to hold the filtered data for each month
liquid_groups = []

# Process each month separately
for month, group in filtered_data.groupby("eom"):
    # Work on a copy of the group's data
    group = group.copy()
    
    # Calculate the total market cap for the month
    total_mcap = group["me"].sum()
    
    # Compute the cumulative market cap and the fraction of the total
    group["cum_mcap"] = group["me"].cumsum()
    group["cum_fraction"] = group["cum_mcap"] / total_mcap
    
    # Select all stocks until the cumulative fraction is below or equal to 90%
    selected = group[group["cum_fraction"] < 0.9]
    
    # If the last selected stock doesn't push the cumulative fraction over 90%, 
    # include the next stock (if available) to ensure the 90% threshold is reached.
    if len(selected) < len(group):
        if selected.empty:
            next_row = group.iloc[[len(selected)]]
            selected = pd.concat([selected, next_row])
    
    # Append the filtered group for this month
    liquid_groups.append(selected)

# Combine the results from all months
liquid_universe = pd.concat(liquid_groups)

# Optionally, drop helper columns used for calculations
liquid_universe = liquid_universe.drop(columns=["cum_mcap", "cum_fraction"])

# (Optional) If needed, drop duplicates
liquid_universe = liquid_universe.drop_duplicates()

#%%
# Book values are lagged 6 months to ensure data availability to
# investors at the time, and the most recent market values are used to compute
# the ratios

liquid_universe = liquid_universe[liquid_universe["tradeable"]==1]
liquid_universe["be"] = liquid_universe.groupby("permno")["be"].shift(6)

#%%
# Calculating Book to Market
# Shift price and me back to eom instead of beginning of month
liquid_universe["price"] = liquid_universe.groupby("permno")["price"].shift(-1)
liquid_universe["me"] = liquid_universe.groupby("permno")["me"].shift(-1)
liquid_universe["Book_to_Market"] = liquid_universe["be"] / liquid_universe["me"]

#%%
# Start from 6 onwards for all permno as book equity has been lagged
liquid_universe = liquid_universe.groupby("permno").apply(lambda x: x.iloc[6:]).reset_index(drop=True)

#%%
liquid_universe = liquid_universe.dropna(subset="Book_to_Market")
# Rank be/me according to the paper and subtract the mean rank
liquid_universe["ranked_beme"] = liquid_universe.groupby("eom")["Book_to_Market"].transform(lambda x: x.rank()).astype(int)
liquid_universe["weights"] = liquid_universe.groupby("eom")["ranked_beme"].transform(lambda x: x-x.mean())

#%%
# Drop na in case some data is still missing - just in case
liquid_universe = liquid_universe.dropna(subset = ["ranked_beme", "weights","ret"])

#%%
# Save Point
liquid_universe2 = liquid_universe.copy()

#%%
liquid_universe = liquid_universe2.copy()

#%%
# Compute the sum of positive weights per eom
liquid_universe['sum_positive'] = liquid_universe.groupby('eom')['weights'].transform(lambda x: x[x > 0].sum())

# Multiply original weights by c to get true_weights
liquid_universe['scaled_weights'] = liquid_universe['weights'] * (1/liquid_universe['sum_positive'])

# Drop the temporary columns
liquid_universe.drop(columns=['sum_positive'], inplace=True)

#%%
# SHift weights down 1 to get the returns for the appropriate month
liquid_universe["scaled_weights"] = liquid_universe.groupby("permno")["scaled_weights"].shift(1)

# Then compute each stocks's contribution to daily portfolio returns:
liquid_universe["stock_returns"] = liquid_universe["ret"] * liquid_universe["scaled_weights"]

# Just in case
liquid_universe = liquid_universe.drop_duplicates()

#%%
# Extract the monthly portfolio returns and then the cumulative returns
monthly_portfolio_returns = liquid_universe.groupby("eom")["stock_returns"].sum().reset_index()
monthly_portfolio_returns.columns = ["eom", "portfolio_returns"]
monthly_portfolio_returns["cumulative_returns"] =  (1+monthly_portfolio_returns["portfolio_returns"]).cumprod()

#%%
# Import actual factor returns. Has been cleaned in excel
actual = pd.read_excel('/Users/vedant/Desktop/BK/Value_Factor/Value and Momentum Everywhere Factors Monthly.xlsx')
actual["cumulative_returns"] =  (1+actual["VALLS_VME_US90"]).cumprod()

#%%
# Align indexes
start = max(min(monthly_portfolio_returns["eom"]),min(actual["DATE"]))
end = min(max(monthly_portfolio_returns["eom"]),max(actual["DATE"]))

monthly_portfolio_returns = \
    monthly_portfolio_returns[(monthly_portfolio_returns["eom"] >= start)\
    & (monthly_portfolio_returns["eom"] <= end)]

actual = \
    actual[(actual["DATE"] >= start)\
    & (actual["DATE"] <= end)]

#%%
# Plot my factor returns vs the actual returns
plt.figure(figsize=(10, 6))
# Plot my factor returns with circular markers
plt.plot(monthly_portfolio_returns["eom"], monthly_portfolio_returns["cumulative_returns"], 
         marker='.', color='green', label="Factor Returns")
         
# Plot actual returns with square markers
plt.plot(actual["DATE"], actual["cumulative_returns"], 
         marker='.', color='blue', label="Actual Returns")

plt.xlabel("End of Month")
plt.ylabel("Cumulative Returns")
plt.title("Cumulative Monthly Portfolio Returns")
plt.grid(True)
plt.legend()
plt.show()
#%%
monthly_portfolio_returns["eom"] = pd.to_datetime(monthly_portfolio_returns["eom"])
actual["eom"] = pd.to_datetime(actual["DATE"])

monthly_portfolio_returns.set_index("eom",inplace=True)
actual.set_index("eom",inplace=True)
x = monthly_portfolio_returns["portfolio_returns"].corr(actual["VALLS_VME_US90"])
print("Correlation of my factor with actual: ",x)

#%%

my_mean = monthly_portfolio_returns["portfolio_returns"].mean()
my_std = monthly_portfolio_returns["portfolio_returns"].std(ddof=1)


actual_mean = actual["VALLS_VME_US90"].mean()
actual_std = actual["VALLS_VME_US90"].std(ddof=1)

print('My Mean: ', my_mean)
print('My STD: ', my_std)
print('Actual Mean: ', actual_mean)
print('Actual STD: ', actual_std)

#%%
