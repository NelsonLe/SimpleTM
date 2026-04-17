import pandas as pd

df = pd.read_csv('annual.csv')

# reformatting date column to work with code
df = df.rename(columns={'Date':'date'})
df['date'] = pd.to_datetime(df['date'])

# make "multivar"
df2 = df.pivot(index='date', columns='Country', values='Exchange rate')
df2 = df2.sort_values('date')  # make sure date is in order

# fill null w/ 0 for now
df2 = df2.fillna(0)

df2.to_csv('proc_annual.csv')
