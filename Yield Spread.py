import os
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import urllib.request
import patsy

os.chdir('C:/Users/Nelson/PycharmProjects/403 Shira')

df_AAA_DATA = pd.read_csv('Data/AAA.csv', index_col=0)
df_AAA_DATA.index = pd.to_datetime(df_AAA_DATA.index)
df_IIP_DATA = pd.read_csv('Data/IIP Quarters.csv')
df_NFIA_DATA = pd.read_csv('Data/NFIA Quarterly.csv')
df_WORLD_DATA = pd.read_csv('Data/countries.csv')
df_US_DATA = pd.read_csv('Data/REAL GDP.csv')

# -----------------------------------------------------------------
# downloading fred us real gdp levels data, unit: 2017 USD
link = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDPC1&scale=left&cosd=1947-01-01&coed=2023-07-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-11-16&revision_date=2023-11-16&nd=1947-01-01'
urllib.request.urlretrieve(link, 'us.csv')
us_lev = pd.read_csv('us.csv', index_col=0)
# formatting DATE strings to pandas datetimes
us_lev.index = pd.to_datetime(us_lev.index)

# same for japan
link = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=JPNRGDPEXP&scale=left&cosd=1994-01-01&coed=2023-07-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-11-16&revision_date=2023-11-16&nd=1994-01-01'
urllib.request.urlretrieve(link, 'jp.csv')
jp_lev = pd.read_csv('jp.csv', index_col=0)
jp_lev.index = pd.to_datetime(jp_lev.index)

# same for china
link = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=RGDPNACNA666NRUG&scale=left&cosd=1952-01-01&coed=2019-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Annual&fam=avg&fgst=lin&fgsnd=2019-01-01&line_index=1&transformation=lin&vintage_date=2023-11-16&revision_date=2023-11-16&nd=1952-01-01'
urllib.request.urlretrieve(link, 'cn.csv')
cn_lev = pd.read_csv('cn.csv', index_col=0)
cn_lev.index = pd.to_datetime(cn_lev.index)
# china is annual. needs interpolation (its current quarterly gdp at fred is also junk)
# also china craps out at Fred at 2019. The Fred's source is Penn's tables. Perhaps you can see if they have more recent observations? If not, you can't use china past 2019.

# same for germany
link = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CLVMNACSCAB1GQDE&scale=left&cosd=1991-01-01&coed=2023-07-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-11-16&revision_date=2023-11-16&nd=1991-01-01'
urllib.request.urlretrieve(link, 'de.csv')
de_lev = pd.read_csv('de.csv', index_col=0)
de_lev.index = pd.to_datetime(de_lev.index)

# merging the data into one dataframe
gdp_lev = pd.concat([us_lev, jp_lev, cn_lev, de_lev], axis=1)
gdp_lev.columns = ['us', 'jp', 'cn', 'de']
# china is annual. needs interpolation (its current quarterly gdp at fred is also junk)
gdp_lev['cn'] = gdp_lev['cn'].interpolate('cubic')

# growth rates
g = (gdp_lev.iloc[1::].values - gdp_lev.iloc[:-1:]) / gdp_lev.iloc[:-1:]
g = g.loc[~np.isnan(g['de'])]

# the growth rate has three components: the worldwide shock, and the country shock
# g_it = g_w + epsilon_it
# where g_w is the time dummy * the marginal effect of time dummy on g_it
# and epsilon_it is the country-specific shock
# format data into a regression-ready longitudinal format
data = pd.melt(g.reset_index(), id_vars=['DATE'], var_name='country', \
               value_vars=['us', 'jp', 'cn', 'de'], value_name='g')
# write the regression in a patsy string. growth  is the explained variable, 
# DATE is the explanatory variable, C is a dummy for country
f = 'g ~ DATE + C(country)'
y, X = patsy.dmatrices(f, data, return_type='dataframe')
# estimate with statsmodels
estm = sm.OLS(y, X, missing='drop').fit()
# print(estm.summary())

# common world shock is the sum of the g intercept and the year effect (but not the country residual!)
# also adding the zero for the omitted first year
# and ignoring the last three estimates of the country effects
world_shock = pd.concat([pd.Series(0), pd.Series(estm.params.values[1:-3])])
world_shock = estm.params[0] + world_shock
world_shock.index = g.index
world_shock.name = 'world'

# country shocks are the sum of the time-invariant country effects plus the residuals
country_shocks = g.subtract(world_shock, axis=0)
g = pd.concat([world_shock, country_shocks], axis=1)
# ------------------------------------------------------------------------------------------------------------


# AAA dataframe
df_AAA_DATA.index = pd.to_datetime(df_AAA_DATA.index)
# df_AAA_DATA['DATE'] = df_AAA_DATA['DATE'].dt.strftime('%Y-%m')

# NFIA/Receipts dataframe
df_NFIA = pd.DataFrame()
df_NFIA[['Exports', 'Receipts', 'Imports', 'Payments']] = df_NFIA_DATA.iloc[range(1, 5), 2:].T.values
df_NFIA = df_NFIA.set_index(pd.date_range(start='1960Q1', periods=254, freq='Q'))
# df_NFIA.index = df_NFIA.index.strftime('%Y-%m-%d')
# df_NFIA = df_NFIA.reset_index(drop=False)
df_NFIA.index.name = 'DATE'

# assets and liabilities dataframe
df_IIP = pd.DataFrame()
df_IIP[['Assets', 'Liabilities']] = df_IIP_DATA.iloc[range(1, 3), 2:].T.values
df_IIP = df_IIP.set_index(pd.date_range(start='2006Q1', periods=70, freq='Q'))
# df_IIP.index = df_IIP.index.strftime('%Y-%m-%d')
# df_IIP = df_IIP.reset_index(drop=False)
df_IIP.index.name = 'DATE'

# merge dataframes
df_DATA = pd.merge(df_IIP, df_NFIA, on=['DATE'], how='inner')
# for column in df_DATA.columns[1:]:
df_DATA = df_DATA.astype(float)
x = df_DATA.index.to_series().reset_index(drop=True)

for t in range(len(x)):
    x[t] = pd.Timestamp(year=df_DATA.index[t].year, month=df_DATA.index[t].month - 2, day=1)
df_DATA.index = x

# Volatility
window = 5
spread_years = 7
for country in g.columns:
    g[f'GROWTH {country} volatility'] = g[country].rolling(window=window).std()

# r_row
df_DATA = df_DATA.merge(df_AAA_DATA, on='DATE', how='inner')
df_DATA['r_row'] = (df_DATA['Receipts'] / df_DATA['Assets']) * 400

# Spread
df_DATA['SPREAD'] = df_DATA['r_row'] - df_DATA['AAA']
spread_list = []
for y in range(spread_years):
    df_DATA[f'SPREAD{y}'] = df_DATA['SPREAD'].shift(y)
    spread_list.append(f'SPREAD{y}')

# DataFrame for regression Spread2 with lag 1 produces MEs consistent with theory
df = g.copy()
df = df.merge(df_DATA[spread_list], on='DATE', how='inner')
df[df.columns[5:10:]] = df[df.columns[5:10:]] * 100

for y in range(spread_years):
    model = f'SPREAD{y} ~ Q("GROWTH world volatility" ) + Q("GROWTH us volatility" ) \
        + Q("GROWTH jp volatility")\
        + Q("GROWTH cn volatility") + Q("GROWTH de volatility") '
    Y, X = patsy.dmatrices(model, df, return_type='dataframe')
    model = sm.OLS(Y, X).fit()
    print(model.summary())

df.describe()

# --------------------------------------------------------
# ols regression, spec 2: year effects dominate, 
# time-independent volatility component effect is small and insignificamnt
vol = pd.melt(g.iloc[:, 4:].reset_index(), id_vars='DATE', var_name='country', value_vars= \
    g.iloc[:, 4:].columns, value_name='vol')
vol.set_index('DATE', inplace=True)
vol['SPREAD'] = df['SPREAD']
vol.reset_index(inplace=True)
model = 'SPREAD ~ vol + C(country) + C(DATE) '
Y, X = patsy.dmatrices(model, vol, return_type='dataframe')
model = sm.OLS(Y, X).fit()
#print(model.summary())

df = df.merge(df_DATA[['AAA', 'r_row', 'DATE']], on=['DATE'], how='inner')
out = []
for y in range(len(df['DATE'])):
    out.append(2006.25 + y * .25)
df['DATE'] = out
df = df.dropna()

plt.plot(df['DATE'], df['GROWTH US'], label='Growth US')
plt.plot(df['DATE'], df['GROWTH CANADA'], label='Growth CANADA')
plt.plot(df['DATE'], df['GROWTH EU'], label='Growth EU')
plt.plot(df['DATE'], df['GROWTH CHINA'], label='Growth CHINA')

# plt.plot(df['DATE'], df['AAA'], label='AAA')
# plt.plot(df['DATE'], df['r_row'], label='r_row')
plt.plot(df['DATE'], df['SPREAD'], label='SPREAD')
plt.xlabel('Years')
plt.ylabel('Percent Value')
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.show()

df.to_excel('mydata.xlsx', index=False)
