import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import statsmodels.formula.api as smf

# known issues: mismatches between country names between the two datasets
# the event counts are absolute and not per capita

# read acled data and aggregate it
csv_files = glob.glob("data/acled/*.csv")
acled = pd.concat(
    (pd.read_csv(f, parse_dates=["event_date"]) for f in csv_files),
    ignore_index=True
)

# keep only relevant event types
keep = [
    "Peaceful protest", "Violent demonstration", "Protest with intervention",
    "Abduction/forced disappearance", "Arrests", "Excessive force against protesters",
    "Armed clash", "Mob violence", "Looting/property destruction",
    "Attack", "Sexual violence"
]
acled_filtered = acled[acled["sub_event_type"].isin(keep)]

state_keys     = ["Police", "Military", "Security", "Forces", "Army"]
protester_keys = ["Protest", "Civilian", "Crowd"]

# keep civilian-state interactions
mask_protester_attacks = (
    acled_filtered["actor1"].str.contains("|".join(protester_keys), na=False) &
    acled_filtered["actor2"].str.contains("|".join(state_keys),    na=False)
)
mask_state_attacks = (
    acled_filtered["actor1"].str.contains("|".join(state_keys),    na=False) &
    acled_filtered["actor2"].str.contains("|".join(protester_keys), na=False)
)
# keep peaceful protests
mask_peaceful_events = acled_filtered["sub_event_type"].isin([
    "Peaceful protest",
    "Protest with intervention"
])
acled_filtered = acled_filtered[mask_protester_attacks | mask_state_attacks | mask_peaceful_events]

# group events into 3 categories based on actors
is_state = acled_filtered["actor1"].str.contains("|".join(state_keys), na=False)
conditions = [
    acled_filtered["sub_event_type"] == "Peaceful protest",
    # making a separate column for protest with intervention as otherwise we would need to double count
    acled_filtered["sub_event_type"] == "Protest with intervention",
    acled_filtered["sub_event_type"].isin([
        "Violent demonstration", "Armed clash",
        "Mob violence", "Looting/property destruction"
    ]),
    acled_filtered["sub_event_type"].isin([
        "Arrests", "Abduction/forced disappearance",
        "Excessive force against protesters", "Sexual violence"
    ]),
    (acled_filtered["sub_event_type"] == "Attack") & is_state,
    (acled_filtered["sub_event_type"] == "Attack") & ~is_state,
]

choices = ["peaceful", "intervention", "violent", "repression", "repression", "violent"]
acled_filtered["category"] = np.select(conditions, choices, default=np.nan)
acled_filtered = acled_filtered.dropna(subset=["category"])

# get counts
counts = (
    acled_filtered
      .groupby(["country", "year", "category"])
      .size()
      .rename("n_events")
)

# unstack so each category is a column
acled_ctryear = counts.unstack(fill_value=0).reset_index()

# drop columns name
acled_ctryear.columns.name = None

# drop nan column
acled_ctryear = acled_ctryear.loc[:, acled_ctryear.columns.notnull()]

# read polity data
polity = pd.read_excel(
    "data/p5v2018.xls",
    usecols=["country","year","polity2"]
)

acled_ctryear["country"] = acled_ctryear["country"].str.lower().str.strip()
polity["country"] = polity["country"].str.lower().str.strip()
polity["year"]      = polity["year"].astype(int)
acled_ctryear["year"] = acled_ctryear["year"].astype(int)

# merge the two dataframes
df = acled_ctryear.merge(
    polity,
    on=["country","year"],
    how="inner"
)

# summary stats
print(df[['peaceful','intervention','violent','repression','polity2']].describe())

# istributions of unrest counts
for col in ['peaceful','intervention','violent','repression']:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f'Distribution of {col} events per country-year')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.show()

# correlation heatmap
corr = df[['peaceful','intervention','violent','repression','polity2']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# time series for a few example countries
for country in ['turkey','egypt','iraq']:
    sub = df[df.country == country]
    if sub.empty: continue
    plt.figure()
    plt.plot(sub.year, sub.peaceful,      label='peaceful')
    plt.plot(sub.year, sub.intervention,  label='intervention')
    plt.plot(sub.year, sub.violent,       label='violent')
    plt.plot(sub.year, sub.repression,    label='repression')
    plt.title(f'Unrest trends in {country.title()}')
    plt.xlabel('Year')
    plt.legend()
    plt.show()

# hypothesis tests

df = df.sort_values(['country','year'])
df['d_polity'] = df.groupby('country')['polity2'].diff()
df2 = df.dropna(subset=['d_polity'])

# high vs low repression t-test
cut_rep = np.percentile(df2['repression'], 75)
high_rep = df2[df2['repression'] >= cut_rep]['d_polity']
low_rep  = df2[df2['repression']  < cut_rep]['d_polity']
tstat, pval = ttest_ind(high_rep, low_rep, equal_var=False)
print(f"Repression: t={tstat:.2f}, p={pval:.3f}")

# high vs low intervention t-test
cut_int = np.percentile(df2['intervention'], 75)
high_int = df2[df2['intervention'] >= cut_int]['d_polity']
low_int  = df2[df2['intervention']  < cut_int]['d_polity']
tstat_i, pval_i = ttest_ind(high_int, low_int, equal_var=False)
print(f"Intervention: t={tstat_i:.2f}, p={pval_i:.3f}")

# correlation between violent unrest and dpolity
mask = df2[['violent','d_polity']].notnull().all(axis=1)
rho, p = pearsonr(df2.loc[mask,'violent'], df2.loc[mask,'d_polity'])
print(f"Violent vs d_Polity: r={rho:.2f}, p={p:.3f}")

# ols with all the variables
model = smf.ols('d_polity ~ peaceful + intervention + violent + repression', data=df2)
res   = model.fit(cov_type='cluster', cov_kwds={'groups':df2['country']})
print(res.summary())
