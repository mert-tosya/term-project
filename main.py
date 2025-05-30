import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score

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
mask_protester_initiated = (
    acled_filtered["actor1"].str.contains("|".join(protester_keys), na=False) &
    acled_filtered["actor2"].str.contains("|".join(state_keys),    na=False)
)
mask_state_initiated = (
    acled_filtered["actor1"].str.contains("|".join(state_keys),    na=False) &
    acled_filtered["actor2"].str.contains("|".join(protester_keys), na=False)
)
# keep peaceful protests
mask_peaceful_events = acled_filtered["sub_event_type"].isin([
    "Peaceful protest",
    "Protest with intervention"
])
acled_filtered = acled_filtered[mask_protester_initiated | mask_state_initiated | mask_peaceful_events]

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

# replace differing names under acled
fixes = {
    "bosnia and herzegovina":      "bosnia",
    "democratic republic of congo":"congo kinshasa",
    "myanmar":                     "myanmar (burma)",
    "north korea":                 "korea north",
    "south korea":                 "korea south",
    "east timor":                  "timor leste",
    "slovakia":                    "slovak republic",
    "united arab emirates":        "uae",
    "north macedonia":             "macedonia",
    "eswatini":                    "swaziland",
}

acled_ctryear["country"] = acled_ctryear["country"].replace(fixes)

# merge the two dataframes
df = acled_ctryear.merge(
    polity,
    on=["country","year"],
    how="inner"
)

# find year-to-year difference in polity

df = df.sort_values(['country','year'])
df['d_polity'] = df.groupby('country')['polity2'].diff()
df2 = df.dropna(subset=['d_polity'])

# summary stats
print(df[['peaceful','intervention','violent','repression','polity2']].describe())

# time series for a few example countries
for country in ['south africa','egypt','ethiopia']:
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

# correlation matrix
corr = df[['peaceful','intervention','violent','repression','d_polity']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# hypothesis tests

# high vs low repression t-test
cut_rep = np.percentile(df2['repression'], 75)
high_rep = df2[df2['repression'] > cut_rep]['d_polity']
low_rep  = df2[df2['repression'] <= cut_rep]['d_polity']
tstat, pval = ttest_ind(high_rep, low_rep, equal_var=False)
print(f"Repression: t={tstat:.2f}, p={pval:.3f}")

# high vs low intervention t-test
cut_int = np.percentile(df2['intervention'], 75)
high_int = df2[df2['intervention'] > cut_int]['d_polity']
low_int  = df2[df2['intervention'] <= cut_int]['d_polity']
tstat_i, pval_i = ttest_ind(high_int, low_int, equal_var=False)
print(f"Intervention: t={tstat_i:.2f}, p={pval_i:.3f}")

# high vs low peaceful t-test
cut_pea = np.percentile(df2['peaceful'], 75)
high_pea = df2[df2['peaceful'] > cut_pea]['d_polity']
low_pea  = df2[df2['peaceful'] <= cut_pea]['d_polity']
tstat_p, pval_p = ttest_ind(high_pea, low_pea, equal_var=False)
print(f"Peaceful: t={tstat_p:.2f}, p={pval_p:.3f}")

# high vs low violent t-test
cut_vio = np.percentile(df2['violent'], 75)
high_vio = df2[df2['violent'] > cut_vio]['d_polity']
low_vio  = df2[df2['violent'] <= cut_vio]['d_polity']
tstat_v, pval_v = ttest_ind(high_vio, low_vio, equal_var=False)
print(f"Violent: t={tstat_v:.2f}, p={pval_v:.3f}")

# correlation between peaceful protests and dpolity
mask = df2[['peaceful','d_polity']].notnull().all(axis=1)
r, p = pearsonr(df2.loc[mask,'peaceful'], df2.loc[mask,'d_polity'])
print(f"Peaceful vs d_Polity: r={r:.2f}, p={p:.3f}")

# correlation between violent unrest and dpolity
mask = df2[['violent','d_polity']].notnull().all(axis=1)
r, p = pearsonr(df2.loc[mask,'violent'], df2.loc[mask,'d_polity'])
print(f"Violent vs d_Polity: r={r:.2f}, p={p:.3f}")

# ols with all the variables
model = smf.ols('d_polity ~ peaceful + intervention + violent + repression', data=df2)
res   = model.fit(cov_type='cluster', cov_kwds={'groups':df2['country']})
print(res.summary())

#define binary variable improve for classification
df2['improve'] = (df2['d_polity'] > 0).astype(int)

# pick x and y
X = df2[['peaceful','intervention','violent','repression']]
y_reg = df2['d_polity']
y_clf = df2['improve']

# split into train and test
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=4
)
_,      _,      y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=4
)

# random forest regression
rf = RandomForestRegressor(n_estimators=100, random_state=4)
rf.fit(X_train, y_train_reg)
y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test_reg, y_pred, squared=False)
print(f"RF Regression RMSE: {rmse:.3f}")

# random forest classification
rfc = RandomForestClassifier(n_estimators=100, random_state=4)
rfc.fit(X_train, y_train_clf)
y_pred_clf = rfc.predict(X_test)
print(classification_report(y_test_clf, y_pred_clf))
print("AUC:", roc_auc_score(y_test_clf, rfc.predict_proba(X_test)[:,1]))

# feature importances
imp = pd.Series(rf.feature_importances_, index=X.columns)
print(imp.sort_values(ascending=False))
