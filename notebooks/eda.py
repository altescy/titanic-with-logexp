# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# %%
DATADIR = Path("../data")
list(DATADIR.glob("*"))


# %%
train_df = pd.read_csv(DATADIR / "train.csv")
test_df = pd.read_csv(DATADIR / "test.csv")
all_df = pd.concat([train_df, test_df])

print("** train.csv **")
print(train_df.info())

print("** test.csv **")
print(train_df.info())

# %% [markdown]
# - Number of NaNs

# %%
all_df.isnull().sum()

# %% [markdown]
# ### Survived

# %%
sns.countplot(train_df['Survived'])
plt.show()

# %% [markdown]
# ### Pclass

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['Pclass'])

plt.subplot(1, 2, 2)
sns.countplot(x='Pclass', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Sex

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['Sex'])

plt.subplot(1, 2, 2)
sns.countplot(x='Sex', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Age

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.distplot(train_df['Age'].dropna())

plt.subplot(1, 2, 2)
sns.violinplot(x='Survived', y='Age', data=train_df)

plt.show()

# %% [markdown]
# binning

# %%
train_df["AgeBin"] = pd.qcut(train_df["Age"], 10)

plt.figure(figsize=(18, 12))
plt.subplot(2, 1, 1)
sns.countplot(x='AgeBin', data=train_df)
plt.subplot(2, 1, 2)
sns.countplot(x='AgeBin', hue='Survived', data=train_df)
plt.show()

# %% [markdown]
# ### Family size
# %% [markdown]
# #### Sibsp: Number of brothers/sisters or spouses in the ship

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['SibSp'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='SibSp', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# #### Parch: Number of parents or children in the ship

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['Parch'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='Parch', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# #### Family size: `ShibSp` + `Parch` + 1

# %%
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['FamilySize'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='FamilySize', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Ticket

# %%
train_df['Ticket'].unique()[:10]


# %%
train_df["TicketFreq"] = train_df.groupby('Ticket')['Ticket'].transform('count')

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['TicketFreq'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='TicketFreq', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Fare

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.distplot(train_df['Fare'].dropna())

plt.subplot(1, 2, 2)
sns.violinplot(x='Survived', y='Fare', data=train_df)

plt.show()

# %% [markdown]
# binning

# %%
train_df["FareBin"] = pd.qcut(train_df["Fare"], 4)

plt.figure(figsize=(18, 12))
plt.subplot(2, 1, 1)
sns.countplot(x='FareBin', data=train_df)
plt.subplot(2, 1, 2)
sns.countplot(x='FareBin', hue='Survived', data=train_df)
plt.show()

# %% [markdown]
# ### Cabin
#
# - too many NaNs

# %%
cabin_categories = set(
    c[0]
    for cb in train_df['Cabin'].dropna().unique()
    for c in cb.split()
)

for c in cabin_categories:
    train_df['Cabin_' + c] = train_df['Cabin'].apply(
        lambda x: (1 if c in x else 0) if type(x) == str else x
    )


plt.figure(figsize=(18, 30))

for i, c in enumerate(sorted(cabin_categories)):
    plt.subplot(len(cabin_categories), 2, 2 * i + 1)
    sns.countplot(train_df['Cabin_'+c].dropna())

    plt.subplot(len(cabin_categories), 2, 2 * i + 2)
    sns.countplot(x='Cabin_'+c, hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Deck

# %%
train_df["Deck"] = train_df["Cabin"].apply(lambda x: x[0] if isinstance(x, str) else "X")
train_df['Deck'] = train_df['Deck'].replace(['A', 'B', 'C'], 'ABC')
train_df['Deck'] = train_df['Deck'].replace(['D', 'E'], 'DE')
train_df['Deck'] = train_df['Deck'].replace(['F', 'G'], 'FG')

plt.figure(figsize=(18, 12))
plt.subplot(2, 1, 1)
sns.countplot(x='Deck', data=train_df)
plt.subplot(2, 1, 2)
sns.countplot(x='Deck', hue='Survived', data=train_df)
plt.show()

# %% [markdown]
# ### Embarked

# %%
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['Embarked'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='Embarked', hue='Survived', data=train_df)

plt.show()

# %% [markdown]
# ### Title

# %%
import re

train_df['Name_title'] = train_df['Name'].apply(lambda x: re.findall(r', (\w+)\.', x))
train_df['Name_title'] = train_df['Name_title'].apply(lambda x: np.NaN if len(x) < 1 else x[0])

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.countplot(train_df['Name_title'].dropna())

plt.subplot(1, 2, 2)
sns.countplot(x='Name_title', hue='Survived', data=train_df)

plt.show()


# %%



