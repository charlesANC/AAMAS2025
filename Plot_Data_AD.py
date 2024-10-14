import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import seaborn as sns
from functools import partial
import pickle


def add_mean_sup_inf(my_data, confidence_interval):
    means = []
    sups = []
    infs = []

    for ind in my_data.index:
        values = my_data.loc[ind].values.tolist()

        
        mean_v = np.mean(values)
        inf_v, sup_v = st.t.interval(confidence=confidence_interval, df=len(values)-1,
                                    loc=mean_v, scale=st.sem(values))

        means.append(mean_v)
        sups.append(sup_v)
        infs.append(inf_v)
        
    my_data['mean'] = means
    my_data['sup'] = sups
    my_data['inf'] = infs


file_name = "QLBeliefThresholding_AD"

with open(f'C:\\temp\\aamas2025\\{file_name}.pck', 'rb') as my_file:
    my_data = pickle.load(my_file)
    
   
my_df = pd.DataFrame.from_dict(my_data).T


add_mean_sup_inf(my_df, confidence_interval=0.95)

my_df['sup'] = my_df['sup'].fillna(0)
my_df['inf'] = my_df['inf'].fillna(0)


ax = sns.lineplot(data=my_df, x=my_df.index, y='mean', color='b', label='QLBeliefThresholding_AD')
ax.fill_between(my_df.index, my_df['inf'], my_df['sup'], color='b', alpha=.15)


file_name = "QLBeliefThresholding2_AD"

with open(f'C:\\temp\\aamas2025\\{file_name}.pck', 'rb') as my_file:
    my_data2 = pickle.load(my_file)
    
   
my_df2 = pd.DataFrame.from_dict(my_data2).T


add_mean_sup_inf(my_df2, confidence_interval=0.95)

my_df2['sup'] = my_df2['sup'].fillna(0)
my_df2['inf'] = my_df2['inf'].fillna(0)


ax = sns.lineplot(data=my_df2, x=my_df2.index, y='mean', color='r', label='QLBeliefThresholding2_AD')
ax.fill_between(my_df2.index, my_df2['inf'], my_df2['sup'], color='r', alpha=.15)


ax.set_title('Perfect RM - Do not observe vs Observe Opponent')
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_facecolor('whitesmoke')
#ax.set_ylim(0)
#ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
#ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax.figure.autofmt_xdate(rotation=45)


plt.grid(True, color='white', linestyle='--')
plt.legend(loc='lower right')
plt.show()
