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
    
def load_data(file_name):
    with open(file_name, 'rb') as my_file:
        return pickle.load(my_file)


def plot_line(my_data, line_label, color):
    my_df = pd.DataFrame.from_dict(my_data).T
    
    add_mean_sup_inf(my_df, confidence_interval=0.95)
    
    my_df['sup'] = my_df['sup'].fillna(0)
    my_df['inf'] = my_df['inf'].fillna(0)
    
    ax = sns.lineplot(data=my_df, x=my_df.index, y='mean', color=color, label=line_label)
    ax.fill_between(my_df.index, my_df['inf'], my_df['sup'], color=color, alpha=.15)
    
    return ax
    


#my_data = load_data('C:\\temp\\aamas2025\\data_QLPerfectRM.pck')
#my_data = load_data('C:\\temp\\aamas2025\\exp_data\\QLPerfectRM.pck')
#ax = plot_line(my_data, 'Oracle', 'red')

#my_data = load_data('C:\\temp\\aamas2025\\data_QLNorm.pck')
#ax = plot_line(my_data, 'Memory Only', 'gray')

#my_data = load_data('C:\\temp\\aamas2025\\data_QLIndependentBelief_T.pck')
#plot_line(my_data, 'TDM', 'blue')

#my_data = load_data('C:\\temp\\aamas2025\\data_QLIndependentBelief_F.pck')
#plot_line(my_data, 'IBU', 'green')

#ax = my_data = load_data('C:\\temp\\aamas2025\\data_QLBeliefThresholding.pck')
#plot_line(my_data, 'Naive', 'orange')


ax = my_data = load_data('C:\\temp\\aamas2025\\data_QLIndependentBelief_T.pck')
plot_line(my_data, 'TDM', 'red')

ax = my_data = load_data('C:\\temp\\aamas2025\\data_QLIndependentBelief2_T.pck')
plot_line(my_data, 'TDM-2', 'blue')



ax.set_title('Average Return while training')
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_facecolor('whitesmoke')
ax.figure.autofmt_xdate(rotation=45)


plt.grid(True, color='white', linestyle='--')
plt.legend(loc='lower right')
plt.show()
