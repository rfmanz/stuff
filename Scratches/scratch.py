from pyutils import *
import matplotlib.pyplot as plt
import seaborn as sns

read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip')
sample_submission, test, train = read_data('/home/r/Downloads/tabular-playground-series-apr-2021.zip', True)
tr = train

num_feat = ["Age", "Fare"]
tr["target_name"] = tr["Survived"].map({0: "Not Survived", 1: "Survived"})

for column in num_feat:
    fig = plt.figure(figsize=(18, 12));

    sns.distplot(tr[column].dropna(), ax=plt.subplot(221));
    plt.xlabel(column, fontsize=14);
    plt.ylabel("Density", fontsize=14);
    plt.suptitle("Plots for " + column, fontsize=18);

    sns.distplot(tr.loc[tr.Survived == 0, column].dropna(),
                 color="red", label="Not survived", ax=plt.subplot(222))
    sns.distplot(tr.loc[tr.Survived == 1, column].dropna(),
                 color="blue", label="Survived", ax=plt.subplot(222))
    plt.legend(loc="best")
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Density", fontsize=14)

    sns.barplot(x="target_name", y=column, data=tr, ax=plt.subplot(223))
    plt.xlabel("Survived", fontsize=14)
    plt.ylabel("Average " + column, fontsize=14)

    sns.boxplot(x="target_name", y=column, data=tr, ax=plt.subplot(224))
    plt.xlabel("Survived", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.show()

# ------------
s = train[~pd.isnull(train['Ticket_type'])]['Ticket_type']
chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])

plt.figure(figsize=(12, 8))
ax = sns.barplot(y="labels", x="data", data=chart)
ncount = len(train[["Ticket_type"]])
#plt.xlabel(variable_name)
ax2 = ax.twinx()

# Switch so count axis is on right, frequency on left
ax.yaxis.tick_left()
ax2.yaxis.tick_right()

# Also switch the labels over
ax2.yaxis.set_label_position('right')
ax.yaxis.set_label_position('left')

ax2.set_ylabel('%', rotation=180, fontsize=10, va='bottom', ha='right')

for p in ax.patches:
    x = p.get_bbox().get_points()[:, 1]
    y = p.get_bbox().get_points()[1, 0]
    ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y), ha='left', va='bottom')  # set the alignment of the text
plt.show()

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0, 100)
ax.set_ylim(0, ncount)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)

plt.show()

ax.set_ylabel('%', rotation=180, fontsize=10, va='bottom', ha='right')

#---
