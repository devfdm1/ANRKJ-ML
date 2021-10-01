import streamlit as st

# Data processing packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from kmodes.kmodes import KModes

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@st.cache
def loadDataset():
    dataset = pd.read_csv("clustering/bankmarketing.csv")
    return dataset


@st.cache
def clustering(bank_marketing, bank_marketing_copy):

    km_cao = KModes(n_clusters=2, init="Cao", n_init=1, verbose=1)
    fitClusters_cao = km_cao.fit_predict(bank_marketing)

    bank_cust = bank_marketing_copy.reset_index()

    clustersDf = pd.DataFrame(fitClusters_cao)
    clustersDf.columns = ['cluster_predicted']
    combinedDf = pd.concat(
        [bank_cust, clustersDf], axis=1).reset_index()
    combinedDf = combinedDf.drop(['index', 'level_0'], axis=1)

    return combinedDf


def main():
    # st.set_page_config(layout="wide")

    bank = loadDataset()

    st.title("Leo Bank Marketing Analysis")

    st.write('\n')
    st.subheader(f" Marketing Analysis Dataset :")

    st.write(bank)
    st.write()
    st.subheader("Column Fields:")

    st.write('''
            1 - age (numeric)

            2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')

            3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)

            4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')

            5 - default: has credit in default? (categorical: 'no','yes','unknown')

            6 - housing: has housing loan? (categorical: 'no','yes','unknown')

            7 - loan: has personal loan? (categorical: 'no','yes','unknown')

            8 - contact: contact communication type (categorical: 'cellular','telephone')

            9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

            10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

            11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

            12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

            13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

            14 - previous: number of contacts performed before this campaign and for this client (numeric)

            15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
            social and economic context attributes

            16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)

            17 - cons.price.idx: consumer price index - monthly indicator (numeric)

            18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)

            19 - euribor3m: euribor 3 month rate - daily indicator (numeric)

            20 - nr.employed: number of employees - quarterly indicator (numeric)

            21 - y - has the client subscribed a term deposit? (binary: 'yes','no'))'''
             )

    bank_marketing = bank[['age', 'job', 'marital', 'education', 'default',
                           'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']]

    # Data Cleaning
    # checking null values as a percentage of dataset
    # bank_marketing.isnull().sum()*100/bank_marketing.shape[0]

    bank_marketing['age_range'] = pd.cut(bank_marketing['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                         labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    bank_marketing = bank_marketing.drop('age', axis=1)

    bank_marketing_copy = bank_marketing.copy()

    # Convert Categorical Data into Numerical

    le = preprocessing.LabelEncoder()
    bank_marketing = bank_marketing.apply(le.fit_transform)

    #############
    #cost = []
    # for num_clusters in list(range(1, 5)):
    #    kmode = KModes(n_clusters=num_clusters,
    #                   init="Cao", n_init=1, verbose=1)
    #   kmode.fit_predict(bank_marketing)
    #    cost.append(kmode.cost_)
    ###########

    # plot elbow method graph

    #fig = plt.figure()

    st.subheader("Elbow Method :")

    x = np.array([i for i in range(1, 5, 1)])
    cost = [216952.0, 192203.0, 185138.0, 179774.0]

    chart_data = pd.DataFrame(
        cost, x)

    st.line_chart(chart_data)

    combinedDf = clustering(bank_marketing, bank_marketing_copy)

    st.subheader("After categorizing instances into clusters :")
    st.write(combinedDf)

    # Cluster Visualization
    cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
    cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]

    st.markdown('##')
    st.markdown('##')
    st.markdown('##')

    st.subheader("Cluster 0")

    # Cluster 0 data distribution visualization

    fig, ax = plt.subplots(nrows=3, ncols=3)
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[2, 2])

    colors = sns.color_palette('pastel')[0:6]

    ax[0, 0].pie((cluster_0.job.value_counts() * 100 / cluster_0.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 0].set_title('job', fontweight='bold', fontsize=20)
    ax[0, 0].legend(labels=cluster_0.job.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[0, 1].pie((cluster_0.marital.value_counts() * 100 / cluster_0.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 1].set_title('marital', fontweight='bold', fontsize=20)
    ax[0, 1].legend(labels=cluster_0.marital.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[0, 2].pie((cluster_0.education.value_counts() * 100 / cluster_0.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 2].set_title('education', fontweight='bold', fontsize=20)
    ax[0, 2].legend(labels=cluster_0.education.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 0].pie((cluster_0.default.value_counts() * 100 / cluster_0.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 0].set_title('default', fontweight='bold', fontsize=20)
    ax[1, 0].legend(labels=cluster_0.default.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 1].pie((cluster_0.housing.value_counts() * 100 / cluster_0.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 1].set_title('housing', fontweight='bold', fontsize=20)
    ax[1, 1].legend(labels=cluster_0.housing.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 2].pie((cluster_0.loan.value_counts() * 100 / cluster_0.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 2].set_title('loan', fontweight='bold', fontsize=20)
    ax[1, 2].legend(labels=cluster_0.loan.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[2, 0].pie((cluster_0.age_range.value_counts() * 100 / cluster_0.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[2, 0].set_title('age_range', fontweight='bold', fontsize=20)
    ax[2, 0].legend(labels=cluster_0.age_range.value_counts(
    ).index.tolist(), loc="best", fontsize=15)

    fig.suptitle('Percentages on cluster 0', fontsize=20,
                 y=1.07, fontweight='bold', x=0.37)
    fig.set_figheight(22)
    fig.set_figwidth(20)
    fig.tight_layout()

    # plt.show()
    st.pyplot(fig)

    st.markdown('##')
    st.markdown('##')
    st.markdown('##')

    st.subheader("Cluster 1")

    # Cluster 1 data distribution visualization

    fig, ax = plt.subplots(nrows=3, ncols=3)
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[2, 2])

    colors = sns.color_palette('pastel')[0:6]

    ax[0, 0].pie((cluster_1.job.value_counts() * 100 / cluster_1.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 0].set_title('job', fontweight='bold', fontsize=20)
    ax[0, 0].legend(labels=cluster_1.job.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[0, 1].pie((cluster_1.marital.value_counts() * 100 / cluster_1.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 1].set_title('marital', fontweight='bold', fontsize=20)
    ax[0, 1].legend(labels=cluster_1.marital.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[0, 2].pie((cluster_1.education.value_counts() * 100 / cluster_1.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[0, 2].set_title('education', fontweight='bold', fontsize=20)
    ax[0, 2].legend(labels=cluster_1.education.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 0].pie((cluster_1.default.value_counts() * 100 / cluster_1.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 0].set_title('default', fontweight='bold', fontsize=20)
    ax[1, 0].legend(labels=cluster_1.default.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 1].pie((cluster_1.housing.value_counts() * 100 / cluster_1.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 1].set_title('housing', fontweight='bold', fontsize=20)
    ax[1, 1].legend(labels=cluster_1.housing.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[1, 2].pie((cluster_1.loan.value_counts() * 100 / cluster_1.shape[0]), startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[1, 2].set_title('loan', fontweight='bold', fontsize=20)
    ax[1, 2].legend(labels=cluster_1.loan.value_counts(
    ).index.tolist(), loc="best", fontsize=15)
    ax[2, 0].pie((cluster_1.age_range.value_counts() * 100 / cluster_1.shape[0])[0:6], startangle=90, colors=colors,
                 wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', shadow=True, textprops={'fontsize': 14})
    ax[2, 0].set_title('age_range', fontweight='bold', fontsize=20)
    ax[2, 0].legend(labels=cluster_1.age_range.value_counts(
    ).index.tolist(), loc="best", fontsize=15)

    fig.suptitle('Percentages on cluster 1', fontsize=20,
                 y=1.07, fontweight='bold', x=0.37)
    fig.set_figheight(22)
    fig.set_figwidth(20)
    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("Cluster 0 vs Cluster 1")

    ##################################################
    st.write(f"#### Job distribution between clusters : ")

    # Job with cluster

    #fig, ax = plt.subplots(figsize=(15, 5))
    # sns.countplot(x=combinedDf['job'], order=combinedDf['job'].value_counts(
    # ).index, hue=combinedDf['cluster_predicted'])
    # st.pyplot(fig)

    # extract data to a dataframe
    df = pd.DataFrame(
        [
            ["admin.", len(combinedDf[(combinedDf['job'] == "admin.") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "admin.") & (combinedDf['cluster_predicted'] == 1)])],
            ["blue-collar", len(combinedDf[(combinedDf['job'] == "blue-collar") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "blue-collar") & (combinedDf['cluster_predicted'] == 1)])],
            ["technician", len(combinedDf[(combinedDf['job'] == "technician") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "technician") & (combinedDf['cluster_predicted'] == 1)])],
            ["services", len(combinedDf[(combinedDf['job'] == "services") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "services") & (combinedDf['cluster_predicted'] == 1)])],
            ["management", len(combinedDf[(combinedDf['job'] == "management") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "management") & (combinedDf['cluster_predicted'] == 1)])],
            ["retired", len(combinedDf[(combinedDf['job'] == "retired") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "retired") & (combinedDf['cluster_predicted'] == 1)])],
            ["entrepreneur", len(combinedDf[(combinedDf['job'] == "entrepreneur") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "entrepreneur") & (combinedDf['cluster_predicted'] == 1)])],
            ["self-employed", len(combinedDf[(combinedDf['job'] == "self-employed") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "self-employed") & (combinedDf['cluster_predicted'] == 1)])],
            ["housemaid", len(combinedDf[(combinedDf['job'] == "housemaid") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "housemaid") & (combinedDf['cluster_predicted'] == 1)])],
            ["unemployed", len(combinedDf[(combinedDf['job'] == "unemployed") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "unemployed") & (combinedDf['cluster_predicted'] == 1)])],
            ["student", len(combinedDf[(combinedDf['job'] == "student") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "student") & (combinedDf['cluster_predicted'] == 1)])],
            ["unknown", len(combinedDf[(combinedDf['job'] == "unknown") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['job'] == "unknown") & (combinedDf['cluster_predicted'] == 1)])],
        ],
        columns=["Job", "Cluster 0", "Cluster 1"]
    )

    fig = px.bar(df, x="Job", y=[
        "Cluster 0", "Cluster 1"], barmode='group', height=600, width=1500)
    # display as a barchart
    st.plotly_chart(fig)

    ##################################################
    st.write(f"#### Marital status between clusters : ")

    # Marital Status with cluster

    #fig, ax = plt.subplots(figsize=(5, 5))
    # sns.countplot(x=combinedDf['marital'], order=combinedDf['marital'].value_counts(
    # ).index, hue=combinedDf['cluster_predicted'])
    # st.pyplot(fig)

    # extract data to a dataframe
    df = pd.DataFrame(
        [
            ["married", len(combinedDf[(combinedDf['marital'] == "married") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['marital'] == "married") & (combinedDf['cluster_predicted'] == 1)])],
            ["single", len(combinedDf[(combinedDf['marital'] == "single") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['marital'] == "single") & (combinedDf['cluster_predicted'] == 1)])],
            ["divorced", len(combinedDf[(combinedDf['marital'] == "divorced") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['marital'] == "divorced") & (combinedDf['cluster_predicted'] == 1)])],
            ["unknown", len(combinedDf[(combinedDf['marital'] == "unknown") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['marital'] == "unknown") & (combinedDf['cluster_predicted'] == 1)])]
        ],
        columns=["Marital", "Cluster 0", "Cluster 1"]
    )

    fig = px.bar(df, x="Marital", y=[
        "Cluster 0", "Cluster 1"], barmode='group', height=600, width=1500)
    # display as a barchart
    st.plotly_chart(fig)

    ##################################################
    st.write(f"#### Education between clusters : ")
    # Education with cluster
    #fig, ax = plt.subplots(figsize=(15, 5))
    # sns.countplot(x=combinedDf['education'], order=combinedDf['education'].value_counts(
    # ).index, hue=combinedDf['cluster_predicted'])
    # st.pyplot(fig)

    # extract data to a dataframe
    df = pd.DataFrame(
        [
            ["university.degree", len(combinedDf[(combinedDf['education'] == "university.degree") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "university.degree") & (combinedDf['cluster_predicted'] == 1)])],
            ["high.school", len(combinedDf[(combinedDf['education'] == "high.school") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "high.school") & (combinedDf['cluster_predicted'] == 1)])],
            ["basic.9y", len(combinedDf[(combinedDf['education'] == "basic.9y") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "basic.9y") & (combinedDf['cluster_predicted'] == 1)])],
            ["professional.course", len(combinedDf[(combinedDf['education'] == "professional.course") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "professional.course") & (combinedDf['cluster_predicted'] == 1)])],
            ["basic.4y", len(combinedDf[(combinedDf['education'] == "basic.4y") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "basic.4y") & (combinedDf['cluster_predicted'] == 1)])],
            ["basic.6y", len(combinedDf[(combinedDf['education'] == "basic.6y") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "basic.6y") & (combinedDf['cluster_predicted'] == 1)])],
            ["unknown", len(combinedDf[(combinedDf['education'] == "unknown") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "unknown") & (combinedDf['cluster_predicted'] == 1)])],
            ["illiterate", len(combinedDf[(combinedDf['education'] == "illiterate") & (combinedDf['cluster_predicted'] == 0)]),
             len(combinedDf[(combinedDf['education'] == "illiterate") & (combinedDf['cluster_predicted'] == 1)])]
        ],
        columns=["education", "Cluster 0", "Cluster 1"]
    )

    fig = px.bar(df, x="education", y=[
        "Cluster 0", "Cluster 1"], barmode='group', height=600, width=1500)
    # display as a barchart
    st.plotly_chart(fig)

    ##################################################
    st.write(f"#### Credit , Housing , Loan  : ")

    # Credit , housing load  , personal loan acquisition with cluster
    f, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x=combinedDf['default'], order=combinedDf['default'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[0])
    sns.countplot(x=combinedDf['housing'], order=combinedDf['housing'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[1])
    sns.countplot(x=combinedDf['loan'], order=combinedDf['loan'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[2])

    plt.tight_layout()
    st.pyplot(f)

    ##################################################
    st.write(f"#### Month and Day of the week : ")

    # Month and Day of the week with cluster
    f, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(x=combinedDf['month'], order=combinedDf['month'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[0])
    sns.countplot(x=combinedDf['day_of_week'], order=combinedDf['day_of_week'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[1])

    plt.tight_layout()
    st.pyplot(f)

    ##################################################
    st.write(f"#### Campaign Outcome and Age range : ")

    # Campaign outcome and age with cluster
    f, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(x=combinedDf['poutcome'], order=combinedDf['poutcome'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[0])
    sns.countplot(x=combinedDf['age_range'], order=combinedDf['age_range'].value_counts(
    ).index, hue=combinedDf['cluster_predicted'], ax=axs[1])

    plt.tight_layout()
    st.pyplot(f)


if __name__ == '__main__':
    main()
