import zipfile, pandas as pd, matplotlib.pyplot as plt, numpy as np

# with zipfile.ZipFile("armenian-online-job-postings.zip", "r") as myzipfile:
#     myzipfile.extractall()

jobs = pd.read_csv("online-job-postings.csv")
print(jobs.head())
print(jobs.tail())
print(jobs['Year'].value_counts())  # count of unique values
print(jobs.sample(5))
print(jobs.describe())
# isnull
# duplicated

jobs.info()

jobs_clean = jobs.copy()
jobs_clean = jobs_clean.rename(columns={
    'ApplicationP': 'ApplicationProcedure',
    'AboutC': 'AboutCompany',
    'RequiredQual': 'RequiredQualifications',
    'JobRequirment': 'JobRequirement'
})

print(jobs_clean.StartDate.value_counts())

asap_list = ['Immediately', 'As soon as possible', 'Upon hiring',
             'Immediate', 'Immediate employment', 'As soon as possible.', 'Immediate job opportunity',
             '"Immediate employment, after passing the interview."',
             'ASAP preferred', 'Employment contract signature date',
             'Immediate employment opportunity', 'Immidiately', 'ASA',
             'Asap', '"The position is open immediately but has a flexible start date depending on the candidates earliest availability."',
             'Immediately upon agreement', '20 November 2014 or ASAP',
             'immediately', 'Immediatelly',
             '"Immediately upon selection or no later than November 15, 2009."',
             'Immediate job opening', 'Immediate hiring', 'Upon selection',
             'As soon as practical', 'Immadiate', 'As soon as posible',
             'Immediately with 2 months probation period',
             '12 November 2012 or ASAP', 'Immediate employment after passing the interview',
             'Immediately/ upon agreement', '01 September 2014 or ASAP',
             'Immediately or as per agreement', 'as soon as possible',
             'As soon as Possible', 'in the nearest future', 'immediate',
             '01 April 2014 or ASAP', 'Immidiatly', 'Urgent',
             'Immediate or earliest possible', 'Immediate hire',
             'Earliest  possible', 'ASAP with 3 months probation period.',
             'Immediate employment opportunity.', 'Immediate employment.',
             'Immidietly', 'Imminent', 'September 2014 or ASAP', 'Imediately']

for a in asap_list:
    jobs_clean.StartDate.replace(a, 'ASAP', inplace=True)

for a in asap_list:
    assert a not in jobs_clean.StartDate.values

# print(jobs_clean.StartDate.value_counts())

# jobs_clean.info()

labels = np.full(len(jobs_clean.StartDate.value_counts()), "", dtype=object)
labels[0] = "ASAP"
jobs_clean.StartDate.value_counts().plot(kind="pie", labels=labels)

plt.show()

