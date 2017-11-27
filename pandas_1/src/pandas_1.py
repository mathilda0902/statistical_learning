import pandas as pd

df = pd.read_csv("data/hospital-costs.csv")

# question 1
df['Total Charges'] = df['Discharges'] * df['Mean Charge']

'''In [6]: df.head()
Out[6]:
   Year  Facility Id                                Facility Name  \
0  2011          324  Adirondack Medical Center-Saranac Lake Site
1  2011          324  Adirondack Medical Center-Saranac Lake Site
2  2011          324  Adirondack Medical Center-Saranac Lake Site
3  2011          324  Adirondack Medical Center-Saranac Lake Site
4  2011          324  Adirondack Medical Center-Saranac Lake Site


   Mean Cost  Median Cost  Total Charges
0   196080.0     123347.0      1083867.0
1    59641.0      59641.0       102190.0
2     6888.0       6445.0        85032.0
3     4259.0       4259.0         8833.0
4     1727.0       1727.0         5264.0
'''

# question 2
df['Total Costs']=df['Mean Cost'] * df['Discharges']

'''
Out[11]:
   Year  Facility Id                                Facility Name  \
0  2011          324  Adirondack Medical Center-Saranac Lake Site
1  2011          324  Adirondack Medical Center-Saranac Lake Site
2  2011          324  Adirondack Medical Center-Saranac Lake Site
3  2011          324  Adirondack Medical Center-Saranac Lake Site
4  2011          324  Adirondack Medical Center-Saranac Lake Site


   Mean Cost  Median Cost  Total Charges  Total Costs
0   196080.0     123347.0      1083867.0     588240.0
1    59641.0      59641.0       102190.0      59641.0
2     6888.0       6445.0        85032.0      41328.0
3     4259.0       4259.0         8833.0       4259.0
4     1727.0       1727.0         5264.0       1727.0
'''

# question 3
df['Markup'] = df['Total Charges'] / df['Total Costs']

'''
Out[13]:
   Year  Facility Id                                Facility Name  \
0  2011          324  Adirondack Medical Center-Saranac Lake Site
1  2011          324  Adirondack Medical Center-Saranac Lake Site
2  2011          324  Adirondack Medical Center-Saranac Lake Site
3  2011          324  Adirondack Medical Center-Saranac Lake Site
4  2011          324  Adirondack Medical Center-Saranac Lake Site


   Mean Cost  Median Cost  Total Charges  Total Costs    Markup
0   196080.0     123347.0      1083867.0     588240.0  1.842559
1    59641.0      59641.0       102190.0      59641.0  1.713419
2     6888.0       6445.0        85032.0      41328.0  2.057491
3     4259.0       4259.0         8833.0       4259.0  2.073961
4     1727.0       1727.0         5264.0       1727.0  3.048060
'''

# questoin 4
df.sort_values(by='Markup',ascending=False)
df.sort_values('Markup').tail(1)

'''
        Year  Facility Id                                    Facility Name  \
370760  2009           74  TLC Health Network Tri-County Memorial Hospital

          Markup
370760  0.015803
'''

'''
        Year  Facility Id                          Facility Name  \
111925  2011         1302  SUNY Downstate Medical Center at LICH

          Markup
111925  20.83559
'''

# question 5
df.groupby('APR DRG Description')['Markup'].count().sort_values(ascending = False).head(10)

df.groupby('Facility Name').mean().sort_values(by='Markup', ascending=False)

'''
APR DRG Description
Other Pneumonia                                             2375
Chronic Obstructive Pulmonary Disease                       2334
Heart Failure                                               2313
Cellulitis & Other Bacterial Skin Infections                2295
Kidney & Urinary Tract Infections                           2290
Cardiac Arrhythmia & Conduction Disorders                   2262
Diabetes                                                    2200
Other Anemia & Disorders Of Blood & Blood-Forming Organs    2198
Septicemia & Disseminated Infections                        2196
Electrolyte Disorders Except Hypovolemia Related            2184
Name: Markup, dtype: int64
'''

# Follow the money
# question 1
net=df[['Facility Name','Total Charges','Total Costs']]

'''
                                 Facility Name  Total Charges  Total Costs
0  Adirondack Medical Center-Saranac Lake Site      1083867.0     588240.0
1  Adirondack Medical Center-Saranac Lake Site       102190.0      59641.0
2  Adirondack Medical Center-Saranac Lake Site        85032.0      41328.0
3  Adirondack Medical Center-Saranac Lake Site         8833.0       4259.0
4  Adirondack Medical Center-Saranac Lake Site         5264.0       1727.0
'''
# question 2
net.groupby('Facility Name')[['Total Charges','Total Costs']].sum()

'''
                                               Total Charges   Total Costs
Facility Name
Adirondack Medical Center-Saranac Lake Site     1.415735e+08  7.742766e+07
Albany Medical Center - South Clinical Campus   1.802808e+06  1.432784e+06
Albany Medical Center Hospital                  3.763945e+09  1.336299e+09
Albany Memorial Hospital                        2.219740e+08  9.490717e+07
Alice Hyde Medical Center                       8.723797e+07  4.105882e+07
'''

# question 3
net_income['Net Income'] = net_income['Total Charges'] - net_income['Total Costs']
net_income.sort_values('Net Income').head(1)
'''
                                                  Net Income
Facility Name
TLC Health Network Tri-County Memorial Hospital -194816068.0'''

net_income.sort_values('Net Income').tail(1)
'''
                                 Total Charges   Total Costs    Net Income
Facility Name
North Shore University Hospital   7.984556e+09  1.933824e+09  6.050732e+09
'''

# Viral Meningitis
newdf = df[df["APR DRG Description"] == "Viral Meningitis"]

# question 2
newdf[["Facility Name", "APR DRG Description","APR Severity of Illness Description","Discharges", "Mean Charge", "Median Charge", "Mean Cost"]]

# question 3
