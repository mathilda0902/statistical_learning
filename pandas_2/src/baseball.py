import pandas as pd

dfBatting = pd.read_csv('data/baseball-csvs/Batting.csv')

dfBatting['BA'] = dfBatting['H'] / dfBatting['AB']
dfBatting['OBP'] = (dfBatting['H'] + dfBatting['BB'] + dfBatting['HBP']) / (dfBatting['AB'] + dfBatting['BB'] + dfBatting['HBP'] + dfBatting['SF'])
dfBatting['1B'] = dfBatting['H'] - dfBatting['2B'] - dfBatting['3B'] - dfBatting['HR']
dfBatting['SLG'] = (dfBatting['1B'] + 2 * dfBatting['2B'] + 3 * dfBatting['3B'] + 4 * dfBatting['HR']) / dfBatting['AB']

dfSals = pd.read_csv('data/baseball-csvs/Salaries.csv')

dfBatting_new = dfBatting.copy()
dfBatting_1985 = dfBatting_new[dfBatting_new['yearID'] >= 1985]

def merge_batting_and_sals():
    ### REMOVE ALL THE DATA THAT IS BELOW 1985
    dfBatting = dfBatting[dfBatting['yearID'] >= 1985]

    ### MERGE THE TWO TOGETHER ON A DOUBLE CONDITION
    mergeddf = dfBatting_1985.merge(dfSals, on = ['playerID', 'teamID', 'yearID'])

    ### THIS IS A WAY TO DROP A COLUMN IN-PLACE
    mergeddf = mergeddf.drop(['lgID_y', 'G_old'], axis = 1)

    return mergeddf

def main():
    mergeddf = merge_batting_and_sals()

    condition1 = mergeddf['teamID'] == 'OAK'
    condition2 = mergeddf['yearID'] == 2001
    oak2011 = mergeddf[condition1 & condition2]

    ### FIND THE STATS FOR THE PLAYERS WE ARE MISSING
    # THIS IS A LIST OF THE PLAYERS WE ARE LOSING

    lostboys = ['isrinja01', 'giambja01', 'damonjo01', 'saenzol01']
    mask = oak2011['playerID'].isin(lostboys)
    lostboysdf = oak2011[mask]
    lostboydf[['playerID', 'teamID', 'AB', 'HR', 'SLG', 'OBP', 'salary']]

    condition3 = mergeddf.yearID == 2001
    all2001 = mergeddf[condition3]

    condition4 = all2001.AB >= 40
    all2001 = all2001[condition4]

    ### SELECT ONLY THE COLUMNS WE CARE ABOUT, AND SET IT EQUAL TO ITSELF (THUS OVERRIDING IT)
    all2001 = all2001[['playerID', 'teamID_x','AB','HR','SLG', 'OBP', 'salary']]
    # all2001 = all2001.sort('OBP', ascending=False).sort('salary', ascending=True)

    ### SORT BY OPB, IN DESCENDING ORDER
    all2001 = all2001.sort('OBP', ascending=False)

    ### CREATE ANOTHER CONDITION THAT ONLY RETURNS PLAYERS LESS THAN 8MILL
    c4 = all2001.salary < 8000000

    ###  '~' does a select inverse.  so instead of returning .isin()
    ###  it will return .isNOTin().
    c5 = ~all2001['playerID'].isin(lostboys)

    answerdf = all2001[c4 & c5]

    ###
    obp_target = lostboysdf.OBP.mean()
    ab_target = lostboysdf.AB.sum()
    return answerdf, (ab_target, obp_target)




if __name__ == '__main__':
    answerdf, targets = main()
    print answerdf.head(), targets
