import pandas
import math
import numpy as np

def preprocess(filepath):
    years_dict = {}
    data = pandas.read_csv(filepath)
    data = data.sort_values(by="GAME_DATE_EST") #sort by reverse order date
    teamIDs = set()
    i = 0
    for row in data.itertuples():
        if row.SEASON in years_dict:
            if row.HOME_TEAM_ID in years_dict[row.SEASON]:
                home_data = [(row.PTS_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][0]*i)/(i+1)),(row.FG_PCT_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][1]*i)/(i+1)),(row.FT_PCT_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][2]*i)/(i+1)),(row.FG3_PCT_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][3]*i)/(i+1)),(row.AST_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][4]*i)/(i+1)), (row.REB_home + (years_dict[row.SEASON][row.HOME_TEAM_ID][5]*i)/(i+1))]
            else:
                home_data = [row.PTS_home,row.FG_PCT_home,row.FT_PCT_home,row.FG3_PCT_home,row.AST_home, row.REB_home]
            if row.VISITOR_TEAM_ID in years_dict[row.SEASON]:
                away_data = [(row.PTS_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][0]*i)/(i+1)),(row.FG_PCT_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][1]*i)/(i+1)),(row.FT_PCT_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][2]*i)/(i+1)),(row.FG3_PCT_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][3]*i)/(i+1)),(row.AST_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][4]*i)/(i+1)), (row.REB_away + (years_dict[row.SEASON][row.VISITOR_TEAM_ID][5]*i)/(i+1))]
            else:
                away_data = [row.PTS_away,row.FG_PCT_away,row.FT_PCT_away,row.FG3_PCT_away,row.AST_away, row.REB_away]
            if not math.isnan(home_data[0]):
                years_dict[row.SEASON][row.GAME_ID] = [row.HOME_TEAM_WINS, home_data, away_data]
                i+=1
                years_dict[row.SEASON][row.HOME_TEAM_ID] =  home_data #averge stats of season so far
                years_dict[row.SEASON][row.VISITOR_TEAM_ID] = away_data #averge stats of season so far
        else: #start the season with the first game data
            home_data = [row.PTS_home,row.FG_PCT_home,row.FT_PCT_home,row.FG3_PCT_home,row.AST_home, row.REB_home]
            away_data = [row.PTS_away,row.FG_PCT_away,row.FT_PCT_away,row.FG3_PCT_away,row.AST_away, row.REB_away]
            years_dict[row.SEASON] = {row.GAME_ID: [row.HOME_TEAM_WINS, home_data, away_data]} #save game result in dict with team stats
            years_dict[row.SEASON][row.HOME_TEAM_ID] = home_data #save the team data to average later
            years_dict[row.SEASON][row.VISITOR_TEAM_ID] = away_data #save the team data to average later
            i=1
    returnArr = []
    for season in years_dict:
        season_data = []
        for game in years_dict[season]:
            if game not in teamIDs:
                season_data.append(years_dict[season][game])
        returnArr.append(season_data)
    return season_data
    
#season_data = preprocess("archive/games.csv")
# for line in season_data:
#     print(line)

def preprocess_odds(filepath):
    odds_data = pandas.read_excel(filepath)
    odds_data = odds_data.replace('pk', 0)
    #odds_data = convert_to_team_id(odds_data)
    labels_dict = {}
    for i, g in odds_data.groupby(odds_data.index // 2):
        away_team_id = g.iloc[0, 3]
        home_team_id = g.iloc[1, 3]
        away_score = int(g.iloc[0, 8])
        home_score = int(g.iloc[1, 8])
        total_score = away_score + home_score
        over_under = max(float(g.iloc[0, 9]), float(g.iloc[1, 9]))
        if (over_under < 25 and over_under > 16):
            over_under *= 10
        assert over_under > 100
        label = None
        if (total_score < over_under):
            label = 0
        elif (total_score > over_under):
            label = 1
        elif (total_score == over_under):
            label = 0.5
        labels_dict[(away_team_id, home_team_id, away_score, home_score)] = label
    return labels_dict

def convert_to_team_id(data):
    data = data.replace("Atlanta", 1610612737)
    data = data.replace("Boston", 1610612738)
    data = data.replace("GoldenState", 1610612739)
    data = data.replace("NewOrleans", 1610612740)
    data = data.replace("Chicago", 1610612741)
    data = data.replace("Dallas", 1610612742)
    data = data.replace("Denver", 1610612743)
    data = data.replace("Cleveland", 1610612744)
    data = data.replace("Houston", 1610612745)
    data = data.replace("LAClippers", 1610612746)
    data = data.replace("LALakers", 1610612747)
    data = data.replace("Miami", 1610612748)
    data = data.replace("Milwaukee", 1610612749)
    data = data.replace("Minnesota", 1610612750)
    data = data.replace("Brooklyn", 1610612751)
    data = data.replace("NewYork", 1610612752)
    data = data.replace("Orlando", 1610612753)
    data = data.replace("Indiana", 1610612754)
    data = data.replace("Philadelphia", 1610612755)
    data = data.replace("Phoenix", 1610612756)
    data = data.replace("Portland", 1610612757)
    data = data.replace("Sacramento", 1610612758)
    data = data.replace("San Antonio", 1610612759)
    data = data.replace("OklahomaCity", 1610612760)
    data = data.replace("Toronto", 1610612761)
    data = data.replace("Utah", 1610612762)
    data = data.replace("Memphis", 1610612763)
    data = data.replace("Washington", 1610612764)
    data = data.replace("Detroit", 1610612765)
    data = data.replace("Charlotte", 1610612766)
    return data

preprocess_odds("archive/nba_odds_2020_2021.xlsx")