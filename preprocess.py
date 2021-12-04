import pandas
import math

def preprocess(filepath):
    years_dict = {}
    data = pandas.read_csv(filepath)
    data = data.sort_values(by="GAME_DATE_EST") #sort by reverse order date
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
                print(years_dict[row.SEASON][row.GAME_ID])
                i+=1
        else: #start the season with the first game data
            home_data = [row.PTS_home,row.FG_PCT_home,row.FT_PCT_home,row.FG3_PCT_home,row.AST_home, row.REB_home]
            away_data = [row.PTS_away,row.FG_PCT_away,row.FT_PCT_away,row.FG3_PCT_away,row.AST_away, row.REB_away]
            years_dict[row.SEASON] = {row.GAME_ID: [row.HOME_TEAM_WINS, home_data, away_data]} #save game result in dict with team stats
            years_dict[row.SEASON] = {row.HOME_TEAM_ID: home_data} #save the team data to average later
            years_dict[row.SEASON] = {row.VISITOR_TEAM_ID: away_data} #save the team data to average later
            i=1
    return years_dict

preprocess("/Users/williamallstetter/Desktop/school/CS1470/final project/archive/games.csv")
