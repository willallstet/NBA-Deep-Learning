import pandas
import math

def preprocess(filepath):
    years_dict = {}
    data = pandas.read_csv(filepath)
    data['Date'] =pandas.to_datetime(data.Date)
    data = data.sort_values(by="Date") #sort by reverse order date
    teamIDs = set()
    for row in data.itertuples():
        if row.SEASON in years_dict:
            if row.HOME_TEAM_ID in years_dict[row.SEASON]:
                i = years_dict[row.SEASON][row.HOME_TEAM_ID][0]
                home_data = [i+1,(row.PTS_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][1]*i))/(i+1),(row.FG_PCT_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][2]*i))/(i+1),(row.FT_PCT_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][3]*i))/(i+1),(row.FG3_PCT_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][4]*i))/(i+1),(row.AST_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][5]*i))/(i+1), (row.REB_home+(years_dict[row.SEASON][row.HOME_TEAM_ID][6]*i))/(i+1)]
            else:
                home_data = [0,row.PTS_home,row.FG_PCT_home,row.FT_PCT_home,row.FG3_PCT_home,row.AST_home, row.REB_home]
            if row.VISITOR_TEAM_ID in years_dict[row.SEASON]:
                i = years_dict[row.SEASON][row.VISITOR_TEAM_ID][0]
                away_data = [i+1,(row.PTS_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][1]*i))/(i+1),(row.FG_PCT_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][2]*i))/(i+1),(row.FT_PCT_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][3]*i))/(i+1),(row.FG3_PCT_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][4]*i))/(i+1),(row.AST_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][5]*i))/(i+1), (row.REB_away+(years_dict[row.SEASON][row.VISITOR_TEAM_ID][6]*i))/(i+1)]
            else:
                away_data = [0, row.PTS_away,row.FG_PCT_away,row.FT_PCT_away,row.FG3_PCT_away,row.AST_away, row.REB_away]
            if not math.isnan(home_data[0]):
                years_dict[row.SEASON][row.GAME_ID] = [row.HOME_TEAM_WINS, row.HOME_TEAM_ID, row.VISITOR_TEAM_ID, home_data[1:], away_data[1:]]
                years_dict[row.SEASON][row.HOME_TEAM_ID] =  home_data #averge stats of season so far
                years_dict[row.SEASON][row.VISITOR_TEAM_ID] = away_data #averge stats of season so far
        else: #start the season with the first game data
            home_data = [0,row.PTS_home,row.FG_PCT_home,row.FT_PCT_home,row.FG3_PCT_home,row.AST_home, row.REB_home]
            away_data = [0,row.PTS_away,row.FG_PCT_away,row.FT_PCT_away,row.FG3_PCT_away,row.AST_away, row.REB_away]
            years_dict[row.SEASON] = {row.GAME_ID: [row.HOME_TEAM_WINS, row.HOME_TEAM_ID, row.VISITOR_TEAM_ID, home_data[1:], away_data[1:]]} #save game result in dict with team stats
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