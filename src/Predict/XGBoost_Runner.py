import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc



THRESHOLD_ML = 0.594  
THRESHOLD_OU = 0.5  

# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_0.712AUC_ML-4.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_56.8%_UO-9.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]

        # Probabilidades (multi:softprob -> [ [p_class0, p_class1] ])
        prob_home_win = ml_predictions_array[count][0][1]  # classe 1 = vitória casa
        prob_away_win = ml_predictions_array[count][0][0]

        # Aplica THRESHOLD_ML em vez de argmax
        winner = 1 if prob_home_win >= THRESHOLD_ML else 0

        # OU: assumindo indice 1 = OVER e 0 = UNDER (mantém lógica original)
        prob_over = ou_predictions_array[count][0][1]
        prob_under = ou_predictions_array[count][0][0]

        # Aplica THRESHOLD_OU em vez de argmax
        under_over = 1 if prob_over >= THRESHOLD_OU else 0

        # Confiança para imprimir (em %)
        if winner == 1:
            winner_confidence = round(prob_home_win * 100, 1)
            if under_over == 0:
                un_confidence = round(prob_under * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(prob_over * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(prob_away_win * 100, 1)
            if under_over == 0:
                un_confidence = round(prob_under * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(prob_over * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        count += 1

    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                        'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    deinit()
