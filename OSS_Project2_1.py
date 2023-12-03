import pandas as pd
from pandas import Series, DataFrame

#csv 파일 읽어오기
#따로 주신 조건 같은 게 없어서, 같은 폴더 내에 csv파일이 있다고 가정하였습니다.
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
print()
# 1) Print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR), and on base percentage (출루율, OBP)
# for each year from 2015 to 2018. (15 points)

print("1) Print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR), and on base percentage (출루율, OBP) for each year from 2015 to 2018. (15 points)")
# 2015년부터 2018년까지 각 연도별로 데이터를 처리
for year in range(2015, 2019):  
    print(f'\nTop 10 players in {year}:')
    # 현재 연도에 해당하는 데이터만을 선택하여 year_df에 저장
    year_df = df[df['year'] == year]
    
    # 각 통계 지표(H, avg, HR, OBP)별로 상위 10명의 선수를 출력
    for column in ['H', 'avg', 'HR', 'OBP']:
        # 현재 지표를 기준으로 데이터를 내림차순으로 정렬하여 sorted_df만들기
        sorted_df = year_df.sort_values(by=column, ascending=False)
        # 정렬된 데이터프레임에서 상위 10개의 행을 선택하여 top_10_players에 저장
        top_10_players = sorted_df.head(10)
        
        #따로 무엇을 출력하라는 지시 사항이 없어서 batter_name과 기준 열('H', 'avg', 'HR', 'OBP'중 하나)을 출력하였습니다. 
        print(top_10_players[['batter_name', column]])
        print()
        
    
    
# 2) Print the player with the highest war (승리 기여도) by position (cp) in 2018. (15 points)
# ▪ Position info. - 포수, 1루수, 2루수, 3루수, 유격수, 좌익수, 중견수, 우익수
print("2) Print the player with the highest war (승리 기여도) by position (cp) in 2018. (15 points)")
print()

positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

# 2018년에 해당하는 데이터만을 선택하여 df_2018에 저장
df_2018 = df[df['year'] == 2018]

for position in positions:
    # 각 포지션에 대해 승리 기여도(war)를 기준으로 내림차순으로 정렬
    sorted_df = df_2018[df_2018['cp'] == position].sort_values(by='war', ascending=False)
    
    # 상위 1명의 선수를 선택
    max_war_player = sorted_df.head(1)
    
    player_name = max_war_player['batter_name'].iloc[0]
    war_value = max_war_player['war'].iloc[0]
    print(f"Player with the highest war by {position} in 2018 is '{player_name}', and player's war is '{war_value:.2f}'")

print()    
    
    
# 3) Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리 기여도), avg (타율), 
# OBP(출루율), and SLG (장타율), which has the highest correlation with salary (연봉)? (15 points)
#▪ Implement code to calculate correlations and print the answer to the above question.

print("3) Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리 기여도), avg (타율), OBP (출루율), and SLG (장타율), which has the highest correlation with salary (연봉)? (15 points)")
# 각 통계 지표와 연봉 간의 상관 계수 계산: 'R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG'와 'salary' 간의 상관 계수를 계산
correlations = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']].corrwith(df['salary'])

# 상관 계수가 높은 순으로 정렬: 계산된 상관 계수를 높은 순으로 정렬
sorted_correlations = correlations.sort_values(ascending=False)

# 정렬된 상관 계수 중 가장 높은 값을 가진 변수
highest_corr_variable = sorted_correlations.head(1).index[0]
# 정렬된 상관 계수 중 가장 높은 값을 가진 변수의 해당 계수
highest_corr_coefficient = sorted_correlations.head(1).iloc[0]

# 정렬된 상관 계수 중 가장 높은 값을 가진 변수와 해당 계수를 출력
print(f"\nVariable with the highest correlation with salary is '{highest_corr_variable}' and its correlation is '{highest_corr_coefficient:.6f}'")
print()
