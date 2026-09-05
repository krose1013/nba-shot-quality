import pandas as pd
import time
from nba_api.stats.endpoints import shotchartdetail

def fetch_season_shots():
    print("Fetching 2025-26 shot data from NBA API...")
    
    # Player ID 0 with team_id 0 pulls league-wide shot data
    # Context measure 'FGA' gets all field goal attempts
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            season_nullable='2025-26',
            context_measure_simple='FGA'
        )
        
        df = shot_data.get_data_frames()[0]
        
        # Keep only the features we need for our shot model
        features = [
            'PLAYER_NAME', 'TEAM_NAME', 'PERIOD', 'MINUTES_REMAINING',
            'SECONDS_REMAINING', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC',
            'SHOT_ZONE_AREA', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG'
        ]
        
        cleaned_df = df[features].copy()
        cleaned_df.to_csv("shot_data.csv", index=False)
        print(f"Success! Saved {len(cleaned_df)} shots to 'shot_data.csv'")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_season_shots()