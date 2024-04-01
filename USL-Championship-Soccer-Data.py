from itscalledsoccer.client import AmericanSoccerAnalysis
# https://github.com/American-Soccer-Analysis/itscalledsoccer
# https://american-soccer-analysis.github.io/itscalledsoccer/
import pandas as pd
import numpy as np
import logging

# Initialize ASA client
client = AmericanSoccerAnalysis()

class ASADataProcessor:
    """
    This class handles fetching and processing of ASA soccer competition data, providing a structured approach
    to access, transform, and prepare soccer data for analysis or application use cases.
    
    Attributes:
        client (AmericanSoccerAnalysis): An instance of the ASA API client for data retrieval.
    """
    
    # Constants for data processing
    COMPETITION = 'uslc'
    DUPLICATE_SUFFIX = '_dup'
    TEAM_ID = 'team_id'
    PLAYER_ID = 'player_id'
    STADIUM_ID = 'stadium_id'
    MANAGER_ID = 'manager_id'
    REFEREE_ID = 'referee_id'
    HOME_TEAM_ID = 'home_team_id'
    HOME_MANAGER_ID = 'home_manager_id'
    AWAY_TEAM_ID = 'away_team_id'
    AWAY_MANAGER_ID = 'away_manager_id'
    GAME_ID = 'game_id'
    
    def __init__(self, client):
        """
        Initializes the ASADataProcessor with a specific ASA API client.
        
        Args:
            client (AmericanSoccerAnalysis): An initialized client connected to the ASA API.
        """
        self.client = client

    def calculate_result(self, row):
        """
        Determines the result of a game based on the scores.
        
        Args:
            row (pd.Series): A data row representing a single game, including 'home_score' and 'away_score'.
        
        Returns:
            str: 'win', 'loss', or 'draw' from the perspective of the home team.
        """
        if row['home_score'] > row['away_score']: return 'win'
        elif row['home_score'] < row['away_score']: return 'loss'
        else: return 'draw'

    def calculate_points(self, result):
        """
        Calculates the points earned by a team based on the game result.
        
        Args:
            result (str): The result of the game ('win', 'loss', 'draw').
        
        Returns:
            int: The number of points earned (3 for a win, 1 for a draw, 0 for a loss).
        """
        return 3 if result == 'win' else 1 if result == 'draw' else 0

    def add_player_names(self, df, players_df):
        """
        Adds player names to a DataFrame by merging with a players DataFrame on 'player_id'.
        
        Args:
            df (pd.DataFrame): The DataFrame to which player names will be added.
            players_df (pd.DataFrame): The DataFrame containing player IDs and names.
        
        Returns:
            pd.DataFrame: The original DataFrame enriched with player names.
        """
        return df.merge(players_df[[self.PLAYER_ID, 'player_name']], on=self.PLAYER_ID, how='left')

    def add_team_names(self, df, teams_df):
        """
        Adds team names to a DataFrame by merging with a teams DataFrame on 'team_id'.
        
        Args:
            df (pd.DataFrame): The DataFrame to which team names will be added.
            teams_df (pd.DataFrame): The DataFrame containing team IDs and names.
        
        Returns:
            pd.DataFrame: The original DataFrame enriched with team names.
        """
        if self.TEAM_ID in df and any(isinstance(x, list) for x in df[self.TEAM_ID].dropna()):
            df_exploded = df.explode(self.TEAM_ID)
            df_merged = df_exploded.merge(teams_df[[self.TEAM_ID, 'team_name']], on=self.TEAM_ID, how='left')
            return df_merged
        else:
            return df.merge(teams_df[[self.TEAM_ID, 'team_name']], on=self.TEAM_ID, how='left')

    def add_stadium_names(self, df, stadium_df):
        """
        Adds stadium names to a DataFrame by merging with a stadium DataFrame on 'stadium_id'.
        
        Args:
            df (pd.DataFrame): The DataFrame to which stadium names will be added.
            stadium_df (pd.DataFrame): The DataFrame containing stadium IDs and names.
        
        Returns:
            pd.DataFrame: The original DataFrame enriched with stadium names.
        """
        return df.merge(stadium_df[[self.STADIUM_ID, 'stadium_name']], on=self.STADIUM_ID, how='left')

    def add_manager_names(self, df, managers_df):
        """
        Adds manager names to a DataFrame by merging with a managers DataFrame on 'manager_id'.
        
        Args:
            df (pd.DataFrame): The DataFrame to which manager names will be added.
            managers_df (pd.DataFrame): The DataFrame containing manager IDs and names.
        
        Returns:
            pd.DataFrame: The original DataFrame enriched with manager names.
        """
        return df.merge(managers_df[[self.MANAGER_ID, 'manager_name']], on=self.MANAGER_ID, how='left')

    def add_referee_names(self, df, referees_df):
        """
        Adds referee names to a DataFrame by merging with a referees DataFrame on 'referee_id'.
        
        Args:
            df (pd.DataFrame): The DataFrame to which referee names will be added.
            referees_df (pd.DataFrame): The DataFrame containing referee IDs and names.
        
        Returns:
            pd.DataFrame: The original DataFrame enriched with referee names.
        """
        return df.merge(referees_df[[self.REFEREE_ID, 'referee_name']], on=self.REFEREE_ID, how='left')

    def merge_team_statistics(self, teams):
        """
        Merges team goals added, xGoals, and xPass statistics with the teams DataFrame.

        Args:
            teams (pd.DataFrame): The DataFrame containing team information.

        Returns:
            pd.DataFrame: The teams DataFrame enriched with statistics data.
        """
        try:
            # Fetching team statistics.
            team_goals_added = self.client.get_team_goals_added(self.COMPETITION).copy()
            team_xgoals = self.client.get_team_xgoals(self.COMPETITION).copy()
            team_xpass = self.client.get_team_xpass(self.COMPETITION).copy()

            # Merging statistics with the teams DataFrame
            teams = pd.merge(teams, team_goals_added, on=self.TEAM_ID, how='left', suffixes=('', self.DUPLICATE_SUFFIX)).drop(columns=[col for col in teams.columns if col.endswith(self.DUPLICATE_SUFFIX)])
            teams = pd.merge(teams, team_xgoals, on=self.TEAM_ID, how='left', suffixes=('', self.DUPLICATE_SUFFIX)).drop(columns=[col for col in teams.columns if col.endswith(self.DUPLICATE_SUFFIX)])
            teams = pd.merge(teams, team_xpass, on=self.TEAM_ID, how='left', suffixes=('', self.DUPLICATE_SUFFIX)).drop(columns=[col for col in teams.columns if col.endswith(self.DUPLICATE_SUFFIX)])

            expected_columns = ['avg_vertical_distance_diff']
            for col in expected_columns:
                if col not in teams.columns:
                    teams[col] = 0  # Assuming 0 as a placeholder value; adjust as appropriate

        except Exception as e:
            logging.error(f"Error merging team statistics for {self.COMPETITION}: {str(e)}")

        return teams

    def fetch_data(self, competition):
        """
        Fetches and processes all relevant soccer competition data for a specified competition.
        This includes teams, players, games, and associated statistics and metadata.
        
        Args:
            competition (str): The competition identifier for which data is to be fetched and processed.
        
        Returns:
            dict: A dictionary containing processed data tables as pandas DataFrames.
        """
        data = {}
        try:
            # Set the competition to a class variable to ensure consistency across methods
            competition = ASADataProcessor.COMPETITION
            
            # Fetch basic dimension (DIM) tables with error handling for each fetch operation
            try:
                players = self.client.players[self.client.players['competition'] == competition].copy()
            except Exception as e:
                logging.error(f"Error fetching players data for {competition}: {e}")
                players = pd.DataFrame()

            try:
                teams = self.client.teams[self.client.teams['competition'] == competition].copy()
            except Exception as e:
                logging.error(f"Error fetching teams data for {competition}: {e}")
                teams = pd.DataFrame()

            # Merge team statistics to enrich teams data
            teams = self.merge_team_statistics(teams)

            # Convert team_name and team_abbreviation to categorical types
            teams['team_name'] = teams['team_name'].astype('category')
            teams['team_abbreviation'] = teams['team_abbreviation'].astype('category')

            # Reorganizing the columns for clarity
            teams = teams[[
                self.TEAM_ID, 'team_name', 'team_short_name', 'team_abbreviation', 'competition', 'minutes', 'data',
                'count_games', 'shots_for', 'shots_against', 'goals_for', 'goals_against', 'goal_difference',
                'xgoals_for', 'xgoals_against', 'xgoal_difference', 'goal_difference_minus_xgoal_difference',
                'points', 'xpoints', 'attempted_passes_for', 'pass_completion_percentage_for',
                'xpass_completion_percentage_for', 'passes_completed_over_expected_for',
                'passes_completed_over_expected_p100_for', 'avg_vertical_distance_for',
                'attempted_passes_against', 'pass_completion_percentage_against',
                'xpass_completion_percentage_against', 'passes_completed_over_expected_against',
                'passes_completed_over_expected_p100_against', 'avg_vertical_distance_against',
                'passes_completed_over_expected_difference', 'avg_vertical_distance_diff'
            ]]
            
            # Fetch stadium data with error handling
            try:
                stadium = self.client.stadia[self.client.stadia['competition'] == competition].copy()
            except Exception as e:
                logging.error(f"Error fetching stadium data for {competition}: {e}")
                stadium = pd.DataFrame()

            # Fetch managers data with error handling
            try:
                mgrs = self.client.managers[self.client.managers['competition'] == competition].copy()
            except Exception as e:
                logging.error(f"Error fetching managers data for {competition}: {e}")
                mgrs = pd.DataFrame()
            
            # Convert manager_name to categorical type
            mgrs['manager_name'] = mgrs['manager_name'].astype('category')
            
            # Fetch referees data with error handling
            try:
                refs = self.client.get_referees(competition).copy()
            except Exception as e:
                logging.error(f"Error fetching referees data for {competition}: {e}")
                refs = pd.DataFrame()

            # Convert referee_name to categorical type
            refs['referee_name'] = refs['referee_name'].astype('category')
            
            # Process goalkeeper specific FACT tables
            try:
                gk_xG = self.add_player_names(self.client.get_goalkeeper_xgoals(competition).copy(), players)
                gk_xG = self.add_team_names(gk_xG, teams)
            except Exception as e:
                logging.error(f"Error processing goalkeeper xGoals for {competition}: {e}")
                gk_xG = pd.DataFrame()

            try:
                gk_G_added = self.client.get_goalkeeper_goals_added(competition).copy()
            except Exception as e:
                logging.error(f"Error fetching goalkeeper goals added for {competition}: {e}")
                gk_G_added = pd.DataFrame()

            # Merge player statistics FACT tables with players, with error handling
            try:
                player_G_added = self.client.get_player_goals_added(competition).copy()
            except Exception as e:
                logging.error(f"Error fetching player goals added for {competition}: {e}")
                player_G_added = pd.DataFrame()

            try:
                player_xG = self.client.get_player_xgoals(competition).copy()
            except Exception as e:
                logging.error(f"Error fetching player xGoals for {competition}: {e}")
                player_xG = pd.DataFrame()

            try:
                player_xP = self.client.get_player_xpass(competition).copy()
            except Exception as e:
                logging.error(f"Error fetching player xPass for {competition}: {e}")
                player_xP = pd.DataFrame()

            # Enhance player data with merged statistics and team information
            for df in [player_G_added, player_xG, player_xP]:
                try:
                    players = pd.merge(players, df, on='player_id', how='left', suffixes=('', '_dup'))
                    players = players[[c for c in players.columns if not c.endswith('_dup')]]
                except Exception as e:
                    logging.error(f"Error merging player data with {df} for {competition}: {e}")
            
            # Convert relevant columns to categorical types after all merging operations
            players['nationality'] = players['nationality'].astype('category')
            players['primary_broad_position'] = players['primary_broad_position'].astype('category')
            players['primary_general_position'] = players['primary_general_position'].astype('category')
            players['secondary_broad_position'] = players['secondary_broad_position'].astype('category')
            players['secondary_general_position'] = players['secondary_general_position'].astype('category')
            
            # Enhance player data with team information and handle errors         
            try:
                players = self.add_team_names(players.explode('team_id'), teams)
            except Exception as e:
                logging.error(f"Error enhancing player data with team information for {competition}: {e}")
                
            # Reorganize players DataFrame and handle potential errors
            try:
                players = players[[
                    'player_id', 'player_name', 'birth_date', 'nationality', 'height_ft', 'height_in', 'weight_lb', 
                    'primary_broad_position', 'primary_general_position', 'secondary_broad_position', 'secondary_general_position', 
                    'team_id', 'team_name', 'season_name', 'competition', 'general_position', 'minutes_played', 'shots', 
                    'shots_on_target', 'goals', 'xgoals', 'xplace', 'goals_minus_xgoals', 'key_passes', 'primary_assists', 
                    'xassists', 'primary_assists_minus_xassists', 'xgoals_plus_xassists', 'points_added', 'xpoints_added', 
                    'attempted_passes', 'pass_completion_percentage', 'xpass_completion_percentage', 
                    'passes_completed_over_expected', 'passes_completed_over_expected_p100', 'avg_distance_yds', 
                    'avg_vertical_distance_yds', 'share_team_touches', 'count_games', 'data'
                ]]
            except Exception as e:
                logging.error(f"Error reorganizing player data for {competition}: {e}")

            # Split players into field players and goalkeepers
            field_players = players[(players['primary_broad_position'] != 'GK')]
            goalkeepers = players[(players['primary_broad_position'] == 'GK')]

            for df in [gk_xG, gk_G_added]:
                goalkeepers = pd.merge(goalkeepers, df, on=self.PLAYER_ID, how='left', suffixes=('', self.DUPLICATE_SUFFIX))
                goalkeepers = goalkeepers[[c for c in goalkeepers.columns if not c.endswith(self.DUPLICATE_SUFFIX)]]
     
            # Reorganize and rename columns for 'goalkeepers'
            goalkeepers = goalkeepers[[
                'player_id', 'player_name', 'birth_date', 'nationality', 'height_ft', 'height_in', 'weight_lb', 
                'primary_broad_position', 'primary_general_position', 'secondary_broad_position', 'secondary_general_position', 
                'team_id', 'team_name', 'season_name', 'competition', 'general_position', 'minutes_played', 'shots', 
                'shots_on_target', 'goals', 'xgoals', 'xplace', 'goals_minus_xgoals', 'key_passes', 'primary_assists', 
                'xassists', 'primary_assists_minus_xassists', 'xgoals_plus_xassists', 'points_added', 'xpoints_added', 
                'attempted_passes', 'pass_completion_percentage', 'xpass_completion_percentage', 
                'passes_completed_over_expected', 'passes_completed_over_expected_p100', 'avg_distance_yds', 
                'avg_vertical_distance_yds', 'share_team_touches', 'count_games', 'data',
                'minutes_played', 'shots_faced', 'goals_conceded', 'saves', 'share_headed_shots', 'xgoals_gk_faced',
                'goals_minus_xgoals_gk', 'goals_divided_by_xgoals_gk'
            ]]

            # Process games data, enhancing with stadium, referee, and team information, and handle errors
            try:
                games = self.client.get_games(self.COMPETITION).copy()
                games = self.add_stadium_names(games, stadium)
                games = self.add_referee_names(games, refs)
                
                # Merge home and away team information
                games = games.merge(teams[['team_id', 'team_name', 'team_abbreviation']], left_on=self.HOME_TEAM_ID, right_on=self.TEAM_ID)
                games = games.merge(teams[['team_id', 'team_name', 'team_abbreviation']], left_on=self.AWAY_TEAM_ID, right_on=self.TEAM_ID, suffixes=('_home', '_away'))
                
                # Merge home and away manager information
                games = games.merge(mgrs[['manager_id', 'manager_name']], left_on=self.HOME_MANAGER_ID, right_on=self.MANAGER_ID)
                games = games.merge(mgrs[['manager_id', 'manager_name']], left_on=self.AWAY_MANAGER_ID, right_on=self.MANAGER_ID, suffixes=('_home', '_away'))
                
                # Calculate home and away results and points
                games['result_home'] = games.apply(self.calculate_result, axis=1)
                games['points_home'] = games['result_home'].apply(self.calculate_points)
                games['result_away'] = games['result_home'].apply(lambda x: 'win' if x == 'loss' else 'loss' if x == 'win' else 'draw')
                games['points_away'] = games['result_away'].apply(self.calculate_points)
            except Exception as e:
                logging.error(f"Error processing games data for {competition}: {e}")
            
            # Merge game xGoals data with error handling
            try:
                games_xG = self.client.get_game_xgoals(competition).copy()
                games = pd.merge(games, games_xG, on=self.GAME_ID, suffixes=('', '_xG'), how='left')
            except Exception as e:
                logging.error(f"Error merging game xGoals data for {competition}: {e}")
                games_xG = pd.DataFrame()  # Fallback to prevent errors in subsequent processing

            # Error handling for creating match name and dropping unused columns
            try:
                games['match_name'] = games['team_abbreviation_home'] + " v " + games['team_abbreviation_away']
                games.drop(columns=['date_time_utc_xG', 'home_team_id_xG', 'away_team_id_xG', 'last_updated_utc'], inplace=True)
            except Exception as e:
                logging.error(f"Error creating match names or dropping columns for {competition}: {e}")
            
            # Error handling for selecting columns
            try:
                games = games[[
                    'game_id', 'date_time_utc', 'season_name', 'matchday', 'match_name', 'attendance', 'knockout_game', 
                    'extra_time', 'penalties', 'home_penalties', 'away_penalties', 'expanded_minutes', 'home_team_id', 
                    'team_name_home', 'team_abbreviation_home', 'away_team_id', 'team_name_away', 'team_abbreviation_away', 
                    'home_score', 'away_score', 'home_goals', 'away_goals', 'home_team_xgoals', 'away_team_xgoals', 
                    'home_player_xgoals', 'away_player_xgoals', 'goal_difference', 'team_xgoal_difference', 
                    'player_xgoal_difference', 'final_score_difference', 'home_xpoints', 'away_xpoints', 'result_home', 
                    'points_home', 'result_away', 'points_away', 'referee_id', 'referee_name', 'stadium_id', 'stadium_name', 
                    'home_manager_id', 'manager_name_home', 'away_manager_id', 'manager_name_away'
                ]]
                
            except Exception as e:
                logging.error(f"Error selecting and renaming columns in games data for {competition}: {e}")

            # Compile the processed tables into a data dictionary
            data = {
                'games': games,
                'players': players,
                'field_players': field_players,
                'goalkeepers': goalkeepers,
                'teams': teams,
                'stadium': stadium,
                'mgrs': mgrs,
                'refs': refs
            }

        except Exception as e:
            logging.error(f"General error in fetch_data for {competition}: {str(e)}", exc_info=True)

        return data



###  TO DO  /  TO RESEARCH  ###


#Check team name merge RE: lists
    #players = players.merge(teams[['team_id', 'team_name', 'team_abbreviation']], on='team_id', how='left')

#ELO calculation?
    #Supplemental file to follow

#ENHANCEMENTS
    #Develop Unit Tests: Creating unit tests for functions, especially those performing data transformations, ensures they work as intended and remain stable through future updates. Testing individual components isolates and identifies issues more efficiently.
    #Add Integration Tests: Since the script relies on external APIs, implementing integration tests verifies the end-to-end data fetching and processing pipeline, ensuring the entire system functions correctly together and with external dependencies.



if __name__ == "__main__":
    asa_client = AmericanSoccerAnalysis()
    processor = ASADataProcessor(asa_client)
    datasets = processor.fetch_data(ASADataProcessor.COMPETITION)

    # Setting display options for better readability of output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    try:
        games = datasets['games']
        players = datasets['players']
        field_players = datasets['field_players']
        goalkeepers = datasets['goalkeepers']
        teams = datasets['teams']
        stadium = datasets['stadium']
        mgrs = datasets['mgrs']
        refs = datasets['refs']

        #print(players[players['team_name'] == 'Louisville City FC'].head())
        #print(stadium[stadium['postal_code'].isna() | (stadium['postal_code'] == '')].head())
        #print(stadium.head())
        #print(teams.head())
    except KeyError as e:
        logging.error(f"Dataset key error: {e}")