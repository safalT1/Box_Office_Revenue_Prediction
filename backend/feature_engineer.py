from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering for Movie Revenue Prediction
    """

    def __init__(self):
        self.top_genres = []
        self.top_companies = []
        self.top_countries = []
        self.top_languages = []
        self.label_encoder = LabelEncoder()

    def extract_features_from_list(self, df, column_name, feature_prefix, top_n=15):
        """
        Extract features from list/dictionary columns
        """
        print(f"  Extracting features from {column_name}...")

        # Extract all items from the list
        all_items = []
        for item_list in df[column_name].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []):
            all_items.extend(item_list)

        # Get top N most frequent items
        if all_items:
            top_items = [item for item, count in Counter(all_items).most_common(top_n)]

            # Create binary features for top items
            for item in top_items:
                safe_item_name = item.replace(" ", "_").replace("&", "and").replace(".", "")
                df[f'{feature_prefix}_{safe_item_name}'] = df[column_name].apply(
                    lambda x: 1 if any(
                        i.get('name', '') == item for i in (x if isinstance(x, list) else [])
                    ) else 0
                )

            # Create count feature
            df[f'{column_name}_count'] = df[column_name].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

            return df, top_items
        return df, []

    def create_temporal_features(self, df):
        """
        Create time-based features from release date
        """
        print("  Creating temporal features...")

        # Convert release_date to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        # Extract date components
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day'] = df['release_date'].dt.day
        df['release_dayofweek'] = df['release_date'].dt.dayofweek
        df['release_weekofyear'] = df['release_date'].dt.isocalendar().week
        df['release_quarter'] = df['release_date'].dt.quarter

        # Special time periods
        df['release_is_summer'] = df['release_month'].isin([5, 6, 7, 8]).astype(int)
        df['release_is_holiday'] = df['release_month'].isin([11, 12]).astype(int)
        df['release_is_weekend'] = (df['release_dayofweek'] >= 4).astype(int)

        # Movie age
        current_year = datetime.now().year
        df['movie_age'] = current_year - df['release_year']

        return df

    def create_financial_features(self, df):
        """
        Create financial and efficiency metrics
        """
        print("  Creating financial features...")

        # Handle zero budgets
        df['budget'] = df['budget'].replace(0, df['budget'].median())

        # Basic financial ratios
        df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
        df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)

        # Log transformations for skewed features
        for col in ['budget', 'popularity', 'runtime']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])

        # Efficiency metrics
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            df['rating_power'] = df['vote_average'] * np.log1p(df['vote_count'])

        return df

    def create_production_features(self, df):
        """
        Create production-related features
        """
        print("  Creating production features...")

        # Production scale features
        df['production_companies_count'] = df['production_companies'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        df['production_countries_count'] = df['production_countries'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Language features
        df['spoken_languages_count'] = df['spoken_languages'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Encode original language
        if 'original_language' in df.columns:
            df['original_language_encoded'] = self.label_encoder.fit_transform(
                df['original_language'].fillna('en')
            )

        return df

    def create_cast_crew_features(self, df):
        """
        Create cast and crew-related features
        """
        print("  Creating cast & crew features...")

        # Count features
        df['cast_count'] = df['cast'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        df['crew_count'] = df['crew'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Total people involved
        df['total_people'] = df['cast_count'] + df['crew_count']
        df['people_per_budget'] = df['total_people'] / (df['budget'] + 1)

        # Extract director
        df['has_director'] = df['crew'].apply(
            lambda x: 1 if any(i.get('job') == 'Director' for i in (x if isinstance(x, list) else [])) else 0
        )

        return df

    def create_genre_features(self, df):
        """
        Create genre-related features
        """
        print("  Creating genre features...")

        # Extract genres
        df['genres_list'] = df['genres'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )

        # Genre count
        df['genre_count'] = df['genres_list'].apply(len)

        # Get top genres
        all_genres = []
        for genres in df['genres_list']:
            all_genres.extend(genres)

        self.top_genres = [genre for genre, count in Counter(all_genres).most_common(15)]

        # Create binary features for top genres
        for genre in self.top_genres:
            safe_genre = genre.replace(" ", "_").replace("&", "and")
            df[f'genre_{safe_genre}'] = df['genres_list'].apply(
                lambda x: 1 if genre in x else 0
            )

        return df

    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        """
        print("  Creating interaction features...")

        # Budget interactions
        df['budget_year_interaction'] = df['budget'] * df['release_year']
        df['budget_popularity_interaction'] = df['budget'] * df['popularity']

        # Genre-budget interaction
        for genre in self.top_genres[:5]:  # Top 5 genres
            safe_genre = genre.replace(" ", "_").replace("&", "and")
            df[f'budget_x_genre_{safe_genre}'] = df['budget'] * df.get(f'genre_{safe_genre}', 0)

        return df

    def engineer_all_features(self, df, is_train=True):
        """
        Execute complete feature engineering pipeline
        """
        print(f"\n{'='*60}")
        print(f"🚀 FEATURE ENGINEERING PIPELINE")
        print(f"{'='*60}")

        # Make a copy to avoid modifying original
        df_engineered = df.copy()

        # Step 1: Temporal features
        df_engineered = self.create_temporal_features(df_engineered)

        # Step 2: Financial features
        df_engineered = self.create_financial_features(df_engineered)

        # Step 3: Genre features
        df_engineered = self.create_genre_features(df_engineered)

        # Step 4: Production features
        df_engineered = self.create_production_features(df_engineered)

        # Step 5: Cast & Crew features
        df_engineered = self.create_cast_crew_features(df_engineered)

        # Step 6: Extract features from list columns
        df_engineered, self.top_companies = self.extract_features_from_list(
            df_engineered, 'production_companies', 'company'
        )

        df_engineered, self.top_countries = self.extract_features_from_list(
            df_engineered, 'production_countries', 'country'
        )

        df_engineered, self.top_languages = self.extract_features_from_list(
            df_engineered, 'spoken_languages', 'language'
        )

        # Step 7: Interaction features
        df_engineered = self.create_interaction_features(df_engineered)

        # Step 8: Drop original columns
        columns_to_drop = [
            'belongs_to_collection', 'genres', 'genres_list', 'homepage',
            'imdb_id', 'original_title', 'overview', 'poster_path',
            'production_companies', 'production_countries', 'release_date',
            'spoken_languages', 'tagline', 'title', 'Keywords', 'cast', 'crew',
            'original_language', 'status'
        ]

        # Only drop columns that exist
        existing_columns = [col for col in columns_to_drop if col in df_engineered.columns]
        df_engineered = df_engineered.drop(columns=existing_columns, errors='ignore')

        # Handle any remaining missing values
        df_engineered = df_engineered.fillna(df_engineered.median(numeric_only=True))

        print(f"\n✅ Feature engineering complete!")
        print(f"   Original shape: {df.shape}")
        print(f"   Engineered shape: {df_engineered.shape}")
        print(f"   Features added: {df_engineered.shape[1] - df.shape[1]}")

        return df_engineered