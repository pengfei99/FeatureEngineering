# Importing dependencies
import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage


# build the entity DataFrame
# The entity dataframe is the dataframe we want to enrich with feature values
# The joining id for all feature views
entity_df = pd.read_parquet(path="raw_data/data_target.parquet")

print(entity_df.head())

# Configure store path
store_path = "/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/breast_cancer"

# Getting our FeatureStore
store = FeatureStore(repo_path=store_path)



# Getting the indicated historical features
# and joining them with our entity DataFrame
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "df1_feature_view:mean_radius",
        "df1_feature_view:mean_texture",
        "df1_feature_view:mean_perimeter",
        "df1_feature_view:mean_area",
        "df1_feature_view:mean_smoothness",
        "df2_feature_view:mean_compactness",
        "df2_feature_view:mean_concavity",
        "df2_feature_view:mean_concave_points",
        "df2_feature_view:mean_symmetry",
        "df2_feature_view:mean_fractal_dimension",
        # "df3_feature_view:radius_error",
        # "df3_feature_view:texture_error",
        # "df3_feature_view:perimeter_error",
        # "df3_feature_view:area_error",
        # "df3_feature_view:smoothness_error",
        # "df3_feature_view:compactness_error",
        # "df3_feature_view:concavity_error",
        # "df4_feature_view:concave_points_error",
        # "df4_feature_view:symmetry_error",
        # "df4_feature_view:fractal_dimension_error",
        # "df4_feature_view:worst_radius",
        # "df4_feature_view:worst_texture",
        # "df4_feature_view:worst_perimeter",
        # "df4_feature_view:worst_area",
        # "df4_feature_view:worst_smoothness",
        # "df4_feature_view:worst_compactness",
        # "df4_feature_view:worst_concavity",
        # "df4_feature_view:worst_concave_points",
        # "df4_feature_view:worst_symmetry",
        # "df4_feature_view:worst_fractal_dimension"
    ]
)

# preview the training data
pdf = training_data.to_df()
print(pdf.head())

# Storing the dataset as a local file
dataset = store.create_saved_dataset(
    from_=training_data,
    name="breast_cancer_dataset",
    storage=SavedDatasetFileStorage("breast_cancer/data/breast_cancer_dataset.parquet")
)
