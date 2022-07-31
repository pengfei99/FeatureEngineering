# Importing dependencies
from feast import Entity, Feature, FeatureView, FileSource, ValueType, Field, FeatureService
from feast.types import Float32, Int64, Int32
from datetime import timedelta

# Declaring an entity for the dataset
# Because all the features in this project describe the properties of a patient, so naturally the entity is patient
patient = Entity(
    name="patient_id",
    join_keys=["patient_id"],
    # value_type=ValueType.INT64,
    description="The ID of the patient")

# Declaring the source of the first set of features
f_source1 = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data1.parquet",
    timestamp_field="event_timestamp",
)
# Known issues for timestamp definition https://github.com/feast-dev/feast/issues/2865

# Defining the first set of features

df1_fv = FeatureView(
    # the name of the feature view
    name="df1_feature_view",
    # the entity of this feature view. can be empty
    entities=[patient],
    #  The time that the features in the feature view should be cached for. In our case, ttl is set to three days.
    #  Feast uses ttl to make sure that only new features are served to the model for inference — you’ll understand
    #  better later.
    ttl=timedelta(days=3),
    # features of this feature view
    schema=[
        Field(name="mean_radius", dtype=Float32),
        Field(name="mean_texture", dtype=Float32),
        Field(name="mean_perimeter", dtype=Float32),
        Field(name="mean_area", dtype=Float32),
        Field(name="mean_smoothness", dtype=Float32),
        Field(name="patient_id", dtype=Int64)
    ],
    online=True,
    # source of this feature view
    source=f_source1,
    tags={},
)

# Declaring the source of the second set of features
f_source2 = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data2.parquet",
    timestamp_field="event_timestamp",
)

# Defining the second set of features
df2_fv = FeatureView(
    name="df2_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="mean_compactness", dtype=Float32),
        Field(name="mean_concavity", dtype=Float32),
        Field(name="mean_concave_points", dtype=Float32),
        Field(name="mean_symmetry", dtype=Float32),
        Field(name="mean_fractal_dimension", dtype=Float32),
        Field(name="patient_id", dtype=Int64)
    ],
    online=True,
    source=f_source2,
    tags={},
)

# Declaring the source of the third set of features
f_source3 = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data3.parquet",
    timestamp_field="event_timestamp",
)

# Defining the third set of features
df3_fv = FeatureView(
    name="df3_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="radius_error", dtype=Float32),
        Field(name="texture_error", dtype=Float32),
        Field(name="perimeter_error", dtype=Float32),
        Field(name="area_error", dtype=Float32),
        Field(name="smoothness_error", dtype=Float32),
        Field(name="compactness_error", dtype=Float32),
        Field(name="concavity_error", dtype=Float32),
        Field(name="patient_id", dtype=Int64)
    ],
    online=True,
    source=f_source3,
    tags={},
)

# Declaring the source of the fourth set of features
f_source4 = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data4.parquet",
    timestamp_field="event_timestamp",
)

# Defining the fourth set of features
df4_fv = FeatureView(
    name="df4_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="concave_points_error", dtype=Float32),
        Field(name="symmetry_error", dtype=Float32),
        Field(name="fractal_dimension_error", dtype=Float32),
        Field(name="worst_radius", dtype=Float32),
        Field(name="worst_texture", dtype=Float32),
        Field(name="worst_perimeter", dtype=Float32),
        Field(name="worst_area", dtype=Float32),
        Field(name="worst_smoothness", dtype=Float32),
        Field(name="worst_compactness", dtype=Float32),
        Field(name="worst_concavity", dtype=Float32),
        Field(name="worst_concave_points", dtype=Float32),
        Field(name="worst_symmetry", dtype=Float32),
        Field(name="worst_fractal_dimension", dtype=Float32),
        Field(name="patient_id", dtype=Int64)
    ],
    online=True,
    source=f_source4,
    tags={},
)

# Declaring the source of the targets
target_source = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data_target.parquet",
    timestamp_field="event_timestamp",
)

# Defining the targets
target_fv = FeatureView(
    name="target_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="target", dtype=Int32),
        Field(name="patient_id", dtype=Int64)
    ],
    online=True,
    source=target_source,
    tags={},
)

# define a feature service
patient_stats_fs = FeatureService(
    name="patient_stats", features=[df1_fv, df2_fv, df3_fv, df4_fv, target_fv]
)


