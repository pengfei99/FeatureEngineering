# Feast introduction


**Feast (Feature Store)** is an operational data system for managing and serving machine learning features to 
models in production. Feast is able to serve feature data to models from a low-latency online store 
(for real-time prediction) or from an offline store (for scale-out batch scoring or model training).

Below is the general architecture of feast

![feast-marchitecture.png](../../img/feast-marchitecture.png)

## Feast sdk

Feast provide a sdk and CLI. You can realize all feast operations with the CLI. To see all CLI commands, please
visit this [page](https://docs.feast.dev/reference/feast-cli-commands)

## Data model

**Feast uses a time-series data model to represent data.** This data model is used to interpret feature data in 
data sources in order to build training datasets or when materializing features into an online store.

Below is an example data source with a single entity (driver) and two features (trips_today, and rating).

![feast_data_model.png](../../img/feast_data_model.png)

You can notice each line represent a record which contains 
- timestamp: timestamp of the record
- entity value: is the unique identifier of each record. 
- a list of feature values:


## Feast key Concepts

Feast contains the following key concepts:

- Project : A project is a collection of related features and their data sources. Projects are isolated from each 
            other, and you can’t reuse features from one project in another. As of version 0.19.3, projects were 
            supported to ensure backward compatibility with previous versions of Feast. The concept of projects 
                might change as Feast developers simplify the framework.
- Feature view : A feature view is a group of feature data from a specific data source. Feature views allow you to 
                 consistently define features and their data sources, enabling the reuse of feature groups across a 
                 project. If your features are stored in more than one location, you can specify a feature view for 
                 each location and later join all the features together. Feature views make the addition of new features 
                  to your existing data very easy as well — as you gather new groups of features, you can create 
                 separate feature views for them and then merge them with your old data.
- Data source: In Feast, each feature view has a data source. A data source is where the raw feature data is stored, 
               like a local .parquet file or a s3 bucket. You can have as many data sources as you want, but you 
               can’t mix different types of sources together.
- Feature Service: A feature service is an object that contains features from one or more feature views. 
                You can use feature services to create logically related groups of feature views.
- Dataset : A dataset is a group of feature views and entities. Feast datasets allow engineers to combine data from 
            different feature views for analysis and training.
- Entity : The Feast docs describe entities as a collection of semantically related features. In practical terms, 
           entities can be the individuals or objects that your feature data relates to. As an example, if you 
             have a pneumonia dataset, you can set your patients as your entities and assign unique IDs to them for 
             identification. You can then use the IDs to store and retrieve specific feature values. Feast also uses 
             entities to correctly join data from different feature views.
- Stream feature view
- Feature retrieval
- Point-in-time joins
- Registry
- timestamp: Feast uses timestamps to ensure that features from different sources are joined in the correct 
             chronological order. Primarily, this is so that you can avoid using very old data for
             training or prediction.
### General Organization 

- **Project** is the top-level namespace within Feast. 
- A project contains one or more **feature views**. 
- A feature view contains one or more **features**. 
- A feature view must always have a **data source**, which in turn is used during the generation of training datasets 
    and when materializing feature values into the online store.
- A Feature must relate to at least one **entity** (multiple entities are allowed).

![feast_concept_overview.png](../../img/feast_concept_overview.png)

**Projects provide complete isolation of feature stores at the infrastructure level**. 
It is not possible to retrieve features from multiple projects in a single request. 
We recommend having a single feature store and a single project per environment (dev, staging, prod).

## Feast infrastructure

Below figure shows the main component of feast infrastructure

![feast_infra.png](../../img/feast_infra.png)

- Offline store: The offline store is where you store your features. In the terminology of Feast, the offline 
                 store contains historical features that you can use for analysis or training. Feast can both 
                  retrieve and write data to the offline store. This type of store is “offline” because it is 
                  located outside a Feast environment. **It's for generating training data (training phase)**

- Online store: The online store is where Feast stores features for low-latency access. The online store is designed to 
                complete prediction request (serving phase).

- Feature repo: A feature repository contains the definitions of a Feast feature store. It defines where features 
                are stored, how they should be retrieved, and what they contain. If your offline store is on your 
                local machine, a feature repository can also contain the raw feature data.

- Registry: A registry is a catalog of feature definitions and their metadata. It defines the infrastructure of your 
            feature repository and where feature data comes from.

- Provider: A provider is the implementation of a specific offline or online feature store. Feast has specialized 
            providers for AWS, GCP, and local environments.

## Fest standard workflow

### Step 1: Install feast with required dependencies

```shell
pip install feast
```

Now, you need to create a feast feature store with below command

```shell
feast init <repo-name>
```

Note, by default feast will generate a repo with some demo data and feature view definition. If you don't want them,
you can use the option **-m**. Below example will generate an empty feast store called `breast_cancer`

```shell
feast init -m breast_cancer
```

You should have below generated files

```text
feast
├── breast_cancer
│   └── feature_store.yaml
```

### Step 2: Prepare fest feature store infrastructure

Edit `feature_store.yaml` to configure the infrastructure for offline store, online store, registry, etc.

Base on what you have passed as provider (default is local, you can choose gcp, aws, etc.), the content of the generated
feature_store.yaml is different. Below `feature_store.yaml` is generated by using the default local provider.

```yaml
project: breast_cancer
registry: /path/to/registry.db
provider: local
online_store:
    path: /path/to/online_store.db
```

It contains the following parameters:

- project: the name of the current project.

- registry: the path of the registry file where Feast will store your feature definitions.

- provider: the target environment where the features are stored — in our case, a local environment.

- online_store: the environment that Feast will use to store features for low-latency inference.

To make the feature store work, we need to edit the values for registry and online_store with a valid path, for example
`/tmp/feature_store/data`:

So, the new `feature_store.yaml` will look like:

```yaml
project: breast_cancer
registry: /tmp/feature_store/data/registry.db
provider: local
online_store:
    path: /tmp/feature_store/data/online_store.db
```

With above definition, Feast will store 
- **online data** in `/tmp/feature_store/data/online_store.db`
- **feature registry** in `/tmp/feature_store/data/registry.db` 

Note by default feast uses sqlite as backend to store data, not recommended for production

### Step 3: Define features

Now, we need to create a definition script (python file) for features, name the script whatever you want. Here, I call
it `feature_definition.py`. 

One important point, **make sure that there are no other Python scripts that defines feature in the directory**. 
Otherwise, Feast might be unable to correctly register your features.

Below is an example of `feature_definition.py` of `02.Breast_cancer`

```python
# Importing dependencies
from feast import Entity, Feature, FeatureView, FileSource, ValueType, Field
from feast.types import Float32, Int64, Int32
from datetime import timedelta

# Declaring an entity for the dataset
# Because all the features in this project describe the properties of a patient, so naturally the entity is patient
patient = Entity(
    name="patient_id",
    join_keys=["patient_id"],
    value_type=ValueType.INT64,
    description="The ID of the patient")

# Declaring the source of the first set of features
f_source1 = FileSource(
    path="/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/raw_data/data1.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

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
        Field(name="mean radius", dtype=Float32),
        Field(name="mean texture", dtype=Float32),
        Field(name="mean perimeter", dtype=Float32),
        Field(name="mean area", dtype=Float32),
        Field(name="mean smoothness", dtype=Float32),
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
    created_timestamp_column="created",
)

# Defining the second set of features
df2_fv = FeatureView(
    name="df2_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="mean compactness", dtype=Float32),
        Field(name="mean concavity", dtype=Float32),
        Field(name="mean concave points", dtype=Float32),
        Field(name="mean symmetry", dtype=Float32),
        Field(name="mean fractal dimension", dtype=Float32),
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
    created_timestamp_column="created",
)

# Defining the third set of features
df3_fv = FeatureView(
    name="df3_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="radius error", dtype=Float32),
        Field(name="texture error", dtype=Float32),
        Field(name="perimeter error", dtype=Float32),
        Field(name="area error", dtype=Float32),
        Field(name="smoothness error", dtype=Float32),
        Field(name="compactness error", dtype=Float32),
        Field(name="concavity error", dtype=Float32),
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
    created_timestamp_column="created",
)

# Defining the fourth set of features
df4_fv = FeatureView(
    name="df4_feature_view",
    entities=[patient],
    ttl=timedelta(days=3),
    schema=[
        Field(name="concave points error", dtype=Float32),
        Field(name="symmetry error", dtype=Float32),
        Field(name="fractal dimension error", dtype=Float32),
        Field(name="worst radius", dtype=Float32),
        Field(name="worst texture", dtype=Float32),
        Field(name="worst perimeter", dtype=Float32),
        Field(name="worst area", dtype=Float32),
        Field(name="worst smoothness", dtype=Float32),
        Field(name="worst compactness", dtype=Float32),
        Field(name="worst concavity", dtype=Float32),
        Field(name="worst concave points", dtype=Float32),
        Field(name="worst symmetry", dtype=Float32),
        Field(name="worst fractal dimension", dtype=Float32),
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
    created_timestamp_column="created",
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

```

You can notice, in the above file, we define the:
- entities, 
- data sources,
- feature views, 
- feature services


```shell
feast apply
```
Run the above command register the features in your feature store. Feast will register feature and data source 
definitions in your feature repository’s registry.

If you want to delete the feast feature store, you can run 
```shell
feast teardown
```

You can now check the feature views and entities registered in the repository.

```shell
feast entities list
```

It shows registered entities:

```text
NAME        DESCRIPTION            TYPE
patient_id  The ID of the patient  ValueType.INT64

```

Run below command to show the registered feature views:
```shell
feast feature-views list
```

```text
NAME                 ENTITIES        TYPE
df2_feature_view     {'patient_id'}  FeatureView
target_feature_view  {'patient_id'}  FeatureView
df3_feature_view     {'patient_id'}  FeatureView
df1_feature_view     {'patient_id'}  FeatureView
df4_feature_view     {'patient_id'}  FeatureView

```

## Step 4. Retrieving features and creating a training dataset

After the feature repository is initialized, You can join different feature views together to create a dataset that 
you can then analyze, save, or use for training. In the language of Feast, **use your feature views to fetch feature 
data from your offline stores (data sources) is called historical retrieval**.



Let’s start by creating a dataset for training.

Check `create_training_data.py`

## Step 5. Using the dataset to train a model

To start training, the `train.py` will handle training for us.
 

· After you train and deploy a model, you can fetch (materialize) the latest feature values from the offline store for inference. When you materialize features, they are stored in the online store for performance reasons.

· As you add new features to your offline stores, you can continuously materialize them to keep your online store fresh. Additionally, if necessary, you can do historical retrieval of your new and old features, join them all together, and retrain your model on the new dataset.