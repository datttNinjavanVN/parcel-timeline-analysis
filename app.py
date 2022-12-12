from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
import pyarrow
from streamlit_pandas_profiling import st_profile_report
import os 
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pickle

df = pd.read_parquet("clean.pq")

with st.sidebar: 
  st.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHcAAAB3CAMAAAAO5y+4AAAA/1BMVEXuACr////uDy7uCCztAAAAIR/1ACruACbtABMkHyDuAB7uACMAAADxACqJFyTtABYOICDvOD/zfH75wsP+8/P1np8fHyD71dX/ACuYFSUFICAaHyDiCCnwTlzxYWTtAAwrHyAcFxhKHSHCAC7wVlryc3nFDSf3q7D5ur30jpHf3t7vIzraCSn96etxGSOzESb83uDAABv3mJ/pwMPIS1i9AADAABHCKDPlrrLVhovjl53LvL27tbbcsLTwQ0zcvcFrcXEnMTHWcHrQZG1VX17MQEmDg4PGxsacnZ1TUVGmFCVgXl8zMDFGUE9nGiN9GSM2HiBbGyKIMjd9AACdAAANAww2AAAEb0lEQVRoge3bbVfiOBQA4KYk6Rt9EW0pLVJkoBQoAr4ig9UZBkF3RnF3//9v2ZtWzqAyK8fj6J6zyYeGlNjHe5vc1g8K4sc04WNajrvc5S53uctd7nKXu9zlLne5y13ucpe73OUudx8PJTXA2JTe25U7drQdNXfx+7rBweHRcXhyOoy0ZyGrz7IgmerbuNg+Oxl9Ho/Pj462n15SK7Q6j2E1OWgEb+FqleHg4jKfz38Zf50cyI+mBRFCO4+yL23tINR6dcQrrtw8HX7L70HLj6fDx1dUuwihZDVgtQFnts03cPHhdHzVZm12OTqraKvTgiZC3efxFt4iXnw+uWpfzlH7eja/OIIUSpIEGytbPWajwEbslGamS0zdalXS+6uZcrbmdF3PrpR90Cml+s/xcrAm3q9Xs3EbofnO5cWkoOY6nY5mFuwGBrAjih1JgjMiTuxWh+UXTomSYMq7tt3aCiShDC29HPsATHhTuokpjEUYU7GXDda4w+k4/wdr338Mz+pmAW5gA5KJupqEi9BjDIcIVhgsKFnagq4pyy02g91pPSaE3MKly6x3e1XieR47Qe8IWYSEeGSfrnNN+/T8YV3NpoemWkDLVjTxduqCkTGorjLXxpXlnGZA932F5ATaIwqJY+J7hFgKMSg1LGVBLF9RSImucaVklO0jf3Y2teXM3W4ySPzpor7dZwxOXTNoITtJIBtdkYYA9ii99Z2qS0P/plw2LEcRmetUS4avOAt3jQuLdvppCHXjYnI8SqTUbWIMKxntrsQrpukuPriCqeJOwvZUXdOrjn9Ly1XHu4Hs6m6uXGahMxeOLvw+/ro8Q8DDk8HpdBqGR3YgpG5dS/fpilvEuazLXDmJulnqKxotWY6fgzR7okDLpdrCURyFhKnrCrTkKX5unQv1eRJ+gjYYQk3cyMXsW9Ttpq7Owgv3fevW1WNYVv6ipmzkSuKQueFZA/bJZi6IxURUU1eg975/v3CsUKc137+NRbpZvFD8RgOAI7Y9N3JVVrQwLmSuDjn2HWdBBdcD0KUx2cwV5OgUspxom7os3p1sS7PCSmH9Kh7sFrfmO4uSseH9hZqXDAdhlJa/X7j9zO0/uNkm7y5dg1iWFeuCHsKG9Ui1ajGXWKlLLO8Xru4efz4e/amx7a3Vm83mlqTtQpfILTgGARzgAcm6lvywf+tRv9/EbA7UaD02DCMtDjS+q9UMAYYx7RnGHWysEAbrXT2+v0KT7uy6l8Yuy7K07FQ4QkmTZfaUgM40mXtgwtdBIEtmOjV7FmQ1Kf2UPRr07Jy+/OqZS40vY1RE7cu9svDvTeoUWOYbb/EcBPev2fy6Pd/LL15ytawuv/7N84nbRu3veaijL7ppvWi9+nXjWbwIXe3B6n/R3S0Wo8rrX+uerKtBbQ6ubxkv/pyGMdZenLWhC0vu7x/fvPvVF4Pf1J7+fUSlu9DV1079rS7I76D+Z/4e5C53uctd7nKXu9zlLne5y13ucpe73OUud7n7v3Q/6P/p/gHq7IGd1+/SEwAAAABJRU5ErkJggg==")
  st.title("Parcel Timeline Analysis")
  choice = st.radio("Navigation", ["Exploratory Data Analysis","Model", "Prediction"])
  st.info("This project application builds a model to predict late SLA parcels.")

if choice == "Exploratory Data Analysis": 
  st.title("Exploratory Data Analysis")
  #st.dataframe(df)
  #profile_df = df.profile_report()
  #st_profile_report(profile_df)

if choice == "Model":
  cat_cols = ["origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days"]
  cat_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
      ]
  )

  chosen_touchpoint = st.selectbox('Choose the touchpoints', ["No Touchpoint","To FM1","To MM1","To MM2","To MM3"])
  if st.button('Run Modelling'):
    df = pd.read_parquet("clean.pq")
    if chosen_touchpoint == "No Touchpoint":
      df = df[["act_late","origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days"]]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
        ]
    )
    if chosen_touchpoint == "To FM1":
      df = df[["act_late","origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days","fm_duration_hours"]]
      num_cols = ["fm_duration_hours"]
      num_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())])
      preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),("cat", cat_transformer, cat_cols),])
    if chosen_touchpoint == "To MM1":
      df = df[["act_late","origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days","fm_duration_hours","mm1_duration_hours"]]
      num_cols = ["fm_duration_hours","mm1_duration_hours"]
      num_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())])
      preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),("cat", cat_transformer, cat_cols),])
    if chosen_touchpoint == "To MM2":
      df = df[["act_late","origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days","fm_duration_hours","mm1_duration_hours", "mm2_duration_hours"]]
      num_cols = ["fm_duration_hours","mm1_duration_hours", "mm2_duration_hours"]
      num_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())])
      preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),("cat", cat_transformer, cat_cols),])
    if chosen_touchpoint == "To MM3":
      df = df[["act_late","origin_area", "origin_sector", "dest_area","dest_sector","sales_channel","route_type","no_of_hub_movement","sla_days","fm_duration_hours","mm1_duration_hours", "mm2_duration_hours","mm3_duration_hours"]]
      num_cols = ["fm_duration_hours","mm1_duration_hours", "mm2_duration_hours","mm3_duration_hours"]
      num_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())])
      preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),("cat", cat_transformer, cat_cols),])

    full_pp = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())])
    df_train, df_test = train_test_split(df, test_size=0.1)
    X_train = df_train.copy()
    y_train = X_train.pop("act_late")
    X_test = df_test.copy()
    y_test = X_test.pop("act_late")
    # training
    full_pp.fit(X_train, y_train)
    # validation metric
    y_pred = full_pp.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.write(report)

if choice == "Prediction":
  st.title("Upload Your Dataset")
  file = st.file_uploader("Upload Your Dataset")
  if file: 
    df_test = pd.read_pq(file, index_col=None)
    X_test = df_test.copy()
    y_pred = full_pp.predict(X_test)
    print(f"There are {len(y_pred)} late orders in the uploaded data")
