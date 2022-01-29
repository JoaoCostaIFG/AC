#!/usr/bin/env python3

import pandas as pd

account_df = pd.read_csv("data/account.csv")
card_df = pd.read_csv("data/card_train.csv")
client_df = pd.read_csv("data/client.csv")
disp_df = pd.read_csv("data/disp.csv")
district_df = pd.read_csv("data/district.csv")
loan_df = pd.read_csv("data/loan_train.csv")
trans_df = pd.read_csv("data/trans_train.csv")

account_df.join(loan_df, on='account_id')
