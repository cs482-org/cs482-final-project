import json
import pickle
import gradio as gr
import pandas as pd
from torch import nn
import torch
import torch.nn.functional as F

def load_pickle(filename):
  with open("resources/" + filename, "rb") as f:
    return pickle.load(f)
    
lr = load_pickle("lr.pickle")
ada = load_pickle("ada.pickle")
xgb = load_pickle("xgb.pickle")
rf = load_pickle("rf.pickle")

one_hot = load_pickle("one_hot.pickle")
scaler = load_pickle("scaler.pickle")

with open("resources/mcc_codes.json", "rb") as f:
  mcc_codes = json.load(f)

class FraudNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(346, 32)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(32, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = x.flatten()
    return x
    
model = FraudNetwork()
model.load_state_dict(torch.load("resources/nn.pth", weights_only=True))

def nn_predict(X):
  model.eval()
  return (F.sigmoid(model(torch.Tensor(X))) > 0.5).numpy()

models = {
  "Logistic Regression": lr.predict,
  "AdaBoost": ada.predict,
  "XGBoost": xgb.predict,
  "Random Forest": rf.predict,
  "Neural Network": nn_predict,
}

def get_input(input):
  data = pd.DataFrame(input, index=[0])

  use_chip_one_hot = pd.get_dummies(data["use_chip"], prefix="use_chip_").reindex(columns=one_hot["use_chip"], fill_value=0)
  mcc_one_hot = pd.get_dummies(data["mcc"], prefix="mcc_").reindex(columns=one_hot["mcc"], fill_value=0)
  gender_one_hot = pd.get_dummies(data["gender"], prefix="gender_").reindex(columns=one_hot["gender"], fill_value=0)
  card_brand_one_hot = pd.get_dummies(data["card_brand"], prefix="card_brand_").reindex(columns=one_hot["card_brand"], fill_value=0)
  card_type_one_hot = pd.get_dummies(data["card_type"], prefix="card_type_").reindex(columns=one_hot["card_type"], fill_value=0)

  merchant_state_one_hot = pd.get_dummies(data["merchant_state"], prefix="merchant_state_").reindex(columns=one_hot["merchant_state"], fill_value=0)

  errors_one_hot = data["errors"].str.get_dummies(sep=",").add_prefix("errors_").reindex(columns=one_hot["errors"], fill_value=0)

  concat = pd.concat([data, use_chip_one_hot, mcc_one_hot, gender_one_hot, card_brand_one_hot, card_type_one_hot, merchant_state_one_hot, errors_one_hot], axis=1)
  return scaler.transform(concat.drop(columns=["use_chip", "mcc", "gender", "card_brand", "card_type", "zip", "merchant_state", "errors"]))

def get_one_hot(value):
  return [x.removeprefix(value + "_").removeprefix("_") for x in one_hot[value]]

def predict_all(amount, current_age, retirement_age, birthdate, gender, latitude, longitude, per_capita_income, yearly_income, total_debt, credit_score, num_credit_cards, card_brand, card_type, use_chip, merchant_state, zip, mcc, errors, expires, has_chip, num_cards_issued, credit_limit, acct_open_date, year_pin_last_changed, card_on_dark_web):
  X = get_input({
    "amount": amount,
    "current_age": current_age,
    "retirement_age": retirement_age,
    "birth_year": birthdate.year,
    "birth_month": birthdate.month,
    "gender": gender,
    "latitude": latitude,
    "longitude": longitude,
    "per_capita_income": per_capita_income,
    "yearly_income": yearly_income,
    "total_debt": total_debt,
    "credit_score": credit_score,
    "num_credit_cards": num_credit_cards,
    "card_brand": card_brand,
    "card_type": card_type,
    "expires": expires,
    "has_chip": has_chip,
    "num_cards_issued": num_cards_issued,
    "credit_limit": credit_limit,
    "acct_open_date": acct_open_date,
    "year_pin_last_changed": year_pin_last_changed,
    "card_on_dark_web": card_on_dark_web,
    "use_chip": use_chip,
    "merchant_state": merchant_state,
    "zip": zip,
    "mcc": mcc,
    "errors": ",".join(errors),
  })

  res = ""

  for (name, predict) in models.items():
    result = "Fraud" if bool(predict(X)) else "Normal"
    res += f"{name}: {result}\n"

  return res

demo = gr.Interface(
  fn=predict_all,
  inputs=[
    "number",
    "number",
    "number",
    gr.DateTime(type="datetime"),
    gr.Dropdown(choices=get_one_hot("gender")),
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    gr.Dropdown(choices=get_one_hot("card_brand")),
    gr.Dropdown(choices=get_one_hot("card_type")),
    gr.Dropdown(choices=get_one_hot("use_chip")),
    gr.Dropdown(choices=get_one_hot("merchant_state")),
    "number",
    gr.Dropdown(choices=[(name, value) for (value, name) in mcc_codes.items()]),
    gr.CheckboxGroup(choices=get_one_hot("errors")),
    "datetime",
    "checkbox",
    "number",
    "number",
    "datetime",
    "number",
    "checkbox",
  ],
  outputs=["text"],
  examples=[
    [14.57, 48, 48, "1971-6-01 00:00:00", "Male", 40.8, -91.12, 18076, 36853, 112139, 834, 5, "Mastercard", "Credit", "Swipe Transaction", "IA", 52722.0, "5311", [], "2024-12-01 00:00:00", True, 1, 9100, "2005-09-01 00:00:00", 2015, False],
    [339.00, 63, 63, "1956-10-01 00:00:00", "Male", 34.72, -92.35, 13047, 26600, 0, 799, 4, "Mastercard", "Debit", "Online Transaction", float("nan"), float("nan"), "3640", [], "2023-04-01 00:00:00", True, 1, 13555, "2007-11-01 00:00:00", 2011, False],
  ]
)

demo.launch(share=True)