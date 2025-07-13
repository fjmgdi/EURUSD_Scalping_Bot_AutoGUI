from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountDetails

api_key = "dc915cac61c5b1ce9b0ab20dba1e58a8-a7e87e72cd84fa4fe49f2646e121b6d6"
account_id = "101-002-28367236-001"
api = API(access_token=api_key, environment="practice")

# Test account details
endpoint = AccountDetails(accountID=account_id)
response = api.request(endpoint)
print(response)  # Should show account details