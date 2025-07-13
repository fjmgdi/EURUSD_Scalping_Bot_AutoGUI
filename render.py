services:
  - type: web
    name: forex-bot
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python ai_forex_trading_bot.py
    envVars:
      - key: DATABASE_URL
        generateValue: true
      - key: PYTHON_VERSION
        value: 3.9.13