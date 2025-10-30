import schedule
import time
import os
from datetime import datetime

def run_predict():
    os.system("python src/automation/predict_daily.py")

def run_validate():
    os.system("python src/automation/validate_close.py")

schedule.every().monday.at("12:22").do(run_predict)
schedule.every().tuesday.at("12:22").do(run_predict)
schedule.every().wednesday.at("12:22").do(run_predict)
schedule.every().thursday.at("12:22").do(run_predict)
schedule.every().friday.at("12:22").do(run_predict)

schedule.every().monday.at("16:00").do(run_validate)
schedule.every().tuesday.at("16:00").do(run_validate)
schedule.every().wednesday.at("16:00").do(run_validate)
schedule.every().thursday.at("16:00").do(run_validate)
schedule.every().friday.at("16:00").do(run_validate)

print("⏳ Scheduler started... (Ctrl+C to stop)")
while True:
    schedule.run_pending()
    time.sleep(60)
