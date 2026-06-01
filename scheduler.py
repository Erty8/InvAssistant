import time
import datetime
import pytz
from market_calendar import is_trading_day, get_current_ny_time
from main import run_portfolio_pipeline

def run_scheduler():
    """
    Main loop that keeps running.
    Checks the New York timezone (EST/EDT) every 30 seconds.
    If the current New York time is 10:00 AM (30 minutes after US market open),
    and today is a trading day, it executes the portfolio pipeline.
    """
    print("=" * 60)
    print("Portfolio Monitoring AI Agent Scheduler Activated")
    print("Target Trigger Time: 10:00 AM Eastern Time (EST/EDT) on US Trading Days")
    print("=" * 60)

    last_run_date = None
    heartbeat_interval = 120  # Print status message every 120 iterations (1 hour)
    iteration_count = 0

    while True:
        try:
            ny_time = get_current_ny_time()
            current_date = ny_time.date()
            current_hour = ny_time.hour
            current_minute = ny_time.minute

            # Print status update occasionally so the user knows it is running
            if iteration_count % heartbeat_interval == 0:
                local_now = datetime.datetime.now()
                print(f"[Heartbeat] Local Time: {local_now.strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"New York Time: {ny_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                # Check if today is a trading day for informational logging
                is_today_trading = is_trading_day(current_date)
                trading_status = "TRADING DAY" if is_today_trading else "NON-TRADING DAY / HOLIDAY / WEEKEND"
                print(f"            Market Status for {current_date}: {trading_status}")
                if last_run_date:
                    print(f"            Last run completed on: {last_run_date}")
                else:
                    print("            No runs executed in this session yet.")
            
            iteration_count += 1

            # Trigger condition: New York time is 10:00 AM
            # (Runs between 10:00:00 and 10:00:59)
            if current_hour == 10 and current_minute == 0:
                if last_run_date != current_date:
                    # Check if today is a trading day
                    if is_trading_day(current_date):
                        print(f"\n[Scheduler Alert] Time is {ny_time.strftime('%H:%M:%S %Z')}. Trading day detected. Launching pipeline...")
                        
                        # Run pipeline (do not force calendar check since scheduler already did it)
                        success = run_portfolio_pipeline(force=False)
                        
                        if success:
                            last_run_date = current_date
                            print(f"[Scheduler Alert] Pipeline execution finished successfully for {current_date}.\n")
                        else:
                            print(f"[Scheduler Error] Pipeline execution failed. Will retry if within target window.\n")
                    else:
                        # Today is a market holiday or weekend, skip and mark as checked
                        print(f"\n[Scheduler Alert] Time is 10:00 AM, but today {current_date} is a weekend/holiday. Run skipped.")
                        last_run_date = current_date

        except Exception as e:
            print(f"[Scheduler Exception] An error occurred in the scheduler loop: {e}")
            print("Retrying in 30 seconds...")

        # Sleep for 30 seconds to prevent CPU overloading and ensure we catch the 10:00 AM window
        time.sleep(30)

if __name__ == "__main__":
    try:
        run_scheduler()
    except KeyboardInterrupt:
        print("\nScheduler stopped by user. Exiting...")
