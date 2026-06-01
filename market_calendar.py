import datetime
import pytz

def get_easter_date(year):
    """
    Computes Easter Sunday using the Meeus/Jones/Butcher algorithm.
    Returns a datetime.date object.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)

def get_good_friday(year):
    """Good Friday is the Friday before Easter Sunday."""
    easter = get_easter_date(year)
    return easter - datetime.timedelta(days=2)

def get_nyse_holidays(year):
    """
    Returns a set of datetime.date objects representing NYSE holidays for a given year.
    Implements standard NYSE holiday rules, including observations on adjacent weekdays.
    """
    holidays = set()

    # Auxiliary: add a holiday, observing it on Friday if it falls on Saturday,
    # or on Monday if it falls on Sunday.
    def add_observed_holiday(date_obj):
        if date_obj.weekday() == 5:  # Saturday
            holidays.add(date_obj - datetime.timedelta(days=1))
        elif date_obj.weekday() == 6:  # Sunday
            holidays.add(date_obj + datetime.timedelta(days=1))
        else:
            holidays.add(date_obj)

    # 1. New Year's Day (Jan 1)
    # Note: If Jan 1 is a Saturday, the preceding Friday (Dec 31) is observed.
    # If Dec 31 of the previous year is observed, we check that separately or add it.
    new_years_day = datetime.date(year, 1, 1)
    if new_years_day.weekday() == 5:
        # If Saturday, the observed day is Friday (Dec 31) of the PREVIOUS year.
        # So for the current year, we don't observe New Year's Day (it was observed Dec 31 of previous year).
        # However, Dec 31 of the CURRENT year will be observed if Jan 1 of NEXT year is Saturday.
        pass
    elif new_years_day.weekday() == 6:
        # If Sunday, observed on Monday Jan 2
        holidays.add(datetime.date(year, 1, 2))
    else:
        holidays.add(new_years_day)

    # Check if Dec 31 of current year is the observed holiday for Jan 1 of next year
    next_new_years_day = datetime.date(year + 1, 1, 1)
    if next_new_years_day.weekday() == 5:
        holidays.add(datetime.date(year, 1, 31))

    # 2. Martin Luther King Jr. Day (Third Monday of January)
    # January 1st to 7th are the first week. Third Monday falls between Jan 15 and 21.
    mlk = datetime.date(year, 1, 1)
    # Find first Monday
    days_to_monday = (0 - mlk.weekday() + 7) % 7
    first_monday = mlk + datetime.timedelta(days=days_to_monday)
    third_monday = first_monday + datetime.timedelta(weeks=2)
    holidays.add(third_monday)

    # 3. Washington's Birthday / Presidents' Day (Third Monday of February)
    presidents_day = datetime.date(year, 2, 1)
    days_to_monday = (0 - presidents_day.weekday() + 7) % 7
    first_monday = presidents_day + datetime.timedelta(days=days_to_monday)
    third_monday = first_monday + datetime.timedelta(weeks=2)
    holidays.add(third_monday)

    # 4. Good Friday
    holidays.add(get_good_friday(year))

    # 5. Memorial Day (Last Monday of May)
    # Start at May 31st and subtract days until we hit a Monday
    may_31 = datetime.date(year, 5, 31)
    days_to_subtract = (may_31.weekday() - 0 + 7) % 7
    last_monday = may_31 - datetime.timedelta(days=days_to_subtract)
    holidays.add(last_monday)

    # 6. Juneteenth National Independence Day (June 19)
    add_observed_holiday(datetime.date(year, 6, 19))

    # 7. Independence Day (July 4)
    add_observed_holiday(datetime.date(year, 7, 4))

    # 8. Labor Day (First Monday of September)
    sept_1 = datetime.date(year, 9, 1)
    days_to_monday = (0 - sept_1.weekday() + 7) % 7
    first_monday = sept_1 + datetime.timedelta(days=days_to_monday)
    holidays.add(first_monday)

    # 9. Thanksgiving Day (Fourth Thursday of November)
    nov_1 = datetime.date(year, 11, 1)
    days_to_thursday = (3 - nov_1.weekday() + 7) % 7
    first_thursday = nov_1 + datetime.timedelta(days=days_to_thursday)
    fourth_thursday = first_thursday + datetime.timedelta(weeks=3)
    holidays.add(fourth_thursday)

    # 10. Christmas Day (Dec 25)
    add_observed_holiday(datetime.date(year, 12, 25))

    return holidays

def is_trading_day(dt: datetime.date) -> bool:
    """
    Checks if a given date is a valid trading day for the US Stock Market.
    Excludes weekends and NYSE holidays.
    """
    # Check weekend (5 = Saturday, 6 = Sunday)
    if dt.weekday() >= 5:
        return False
    
    # Check holiday list
    nyse_holidays = get_nyse_holidays(dt.year)
    if dt in nyse_holidays:
        return False
        
    return True

def get_current_ny_time() -> datetime.datetime:
    """Returns the current date and time in New York (EST/EDT) timezone."""
    ny_tz = pytz.timezone("America/New York")
    return datetime.datetime.now(ny_tz)
