month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
days_addition = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def is_leap(year):
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    return False


class Time:
    def __init__(self, **kw):
        if 'str' in kw:
            self.str = kw['str']
        else:
            self.str = '{:4d}{:2d}{:2d}'.format(kw['year'], kw['month'], kw['day']) + '{:.3f}'.format(kw['time'])[1:]

    def year(self):
        return int(self.str[0:4])

    def month(self):
        return int(self.str[4:6])

    def day(self):
        return int(self.str[6:8])

    def daytime(self):
        return float('0' + self.str[8:])

    def date(self):
        time = self.daytime()
        hour = int(24 * time)
        minute = int((24 * time - hour) * 60)
        return '{:4d}/{:02d}/{:02d}T{:02d}:{:02d}:00'.format(self.year(), self.month(), self.day(), hour, minute)

    def next_day(self):
        year = self.year()
        month = self.month()
        day = self.day()
        time = self.str[8:]
        if is_leap(year):
            month_days[1] = 29
        else:
            month_days[1] = 28
        if day < month_days[month - 1]:
            day += 1
        else:
            day = 1
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1
        self.str = '{:4d}{:2d}{:2d}'.format(year, month, day) + time

    def year_day(self, reverse=False):
        month = self.month()
        day = days_addition[month - 1] + self.day()
        year = self.year()
        if is_leap(year) and month > 2:
            day += 1
        if reverse:
            if is_leap(year):
                day = 366 - day
            else:
                day = 365 - day
        return day

    def __sub__(self, other):
        year1 = self.year()
        year2 = other.year()
        if year1 == year2:
            return self.year_day() - other.year_day()
        sum_days = 0
        if year1 > year2:
            for year in range(year2 + 1, year1):
                if is_leap(year):
                    sum_days += 366
                else:
                    sum_days += 365
            sum_days += self.year_day() + other.year_day(reverse=True)
        else:
            for year in range(year1 + 1, year2):
                if is_leap(year):
                    sum_days += 366
                else:
                    sum_days += 365
            sum_days += self.year_day(reverse=True) + other.year_day()
            sum_days = -sum_days
        return sum_days

    def __str__(self):
        return self.str