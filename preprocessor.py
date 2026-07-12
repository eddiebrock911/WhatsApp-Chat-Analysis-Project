import re
import pandas as pd


def preprocess(data):
    # Support both 12-hour (am/pm) and 24-hour WhatsApp export formats
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM|am|pm))?\s?[\u202f\s]-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'date': dates})

    # Normalize and parse date strings
    df['date'] = (
        df['date']
        .str.replace('\u202f', ' ', regex=False)
        .str.strip()
        .str.rstrip('-')
        .str.strip()
    )

    # Try multiple datetime formats
    for fmt in ['%d/%m/%Y, %I:%M %p', '%d/%m/%Y, %H:%M', '%m/%d/%Y, %I:%M %p',
                '%m/%d/%Y, %H:%M', '%d/%m/%y, %I:%M %p', '%d/%m/%y, %H:%M']:
        try:
            df['date'] = pd.to_datetime(df['date'], format=fmt)
            break
        except Exception:
            continue

    if df['date'].dtype == object:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')

    # Separate users and messages — supports multi-word usernames
    users = []
    messages = []

    for message in df['user_message']:
        # Match "Name Name Name: message" — greedy match up to first colon+space
        entry = re.split(r'^(.+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1].strip())
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Time features
    df['year']      = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month']     = df['date'].dt.month_name()
    df['only_date'] = df['date'].dt.date
    df['day']       = df['date'].dt.day
    df['day_name']  = df['date'].dt.day_name()
    df['hour']      = df['date'].dt.hour
    df['minute']    = df['date'].dt.minute

    # Hour period label
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append('23-00')
        elif hour == 0:
            period.append('00-01')
        else:
            period.append(f'{hour:02d}-{hour+1:02d}')

    df['period'] = period
    return df