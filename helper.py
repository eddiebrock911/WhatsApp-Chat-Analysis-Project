import re
import pandas as pd
import numpy as np
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji

extractor = URLExtract()

# ─── Positive / Negative keyword sets for lightweight sentiment ──────────────
POSITIVE_WORDS = {
    'good','great','awesome','amazing','love','happy','excellent','wonderful',
    'fantastic','perfect','nice','beautiful','brilliant','best','superb','thanks',
    'thank','haha','lol','hehe','😂','❤️','😍','👍','🎉','🥳','😊','👏',
    'congrats','congratulations','yay','wow','cool','sweet','fun','enjoy'
}
NEGATIVE_WORDS = {
    'bad','hate','worst','terrible','awful','horrible','sad','angry','boring',
    'ugly','stupid','dumb','annoying','problem','issue','sorry','unfortunately',
    'fail','failed','wrong','mistake','😢','😭','😡','👎','💔','😠','ugh','wtf'
}

# ─── 1. Basic Stats ───────────────────────────────────────────────────────────
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'].str.contains('<Media omitted>', na=False)].shape[0]

    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


# ─── 2. Busy Users ───────────────────────────────────────────────────────────
def fetch_busy_users(df):
    x = df['user'].value_counts().head()
    return x


# ─── 3. WordCloud ────────────────────────────────────────────────────────────
def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words = set(f.read().split())
    f.close()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>', na=False)]

    def clean(message):
        return ' '.join([w for w in message.lower().split() if w not in stop_words])

    text = temp['message'].apply(clean).str.cat(sep=' ')

    wc = WordCloud(
        width=800, height=400,
        min_font_size=10,
        background_color='#0E1117',
        colormap='cool',
        max_words=150
    )
    return wc.generate(text)


# ─── 4. Most Common Words ────────────────────────────────────────────────────
def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words = set(f.read().split())
    f.close()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>', na=False)]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words and word.isalpha():
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Frequency'])
    return most_common_df


# ─── 5. Emoji Analysis ───────────────────────────────────────────────────────
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(
        Counter(emojis).most_common(len(Counter(emojis))),
        columns=['Emoji', 'Frequency']
    )
    return emoji_df


# ─── 6. Monthly Timeline ─────────────────────────────────────────────────────
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline


# ─── 7. Daily Timeline ───────────────────────────────────────────────────────
def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily = df.groupby('only_date').count()['message'].reset_index()
    return daily


# ─── 8. Week Activity Map ────────────────────────────────────────────────────
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['day_name'].value_counts().reindex(day_order, fill_value=0)
    return counts


# ─── 9. Month Activity Map ───────────────────────────────────────────────────
def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    counts = df['month'].value_counts().reindex(month_order, fill_value=0).dropna()
    return counts


# ─── 10. Activity Heatmap ────────────────────────────────────────────────────
def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(
        index='day_name', columns='period',
        values='message', aggfunc='count'
    ).fillna(0)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    user_heatmap = user_heatmap.reindex(
        [d for d in day_order if d in user_heatmap.index]
    )
    return user_heatmap


# ─── 11. Average Response Time ───────────────────────────────────────────────
def avg_response_time(df):
    """Returns a DataFrame with average response time (minutes) per user."""
    data = df[df['user'] != 'group_notification'].copy().sort_values('date')
    data['prev_user'] = data['user'].shift(1)
    data['prev_time'] = data['date'].shift(1)
    data['response_gap'] = (data['date'] - data['prev_time']).dt.total_seconds() / 60

    # Only count gaps where two DIFFERENT users talked, and gap < 60 mins (actual replies)
    replies = data[(data['user'] != data['prev_user']) & (data['response_gap'] < 60)]
    result = replies.groupby('user')['response_gap'].mean().reset_index()
    result.columns = ['User', 'Avg Response Time (min)']
    result['Avg Response Time (min)'] = result['Avg Response Time (min)'].round(2)
    result = result.sort_values('Avg Response Time (min)')
    return result


# ─── 12. Message Length Stats ────────────────────────────────────────────────
def message_length_stats(selected_user, df):
    """Returns per-user message length distribution for box plot."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[~temp['message'].str.contains('<Media omitted>', na=False)]
    temp['msg_len'] = temp['message'].apply(lambda x: len(x.split()))
    return temp[['user', 'msg_len']]


# ─── 13. Sentiment Analysis (Keyword-based) ──────────────────────────────────
def sentiment_analysis(selected_user, df):
    """Returns (positive_pct, neutral_pct, negative_pct) as rounded floats."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    pos = neg = neu = 0

    for message in temp['message']:
        words = set(message.lower().split())
        if words & POSITIVE_WORDS:
            pos += 1
        elif words & NEGATIVE_WORDS:
            neg += 1
        else:
            neu += 1

    total = pos + neg + neu or 1
    return round(pos/total*100, 1), round(neu/total*100, 1), round(neg/total*100, 1)