import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import preprocessor, helper

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Chat Analyzer Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1226 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1a1f35 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00d4aa;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Sidebar file uploader & selectbox ── */
[data-testid="stFileUploader"] {
    background: #1e2d4a22;
    border: 1px dashed #00d4aa55;
    border-radius: 12px;
    padding: 8px;
}

/* ── Sidebar button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0095ff) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px #00d4aa44 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px #00d4aa66 !important;
}

/* ── Section titles ── */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #00d4aa33;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(135deg, #151d30, #1a2640);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px #00d4aa22;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.green::before  { background: linear-gradient(90deg, #00d4aa, #00ffc8); }
.kpi-card.blue::before   { background: linear-gradient(90deg, #0095ff, #00d4ff); }
.kpi-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.kpi-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
    margin: 0.3rem 0;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Sentiment badges ── */
.sentiment-bar {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 28px;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.8rem;
    font-weight: 600;
}
.s-pos { background: #00d4aa; display:flex; align-items:center; justify-content:center; color:#0a0e1a; }
.s-neu { background: #64748b; display:flex; align-items:center; justify-content:center; color:#e2e8f0; }
.s-neg { background: #ef4444; display:flex; align-items:center; justify-content:center; color:#fff; }

/* ── Plotly chart cards ── */
.chart-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: #1e2d4a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4aa; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly dark theme defaults ───────────────────────────────────────────────
CHART_BG   = '#111827'
CHART_PAPER= '#111827'
CHART_FONT = '#94a3b8'
GRID_COLOR = '#1e2d4a'
PALETTE    = ['#00d4aa','#0095ff','#7c3aed','#f59e0b','#ef4444','#06b6d4','#84cc16','#f43f5e']

def apply_dark_theme(fig, height=380):
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(family='Inter', color=CHART_FONT, size=12),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
        hovermode='x unified'
    )
    return fig

def section(icon, title):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:3rem;">💬</div>
        <div style="font-size:1.1rem; font-weight:700; color:#00d4aa; letter-spacing:0.05em;">
            WhatsApp Analyzer
        </div>
        <div style="font-size:0.72rem; color:#475569; margin-top:0.2rem; letter-spacing:0.1em; text-transform:uppercase;">
            Professional Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">Upload Chat Export</p>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("", type=["txt"], label_visibility="collapsed")

    st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        st.markdown('<p style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-top:1rem;">Analyze For</p>', unsafe_allow_html=True)
        selected_user = st.selectbox("", user_list, label_visibility="collapsed")

        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
        show_btn = st.button("🚀  Run Analysis")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#334155;">
            <div style="color:#475569; font-weight:600; margin-bottom:0.5rem;">CHAT SNAPSHOT</div>
            📅 {df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}<br>
            👥 {df['user'].nunique() - 1} participants<br>
            💬 {df.shape[0]:,} total messages
        </div>
        """, unsafe_allow_html=True)
    else:
        show_btn = False
        st.markdown("""
        <div style="background:#0f172a; border:1px solid #1e3a5f; border-radius:12px; padding:1rem; font-size:0.8rem; color:#475569; margin-top:1rem;">
            📋 <strong style="color:#64748b">How to export:</strong><br><br>
            WhatsApp → Chat → ⋮ → More → Export Chat → Without Media<br><br>
            Then upload the <code style="color:#00d4aa">.txt</code> file above.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(135deg, #00d4aa18, #0095ff10);
    border: 1px solid #00d4aa22;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
">
    <h1 style="
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #0095ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    ">WhatsApp Chat Analyzer</h1>
    <p style="color:#64748b; font-size:1rem; margin:0;">
        Professional insights from your conversations — timelines, activity patterns, language analysis & more.
    </p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">📂</div>
        <div style="font-size:1.1rem; font-weight:600; color:#475569;">Upload your WhatsApp chat export to begin</div>
        <div style="font-size:0.85rem; color:#334155; margin-top:0.5rem;">Supports group chats and individual conversations</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not show_btn:
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; color:#334155;">
        <div style="font-size:3rem; margin-bottom:0.8rem;">👈</div>
        <div style="font-size:1rem; font-weight:500; color:#475569;">Select a participant and click <strong style="color:#00d4aa">Run Analysis</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  1. KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
section("📊", "Overview")

num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
pos_pct, neu_pct, neg_pct = helper.sentiment_analysis(selected_user, df)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "green",  "💬", f"{num_messages:,}", "Total Messages"),
    (c2, "blue",   "📝", f"{words:,}",        "Total Words"),
    (c3, "purple", "🖼️", f"{num_media:,}",    "Media Shared"),
    (c4, "amber",  "🔗", f"{num_links:,}",    "Links Shared"),
]
for col, color, icon, val, label in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Sentiment bar
st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div>
  <div style="color:#64748b; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">
    Sentiment Overview
  </div>
  <div class="sentiment-bar">
    <div class="s-pos" style="width:{pos_pct}%">{"😊 " + str(pos_pct) + "%" if pos_pct > 8 else ""}</div>
    <div class="s-neu" style="width:{neu_pct}%">{"😐 " + str(neu_pct) + "%" if neu_pct > 8 else ""}</div>
    <div class="s-neg" style="width:{neg_pct}%">{"😠 " + str(neg_pct) + "%" if neg_pct > 8 else ""}</div>
  </div>
  <div style="display:flex; gap:1.5rem; font-size:0.75rem; color:#475569;">
    <span>😊 Positive {pos_pct}%</span>
    <span>😐 Neutral {neu_pct}%</span>
    <span>😠 Negative {neg_pct}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. TIMELINES
# ═══════════════════════════════════════════════════════════════════════════════
section("📅", "Message Timeline")

tab1, tab2 = st.tabs(["📆 Monthly", "📅 Daily"])

with tab1:
    timeline = helper.monthly_timeline(selected_user, df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline['time'], y=timeline['message'],
        mode='lines+markers',
        line=dict(color='#00d4aa', width=2.5),
        marker=dict(size=6, color='#00d4aa', line=dict(color='#0a0e1a', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.08)',
        name='Messages'
    ))
    fig.update_layout(title='Monthly Message Volume')
    fig = apply_dark_theme(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    daily = helper.daily_timeline(selected_user, df)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily['only_date'], y=daily['message'],
        mode='lines',
        line=dict(color='#0095ff', width=1.8),
        fill='tozeroy',
        fillcolor='rgba(0,149,255,0.08)',
        name='Messages'
    ))
    fig2.update_layout(title='Daily Message Volume')
    fig2 = apply_dark_theme(fig2, height=340)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. ACTIVITY PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
section("⏰", "Activity Patterns")

col_a, col_b = st.columns(2)

with col_a:
    day_data = helper.week_activity_map(selected_user, df)
    fig_day = px.bar(
        x=day_data.index, y=day_data.values,
        color=day_data.values,
        color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#0095ff'],[1,'#00d4aa']],
        labels={'x':'Day','y':'Messages','color':''},
        title='Messages by Day of Week'
    )
    fig_day.update_traces(marker_line_width=0)
    fig_day.update_layout(coloraxis_showscale=False)
    fig_day = apply_dark_theme(fig_day, 320)
    st.plotly_chart(fig_day, use_container_width=True)

with col_b:
    month_data = helper.month_activity_map(selected_user, df)
    fig_mon = px.bar(
        x=month_data.index, y=month_data.values,
        color=month_data.values,
        color_continuous_scale=[[0,'#1e1a40'],[0.5,'#7c3aed'],[1,'#a855f7']],
        labels={'x':'Month','y':'Messages','color':''},
        title='Messages by Month'
    )
    fig_mon.update_traces(marker_line_width=0)
    fig_mon.update_layout(coloraxis_showscale=False)
    fig_mon = apply_dark_theme(fig_mon, 320)
    st.plotly_chart(fig_mon, use_container_width=True)

# Heatmap
heatmap_data = helper.activity_heatmap(selected_user, df)
if not heatmap_data.empty:
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale=[[0,'#0a0e1a'],[0.3,'#0d3b6e'],[0.7,'#0095ff'],[1,'#00d4aa']],
        aspect='auto',
        title='Activity Heatmap — Day × Hour Period',
        labels=dict(x='Hour Period', y='Day', color='Messages')
    )
    fig_heat.update_layout(coloraxis_showscale=True)
    fig_heat = apply_dark_theme(fig_heat, 360)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. USER ANALYSIS (group only)
# ═══════════════════════════════════════════════════════════════════════════════
if selected_user == 'Overall':
    section("👥", "User Analysis")

    busy = df['user'].value_counts().head(10).reset_index()
    busy.columns = ['User', 'Messages']
    busy['pct'] = (busy['Messages'] / df.shape[0] * 100).round(2)

    col_u1, col_u2 = st.columns([3, 2])

    with col_u1:
        fig_users = px.bar(
            busy, x='Messages', y='User', orientation='h',
            color='Messages',
            color_continuous_scale=[[0,'#1e3a5f'],[1,'#ef4444']],
            title='Most Active Participants',
            text='Messages'
        )
        fig_users.update_traces(marker_line_width=0, textposition='outside',
                                textfont=dict(color='#94a3b8'))
        fig_users.update_layout(yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        fig_users = apply_dark_theme(fig_users, 380)
        st.plotly_chart(fig_users, use_container_width=True)

    with col_u2:
        fig_pie = px.pie(
            busy, names='User', values='Messages',
            color_discrete_sequence=PALETTE,
            title='Message Share',
            hole=0.55
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent', textfont_size=11)
        fig_pie.update_layout(
            paper_bgcolor=CHART_PAPER,
            font=dict(family='Inter', color=CHART_FONT),
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Response time
    rt_df = helper.avg_response_time(df)
    if not rt_df.empty:
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        fig_rt = px.bar(
            rt_df, x='User', y='Avg Response Time (min)',
            color='Avg Response Time (min)',
            color_continuous_scale=[[0,'#00d4aa'],[0.5,'#f59e0b'],[1,'#ef4444']],
            title='Average Response Time per User (lower = faster)',
            text='Avg Response Time (min)'
        )
        fig_rt.update_traces(marker_line_width=0, textposition='outside',
                             textfont=dict(color='#94a3b8'))
        fig_rt.update_layout(coloraxis_showscale=False)
        fig_rt = apply_dark_theme(fig_rt, 340)
        st.plotly_chart(fig_rt, use_container_width=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. MESSAGE LENGTH
# ═══════════════════════════════════════════════════════════════════════════════
section("📏", "Message Length Distribution")

len_df = helper.message_length_stats(selected_user, df)
if not len_df.empty:
    fig_box = px.box(
        len_df, x='user', y='msg_len',
        color='user',
        color_discrete_sequence=PALETTE,
        title='Words per Message by User',
        labels={'user':'User','msg_len':'Words per Message'}
    )
    fig_box.update_layout(showlegend=False)
    fig_box = apply_dark_theme(fig_box, 360)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. WORD CLOUD
# ═══════════════════════════════════════════════════════════════════════════════
section("☁️", "Word Cloud")

try:
    df_wc = helper.create_wordcloud(selected_user, df)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
    fig_wc.patch.set_facecolor('#111827')
    ax_wc.set_facecolor('#111827')
    ax_wc.imshow(df_wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
except Exception as e:
    st.warning(f"Could not generate word cloud: {e}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. MOST COMMON WORDS
# ═══════════════════════════════════════════════════════════════════════════════
section("🔤", "Most Common Words")

try:
    common_df = helper.most_common_words(selected_user, df)
    col_w1, col_w2 = st.columns([3, 2])

    with col_w1:
        fig_words = px.bar(
            common_df, x='Frequency', y='Word', orientation='h',
            color='Frequency',
            color_continuous_scale=[[0,'#1e3a5f'],[1,'#7c3aed']],
            title='Top 20 Words Used',
            text='Frequency'
        )
        fig_words.update_traces(marker_line_width=0, textposition='outside',
                                textfont=dict(color='#94a3b8'))
        fig_words.update_layout(yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        fig_words = apply_dark_theme(fig_words, 520)
        st.plotly_chart(fig_words, use_container_width=True)

    with col_w2:
        st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
        st.dataframe(
            common_df.style.background_gradient(cmap='Blues', subset=['Frequency']),
            use_container_width=True, height=480
        )
except Exception as e:
    st.warning(f"Could not compute common words: {e}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. EMOJI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("😊", "Emoji Analysis")

try:
    emoji_df = helper.emoji_helper(selected_user, df)
    if emoji_df.empty:
        st.info("No emojis found in this chat.")
    else:
        top_emoji = emoji_df.head(20)
        col_e1, col_e2 = st.columns([2, 3])

        with col_e1:
            st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
            st.dataframe(top_emoji, use_container_width=True, height=400)

        with col_e2:
            fig_em = px.pie(
                top_emoji, names='Emoji', values='Frequency',
                color_discrete_sequence=PALETTE,
                title='Top Emoji Distribution',
                hole=0.4
            )
            fig_em.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
            fig_em.update_layout(
                paper_bgcolor=CHART_PAPER,
                font=dict(family='Inter', color=CHART_FONT),
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_em, use_container_width=True)
except Exception as e:
    st.warning(f"Could not analyze emojis: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  9. EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
section("📥", "Export Data")

export_df = df[df['user'] != 'group_notification'].copy() if selected_user == 'Overall' \
    else df[df['user'] == selected_user].copy()

csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️  Download Filtered Chat Data (CSV)",
    data=csv,
    file_name=f"whatsapp_analysis_{selected_user.replace(' ','_')}.csv",
    mime='text/csv'
)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#1e2d4a; font-size:0.75rem;">
    WhatsApp Chat Analyzer Pro &nbsp;·&nbsp; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
