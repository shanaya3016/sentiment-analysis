import joblib
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="SentimentAI", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');
* { font-family: 'Nunito', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0118 0%, #0d1533 35%, #0a1628 65%, #120a1e 100%);
}

.main-header { text-align: center; padding: 50px 20px 30px; }

.main-title {
    font-size: 54px;
    font-weight: 800;
    background: linear-gradient(135deg, #f472b6, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}

.main-subtitle { color: rgba(255,255,255,0.4) !important; font-size: 16px; margin: 12px 0 0 0; }

.badge-row { margin-top: 20px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }

.badge-1 {
    background: rgba(244,114,182,0.08);
    border: 1px solid rgba(244,114,182,0.25);
    color: #f472b6 !important;
    padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-2 {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    color: #a78bfa !important;
    padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-3 {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8 !important;
    padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-4 {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    color: #34d399 !important;
    padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #f472b6, #a78bfa, #38bdf8, #34d399, transparent);
    margin: 20px 0 30px 0;
    opacity: 0.4;
}

.glass-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.glass-card-title { font-size: 16px; font-weight: 700; color: rgba(255,255,255,0.75) !important; margin-bottom: 18px; }

.result-positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.07), rgba(56,189,248,0.05));
    border: 1px solid rgba(52,211,153,0.25);
    border-left: 4px solid #34d399;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 20px 0;
}

.result-positive-title {
    font-size: 26px; font-weight: 800;
    background: linear-gradient(135deg, #34d399, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.result-negative {
    background: linear-gradient(135deg, rgba(244,114,182,0.07), rgba(167,139,250,0.05));
    border: 1px solid rgba(244,114,182,0.25);
    border-left: 4px solid #f472b6;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 20px 0;
}

.result-negative-title {
    font-size: 26px; font-weight: 800;
    background: linear-gradient(135deg, #f472b6, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.conf-bar-container {
    margin-top: 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    height: 6px;
    overflow: hidden;
}

.conf-bar-green {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #34d399, #38bdf8);
}

.conf-bar-pink {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #f472b6, #a78bfa);
}

.conf-label { color: rgba(255,255,255,0.3) !important; font-size: 13px; margin-top: 6px; }

.comment-display {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
}

.comment-label { color: rgba(255,255,255,0.25) !important; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }
.comment-body { color: rgba(255,255,255,0.7) !important; font-size: 15px; margin-top: 8px; line-height: 1.6; }

.stat-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
}

.stat-number-rainbow {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(135deg, #f472b6, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.stat-number-green {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(135deg, #34d399, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.stat-number-pink {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(135deg, #f472b6, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.stat-label { color: rgba(255,255,255,0.3) !important; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-top: 4px; }

.overall-pos {
    background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(56,189,248,0.05));
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 16px; padding: 22px; text-align: center;
    font-size: 20px; font-weight: 800;
    background-clip: text;
    color: #34d399 !important;
}

.overall-neg {
    background: linear-gradient(135deg, rgba(244,114,182,0.08), rgba(167,139,250,0.05));
    border: 1px solid rgba(244,114,182,0.2);
    border-radius: 16px; padding: 22px; text-align: center;
    font-size: 20px; font-weight: 800;
    color: #f472b6 !important;
}

.comment-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
}

.pos-badge {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399 !important;
    padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 700;
}

.neg-badge {
    background: rgba(244,114,182,0.08);
    border: 1px solid rgba(244,114,182,0.3);
    color: #f472b6 !important;
    padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 700;
}

.item-comment { color: rgba(255,255,255,0.6) !important; font-size: 14px; margin-top: 8px; line-height: 1.5; }

.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.8) !important;
    font-size: 15px !important;
}

        
.stTextArea > div > div {
    background: #0d1117 !important;
    border-radius: 14px !important;
}

div[data-baseweb="textarea"] {
    background: #0d1117 !important;
}

div[data-baseweb="textarea"] > div {
    background: #0d1117 !important;
    color: rgba(255,255,255,0.85) !important;
}

.stButton button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #8b5cf6, #6366f1, #38bdf8) !important;
    box-shadow: 0 6px 30px rgba(124,58,237,0.4) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 5px !important;
}

.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.35) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}

h1,h2,h3,p,label,div { color: rgba(255,255,255,0.85) !important; }
.stSuccess > div { background: rgba(52,211,153,0.06) !important; border: 1px solid rgba(52,211,153,0.2) !important; color: #34d399 !important; border-radius: 12px !important; }
.stWarning > div { background: rgba(251,191,36,0.06) !important; border: 1px solid rgba(251,191,36,0.2) !important; color: #fbbf24 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="main-title">🔮 SentimentAI</h1>
    <p class="main-subtitle">Decode the emotion behind every word — powered by artificial intelligence</p>
    <div class="badge-row">
        <span class="badge-1">🤖 AI Powered</span>
        <span class="badge-2">💫 1.6M Tweets</span>
        <span class="badge-3">⚡ Real-time</span>
        <span class="badge-4">✨ 77% Accuracy</span>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

@st.cache_resource
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    def clean_text(text):
        text = str(text).encode('ascii', 'ignore').decode('ascii')
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    return model, vectorizer, clean_text
with st.spinner("🔮 Loading AI Model..."):
    model, vectorizer, clean_text = load_model()

st.success("✨ SentimentAI Ready — Start Analyzing!")
st.markdown("<br>", unsafe_allow_html=True)

pos_emojis = ['❤','😍','😊','🥰','👍','🔥','💯','😁','🎉','✨','💕','😄','🙏','👏','❤️','😂','🤣']
neg_emojis = ['😡','👎','😢','😭','🤮','💔','😤','🤬','😠','😒','🙄','😞','😔','😣']

tab1, tab2 = st.tabs(["🔮  Single Comment", "⚡  Bulk Analysis"])

with tab1:
    st.markdown('<div class="glass-card"><div class="glass-card-title">🔮 Analyze a Single Comment</div>', unsafe_allow_html=True)
    user_input = st.text_area("", height=150, placeholder="Paste any comment from YouTube, Instagram, Facebook, Twitter...", key="s1")
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        btn1 = st.button("✨ Analyze Now", key="b1")
    st.markdown('</div>', unsafe_allow_html=True)

    if btn1:
        if user_input.strip():
            has_pos = any(e in user_input for e in pos_emojis)
            has_neg = any(e in user_input for e in neg_emojis)
            cleaned = clean_text(user_input)

            if not cleaned.strip():
                is_pos = has_pos
                pct = 95.0
            else:
                vec = vectorizer.transform([cleaned])
                res = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0]
                if res == 1:
                    is_pos = not (has_neg and not has_pos)
                    pct = round(prob[1]*100, 2)
                else:
                    is_pos = has_pos and not has_neg
                    pct = round(prob[0]*100, 2)

            bar_width = int(pct)

            if is_pos:
                st.markdown(f'''
                <div class="result-positive">
                    <div class="result-positive-title">✅ POSITIVE SENTIMENT</div>
                    <div class="conf-bar-container">
                        <div class="conf-bar-green" style="width:{bar_width}%"></div>
                    </div>
                    <div class="conf-label">Confidence Score: {pct}%</div>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="result-negative">
                    <div class="result-negative-title">❌ NEGATIVE SENTIMENT</div>
                    <div class="conf-bar-container">
                        <div class="conf-bar-pink" style="width:{bar_width}%"></div>
                    </div>
                    <div class="conf-label">Confidence Score: {pct}%</div>
                </div>''', unsafe_allow_html=True)

            st.markdown(f'''
            <div class="comment-display">
                <div class="comment-label">ANALYZED COMMENT</div>
                <div class="comment-body">{user_input}</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.warning("Please enter a comment to analyze.")

with tab2:
    st.markdown('<div class="glass-card"><div class="glass-card-title">⚡ Bulk Comment Analysis</div><p style="color:rgba(255,255,255,0.3);font-size:14px;margin-bottom:18px">Paste comments — each on a new line</p>', unsafe_allow_html=True)
    multi_input = st.text_area("", height=220, placeholder="Comment 1\nComment 2\nComment 3\n...", key="m1")
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        btn2 = st.button("🚀 Analyze All", key="b2")
    st.markdown('</div>', unsafe_allow_html=True)

    if btn2:
        if multi_input.strip():
            comments = [c.strip() for c in multi_input.strip().split('\n') if c.strip()]
            pos_count = 0
            neg_count = 0
            results_html = ""
            all_results = []

            for i, comment in enumerate(comments, 1):
                has_pos = any(e in comment for e in pos_emojis)
                has_neg = any(e in comment for e in neg_emojis)
                cleaned = clean_text(comment)

                if not cleaned.strip():
                    is_pos = has_pos
                    conf = 95.0
                else:
                    vec = vectorizer.transform([cleaned])
                    res = model.predict(vec)[0]
                    prob = model.predict_proba(vec)[0]
                    if res == 1:
                        is_pos = not (has_neg and not has_pos)
                        conf = round(prob[1]*100, 1)
                    else:
                        is_pos = has_pos and not has_neg
                        conf = round(prob[0]*100, 1)

                all_results.append({
                    "comment": comment,
                    "is_pos": is_pos,
                    "conf": conf
                })

                if is_pos:
                    pos_count += 1
                    badge = '<span class="pos-badge">✅ POSITIVE</span>'
                else:
                    neg_count += 1
                    badge = '<span class="neg-badge">❌ NEGATIVE</span>'

                results_html += f'''
                <div class="comment-item">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:rgba(255,255,255,0.2);font-size:11px;font-weight:700">#{i}</span>
                        <div>{badge}<span style="color:rgba(255,255,255,0.2);font-size:11px;margin-left:10px">{conf}%</span></div>
                    </div>
                    <div class="item-comment">{comment}</div>
                </div>'''

            total = len(comments)

            # Most Positive & Most Negative
            pos_results = [r for r in all_results if r["is_pos"]]
            neg_results = [r for r in all_results if not r["is_pos"]]

            most_pos = max(pos_results, key=lambda x: x["conf"]) if pos_results else None
            most_neg = max(neg_results, key=lambda x: x["conf"]) if neg_results else None

            def get_reason(comment, is_pos):
                positive_words = ["love", "amazing", "great", "best", "excellent", "wonderful",
                                "fantastic", "happy", "beautiful", "awesome", "perfect",
                                "brilliant", "incredible", "outstanding", "superb", "thank",
                                "helpful", "motivated", "smooth", "improved", "smile"]
                negative_words = ["hate", "terrible", "worst", "bad", "boring", "awful",
                                "horrible", "disappointing", "waste", "useless", "poor",
                                "wrong", "misleading", "regret", "low", "unsubscribed",
                                "confusing", "reported", "disappointing", "slow"]

                words = comment.lower().split()
                if is_pos:
                    found = [w for w in words if w in positive_words]
                    if found:
                        return f"Strong positive words detected: <b style='color:#34d399'>{', '.join(found[:3])}</b>"
                    return "Overall positive tone detected by AI model"
                else:
                    found = [w for w in words if w in negative_words]
                    if found:
                        return f"Strong negative words detected: <b style='color:#f472b6'>{', '.join(found[:3])}</b>"
                    return "Overall negative tone detected by AI model"

            # Stats
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number-rainbow">{total}</div><div class="stat-label">TOTAL COMMENTS</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number-green">{pos_count}</div><div class="stat-label">POSITIVE ({round(pos_count/total*100,1)}%)</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number-pink">{neg_count}</div><div class="stat-label">NEGATIVE ({round(neg_count/total*100,1)}%)</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Overall Mood
            if pos_count > neg_count:
                st.markdown('<div class="overall-pos">✨ Overall Sentiment: POSITIVE AUDIENCE!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="overall-neg">💫 Overall Sentiment: NEGATIVE AUDIENCE!</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Most Positive & Most Negative Cards
            col1, col2 = st.columns(2)

            with col1:
                if most_pos:
                    reason = get_reason(most_pos["comment"], True)
                    st.markdown(f'''
                    <div class="result-positive">
                        <div style="color:rgba(255,255,255,0.35);font-size:11px;font-weight:700;letter-spacing:1px">🏆 MOST POSITIVE COMMENT</div>
                        <div style="color:#34d399;font-size:22px;font-weight:800;margin:6px 0">Confidence: {most_pos["conf"]}%</div>
                        <div style="color:rgba(255,255,255,0.7);font-size:14px;margin:10px 0;line-height:1.5">"{most_pos["comment"]}"</div>
                        <div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);border-radius:8px;padding:10px;margin-top:10px">
                            <div style="color:rgba(255,255,255,0.35);font-size:11px;font-weight:700;letter-spacing:1px">💡 REASON</div>
                            <div style="color:rgba(255,255,255,0.65);font-size:13px;margin-top:4px">{reason}</div>
                        </div>
                    </div>''', unsafe_allow_html=True)

            with col2:
                if most_neg:
                    reason = get_reason(most_neg["comment"], False)
                    st.markdown(f'''
                    <div class="result-negative">
                        <div style="color:rgba(255,255,255,0.35);font-size:11px;font-weight:700;letter-spacing:1px">💔 MOST NEGATIVE COMMENT</div>
                        <div style="color:#f472b6;font-size:22px;font-weight:800;margin:6px 0">Confidence: {most_neg["conf"]}%</div>
                        <div style="color:rgba(255,255,255,0.7);font-size:14px;margin:10px 0;line-height:1.5">"{most_neg["comment"]}"</div>
                        <div style="background:rgba(244,114,182,0.08);border:1px solid rgba(244,114,182,0.2);border-radius:8px;padding:10px;margin-top:10px">
                            <div style="color:rgba(255,255,255,0.35);font-size:11px;font-weight:700;letter-spacing:1px">💡 REASON</div>
                            <div style="color:rgba(255,255,255,0.65);font-size:13px;margin-top:4px">{reason}</div>
                        </div>
                    </div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Pie Chart + Results
            col1, col2 = st.columns([1,1])
            with col1:
                fig, ax = plt.subplots(figsize=(5,5), facecolor='none')
                wedges, texts, autotexts = ax.pie(
                    [pos_count, neg_count],
                    labels=['Positive', 'Negative'],
                    colors=['#34d399', '#f472b6'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.6, edgecolor='none')
                )
                for text in texts:
                    text.set_color('white')
                    text.set_fontsize(13)
                    text.set_fontweight('bold')
                for at in autotexts:
                    at.set_color('white')
                    at.set_fontsize(12)
                    at.set_fontweight('bold')
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                st.pyplot(fig)

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="glass-card-title">📝 Detailed Results</div>', unsafe_allow_html=True)
                st.markdown(results_html, unsafe_allow_html=True)
        else:
            st.warning("Please paste some comments first.")