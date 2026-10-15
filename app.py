"""Titanic Survival Predictor: a Streamlit front end for the XGBoost model."""

import joblib
import pandas as pd
import streamlit as st

from train_model import FEATURES, PORT_MAP, SEX_MAP

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="auto",  # collapses by itself on a phone
)

PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}

# Reference passenger used to explain which inputs move the estimate.
REFERENCE = {"Pclass": 3, "Sex": 1, "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 14.45, "Embarked": 2}

PRESETS = {
    "Custom": None,
    "First class woman, 38, with husband": (1, "female", 38, 1, 0, 71.28, "C", 1),
    "Third class man, 25, travelling alone": (3, "male", 25, 0, 0, 7.90, "S", 0),
    "Young man in first class, large family": (1, "male", 19, 3, 2, 263.00, "S", 0),
    "Boy of 4 in first class, with parents": (1, "male", 4, 0, 2, 81.86, "S", 1),
    "Boy of 4 in third class, large family": (3, "male", 4, 4, 2, 31.28, "S", 0),
    "Woman in third class, alone": (3, "female", 22, 0, 0, 7.75, "Q", 1),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
  --ink: #e9eff5;
  --muted: #90a4b8;
  --rule: #24384a;
  --panel: #122233;
  --brass: #c89b3c;
  --live: #4aa47a;
  --lost: #c05c54;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

.block-container { padding-top: 3.6rem; max-width: 1150px; }

.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--brass); margin-bottom: .35rem;
}
.masthead h1 {
  font-family: 'Spectral', Georgia, serif; font-weight: 600; font-size: 2.5rem;
  line-height: 1.1; margin: 0 0 .5rem; color: var(--ink);
}
.masthead p { color: var(--muted); margin: 0; font-size: .95rem; max-width: 56ch; }
.rule { height: 1px; background: var(--rule); margin: 1.4rem 0 1.6rem; }

.verdict {
  background: var(--panel); border: 1px solid var(--rule); border-left: 4px solid var(--brass);
  border-radius: 4px; padding: 1.5rem 1.7rem;
}
.verdict.live { border-left-color: var(--live); }
.verdict.lost { border-left-color: var(--lost); }
.verdict .label {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted);
}
.verdict .pct {
  font-family: 'Spectral', Georgia, serif; font-size: 3.6rem; font-weight: 600;
  line-height: 1; margin: .35rem 0 .15rem; font-variant-numeric: tabular-nums;
}
.verdict.live .pct { color: var(--live); }
.verdict.lost .pct { color: var(--lost); }
.verdict .says { color: var(--ink); font-size: 1rem; font-weight: 500; }

.gauge {
  position: relative; height: 10px; border-radius: 10px; margin: 1.1rem 0 .5rem;
  background: #1b2d3f; overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 10px; }
.gauge-fill.live { background: linear-gradient(90deg, #2f7d5e, #4aa47a); }
.gauge-fill.lost { background: linear-gradient(90deg, #8e4039, #c05c54); }
.gauge-scale {
  display: flex; justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem; color: var(--muted);
}

.tile {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: .95rem 1.1rem; height: 100%;
}
.tile .k {
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem; letter-spacing: .13em;
  text-transform: uppercase; color: var(--muted); display: block; margin-bottom: .3rem;
}
.tile .v {
  font-family: 'Spectral', Georgia, serif; font-size: 1.65rem; font-weight: 600;
  color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1.1;
}
.tile .s { font-size: .76rem; color: var(--muted); display: block; margin-top: .25rem; }

h3.sec {
  font-family: 'Spectral', Georgia, serif; font-size: 1.15rem; font-weight: 600;
  color: var(--ink); margin: 2rem 0 .3rem;
}
p.sec { color: var(--muted); font-size: .85rem; margin: 0 0 .9rem; }

.factor {
  display: flex; align-items: center; gap: .8rem; padding: .6rem 0;
  border-bottom: 1px solid var(--rule);
}
.factor:last-child { border-bottom: none; }
.factor .name { flex: 1; font-size: .9rem; color: var(--ink); min-width: 0; }
.factor .bar { flex: 0 0 150px; height: 6px; background: #1b2d3f; border-radius: 6px; position: relative; }
.factor .bar::after {
  content: ''; position: absolute; left: 50%; top: -3px; width: 1px; height: 12px;
  background: var(--rule);
}
.factor .bar i { position: absolute; top: 0; height: 100%; border-radius: 6px; display: block; }
.factor .bar i.up { background: var(--live); left: 50%; }
.factor .bar i.dn { background: var(--lost); right: 50%; }
.factor .amt {
  flex: 0 0 62px; text-align: right; font-family: 'IBM Plex Mono', monospace;
  font-size: .8rem; font-variant-numeric: tabular-nums;
}
.factor .amt.up { color: var(--live); }
.factor .amt.dn { color: var(--lost); }

.caveat {
  border-left: 3px solid var(--brass); background: rgba(200,155,60,.07);
  padding: .8rem 1rem; border-radius: 0 3px 3px 0; font-size: .84rem; color: var(--muted);
}
.caveat b { color: var(--ink); font-weight: 600; }
footer, #MainMenu { visibility: hidden; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_data():
    try:
        return pd.read_csv("train.csv")
    except FileNotFoundError:
        return None


model = load_model()
data = load_data()


def predict(pclass, sex_code, age, sibsp, parch, fare, port_code):
    row = pd.DataFrame(
        [[pclass, sex_code, age, sibsp, parch, fare, port_code]], columns=FEATURES
    )
    return float(model.predict_proba(row)[0][1])


def cohort(df, pclass, sex, age, family):
    """Passengers in the data who resemble these inputs."""
    band = df[
        (df["Pclass"] == pclass)
        & (df["Sex"] == sex)
        & (df["Age"].between(age - 10, age + 10))
        & ((df["SibSp"] + df["Parch"] > 0) == (family > 0))
    ]
    return len(band), (band["Survived"].mean() if len(band) else None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="eyebrow">Passenger</div>', unsafe_allow_html=True)

    choice = st.selectbox("Start from", list(PRESETS), help="Load a real passenger, or build your own.")
    p = PRESETS[choice]
    d_pclass, d_sex, d_age, d_sib, d_par, d_fare, d_port = (
        p[:7] if p else (3, "female", 30, 0, 0, 15.0, "S")
    )

    pclass = st.selectbox(
        "Ticket class", [1, 2, 3], index=[1, 2, 3].index(d_pclass),
        format_func=lambda c: {1: "1st class", 2: "2nd class", 3: "3rd class"}[c],
    )
    sex = st.radio("Sex", ["female", "male"], index=["female", "male"].index(d_sex), horizontal=True)
    age = st.slider("Age", 0, 80, int(d_age))

    st.markdown('<div class="eyebrow" style="margin-top:.8rem">Travelling with</div>', unsafe_allow_html=True)
    sibsp = st.number_input("Siblings or spouse aboard", 0, 8, int(d_sib))
    parch = st.number_input("Parents or children aboard", 0, 6, int(d_par))

    st.markdown('<div class="eyebrow" style="margin-top:.8rem">Ticket</div>', unsafe_allow_html=True)
    fare = st.number_input("Fare paid (£)", 0.0, 520.0, float(d_fare), step=1.0,
                           help="A 3rd class ticket was usually under £15. First class could exceed £100.")
    port = st.selectbox("Port of embarkation", list(PORTS), index=list(PORTS).index(d_port),
                        format_func=lambda k: PORTS[k])

prob = predict(pclass, SEX_MAP[sex], age, sibsp, parch, fare, PORT_MAP[port])
lives = prob >= 0.5
tone = "live" if lives else "lost"

# ---------------------------------------------------------------- masthead
st.markdown(
    '<div class="masthead"><div class="eyebrow">RMS Titanic, April 1912</div>'
    "<h1>Would this passenger have survived?</h1>"
    "<p>A gradient boosted model trained on the 891 passengers whose details and outcomes "
    "are recorded in the Kaggle Titanic dataset. Change the passenger details and the estimate "
    "updates straight away.</p></div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown(
        f'<div class="verdict {tone}">'
        f'<span class="label">Model estimate</span>'
        f'<div class="pct">{prob:.0%}</div>'
        f'<div class="says">{"Likely to survive" if lives else "Unlikely to survive"}</div>'
        f'<div class="gauge"><div class="gauge-fill {tone}" style="width:{max(prob, 0.012) * 100:.1f}%"></div></div>'
        f'<div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

with right:
    n, rate = (cohort(data, pclass, sex, age, sibsp + parch) if data is not None else (None, None))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="tile"><span class="k">Similar passengers</span>'
            f'<span class="v">{n if n is not None else "n/a"}</span>'
            '<span class="s">in the training data</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="tile"><span class="k">Of those, survived</span>'
            f'<span class="v">{f"{rate:.0%}" if rate is not None else "n/a"}</span>'
            '<span class="s">what actually happened</span></div>',
            unsafe_allow_html=True,
        )

    if n is not None:
        if n == 0:
            msg = ("<b>No close match in the data.</b> No recorded passenger resembles this "
                   "combination, so the estimate is extrapolated. Treat it as a rough guide.")
        elif n < 12:
            msg = (f"<b>Thin evidence.</b> Only {n} recorded passengers resemble this one, "
                   "so the estimate rests on a small sample.")
        else:
            msg = (f"<b>Well covered.</b> {n} recorded passengers resemble this one, so the "
                   "estimate sits on a reasonable base of evidence.")
        st.markdown(f'<div class="caveat" style="margin-top:.9rem">{msg}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- factors
st.markdown('<h3 class="sec">What moves this estimate</h3>', unsafe_allow_html=True)
st.markdown(
    '<p class="sec">Each line asks one question: if this passenger kept everything else but had '
    "that one detail changed to a plain reference value, a man of 28 in third class travelling alone "
    "on a £14 ticket from Southampton, how far would the estimate move? The figures will not add up "
    "to the total, because the model weighs these details together rather than separately.</p>",
    unsafe_allow_html=True,
)

current = {"Pclass": pclass, "Sex": SEX_MAP[sex], "Age": age, "SibSp": sibsp,
           "Parch": parch, "Fare": fare, "Embarked": PORT_MAP[port]}
labels = {
    "Sex": f"Being {sex}",
    "Pclass": {1: "Travelling first class", 2: "Travelling second class", 3: "Travelling third class"}[pclass],
    "Age": f"Aged {age}",
    "Fare": f"A £{fare:,.0f} ticket",
    "SibSp": f"{sibsp} sibling or spouse aboard" if sibsp == 1 else f"{sibsp} siblings or spouses aboard",
    "Parch": f"{parch} parent or child aboard" if parch == 1 else f"{parch} parents or children aboard",
    "Embarked": f"Boarding at {PORTS[port]}",
}

# Age and fare are continuous, so a near-identical value is not a real difference.
# Comparing a £15 ticket with the £14 reference would surface model noise as a finding.
TOLERANCE = {"Age": 6, "Fare": 7}

deltas = []
for feat in FEATURES:
    gap = abs(current[feat] - REFERENCE[feat])
    if gap <= TOLERANCE.get(feat, 0):
        continue
    swapped = dict(current)
    swapped[feat] = REFERENCE[feat]
    without = float(model.predict_proba(pd.DataFrame([[swapped[f] for f in FEATURES]], columns=FEATURES))[0][1])
    deltas.append((labels[feat], prob - without))

# A row that rounds to 0% tells the reader nothing.
deltas = [d for d in deltas if abs(d[1]) >= 0.005]
deltas.sort(key=lambda x: abs(x[1]), reverse=True)

if not deltas:
    st.markdown('<p class="sec">This is the reference passenger, so there is nothing to compare.</p>',
                unsafe_allow_html=True)
else:
    biggest = max(abs(d) for _, d in deltas) or 1
    rows = ""
    for name, d in deltas[:5]:
        dirn = "up" if d >= 0 else "dn"
        width = abs(d) / biggest * 50
        rows += (f'<div class="factor"><span class="name">{name}</span>'
                 f'<span class="bar"><i class="{dirn}" style="width:{width:.1f}%"></i></span>'
                 f'<span class="amt {dirn}">{d:+.0%}</span></div>')
    st.markdown(rows + '<div style="height:1.6rem"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- notes
with st.expander("How this model works, and where it is weak"):
    st.markdown(
        """
**The model.** An XGBoost classifier in a scikit-learn pipeline with a MinMax scaler, trained on
891 passengers from the Kaggle Titanic training set. It scores about 83% accuracy on a held-out
fifth of that data. It reads seven details: ticket class, sex, age, siblings or spouse aboard,
parents or children aboard, fare, and port of embarkation.

**Why sex dominates.** The recorded survival rates are stark. 97% of first class women survived
and 14% of third class men did. The model reflects that, because it is what the data shows.

**Where it is weak.** Some combinations barely appear in the data. Only six recorded passengers
had five or more parents or children aboard, so a prediction for a large family rests on almost
nothing. The tiles above tell you how much evidence sits behind each estimate, which is worth
checking before you trust a number.

**What it is not.** A historical model fitted to one ship's passenger list. It describes patterns
in that record and says nothing about any living person.
        """
    )
