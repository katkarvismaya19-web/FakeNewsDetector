"""Styling + HTML building blocks for the Streamlit app."""
import html

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Devanagari:wght@400;500&display=swap');
:root{--ink:#16202E;--muted:#5B6576;--faint:#8A93A3;--line:#E2E6EC;--card:#FFFFFF;--accent:#1F4FD1;
--fake:#B42318;--fake-bg:#FEF3F2;--real:#067647;--real-bg:#ECFDF3;--warn:#B54708;--warn-bg:#FFFAEB}
html,body,[class*="css"],.stMarkdown,textarea,input,button{font-family:'IBM Plex Sans','IBM Plex Sans Devanagari',sans-serif!important}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
.block-container{max-width:1080px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border-radius:12px!important;border-color:var(--line)!important}
.stTextArea textarea{font-size:15px;line-height:1.6}
.brand{display:flex;align-items:center;gap:10px}
.brand svg{flex:none}
.brand b{font-size:17px;font-weight:600;color:var(--ink)}
.brand span{display:block;font-size:12px;color:var(--muted)}
.hr{border-top:1px solid var(--line);margin:14px 0 24px}
.label{font-size:12px;color:var(--muted);margin:0 0 6px;letter-spacing:.2px}
.verdict{border-radius:12px;padding:20px 22px;border:1px solid}
.verdict h2{margin:0;font-size:26px;font-weight:600}
.verdict p{margin:6px 0 0;font-size:14px}
.meter{height:8px;border-radius:4px;background:rgba(0,0,0,.08);margin-top:14px;overflow:hidden}
.meter div{height:100%;border-radius:4px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;height:100%}
.stat .v{font-size:18px;font-weight:600;color:var(--ink);margin-top:2px}
.mrow{display:grid;grid-template-columns:58px 1fr 64px 52px;gap:12px;align-items:center;padding:10px 0;border-top:1px solid var(--line);font-size:14px}
.mrow.head{border-top:none;font-size:12px;color:var(--faint);padding-top:0}
.track{height:6px;border-radius:3px;background:#EEF1F5;overflow:hidden}
.track div{height:100%}
.tag{font-size:12px;font-weight:500;padding:2px 8px;border-radius:6px;display:inline-block;text-align:center}
.t-fake{background:var(--fake-bg);color:var(--fake)}.t-real{background:var(--real-bg);color:var(--real)}
.t-unclear{background:#F2F4F7;color:var(--muted)}.t-fc{background:#EEF2FF;color:var(--accent)}
.src{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;background:var(--card)}
.src .top{display:flex;justify-content:space-between;gap:8px;align-items:center}
.src .pub{font-weight:600;font-size:14px;color:var(--ink)}
.src a{display:block;margin:6px 0;font-size:14px;color:var(--accent);text-decoration:none;line-height:1.45}
.src a:hover{text-decoration:underline}
.src .meta{font-size:12px;color:var(--faint)}
.empty{border:1px dashed var(--line);border-radius:12px;padding:28px;text-align:center;color:var(--muted);font-size:14px}
.pitch h1{font-size:34px;line-height:1.2;font-weight:600;color:var(--ink);margin:18px 0 12px}
.pitch p{color:var(--muted);font-size:15px;line-height:1.6;margin:0 0 22px}
.pitch li{color:var(--ink);font-size:14px;margin:0 0 10px;list-style:none;padding-left:22px;position:relative}
.pitch li:before{content:"";position:absolute;left:0;top:7px;width:8px;height:8px;border-radius:2px;background:var(--accent)}
.pitch ul{padding:0;margin:0}
</style>
"""

LOGO = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#1F4FD1" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6z"/>'
        '<circle cx="11.5" cy="11.5" r="2.6"/><path d="M13.4 13.4l2.1 2.1"/></svg>')

e = html.escape


def brand(sub="Hindi and English news verification"):
    return f'<div class="brand">{LOGO}<div><b>Fake News Detector</b><span>{e(sub)}</span></div></div>'


def verdict_card(p_fake, basis):
    conf = max(p_fake, 1 - p_fake) * 100
    if conf < 60:
        title, fg, bg, bar = "Inconclusive", "var(--warn)", "var(--warn-bg)", "var(--warn)"
    elif p_fake >= 0.5:
        title, fg, bg, bar = "Likely fake", "var(--fake)", "var(--fake-bg)", "var(--fake)"
    else:
        title, fg, bg, bar = "Likely real", "var(--real)", "var(--real-bg)", "var(--real)"
    return (f'<div class="verdict" style="background:{bg};border-color:{fg}33">'
            f'<div class="label" style="color:{fg}">Verdict</div><h2 style="color:{fg}">{title}</h2>'
            f'<p style="color:var(--ink)">{conf:.1f}% confidence · {e(basis)}</p>'
            f'<div class="meter"><div style="width:{conf:.1f}%;background:{bar}"></div></div></div>')


def stat(label, value):
    return f'<div class="stat"><div class="label">{e(label)}</div><div class="v">{e(value)}</div></div>'


def model_table(probs):
    rows = ['<div class="mrow head"><span>Model</span><span>Probability of fake</span><span>Result</span><span>Conf.</span></div>']
    for n, p in sorted(probs.items(), key=lambda kv: ["bert", "lstm", "rnn", "llm"].index(kv[0])):
        fake = p >= 0.5
        col = "var(--fake)" if fake else "var(--real)"
        rows.append(f'<div class="mrow"><span style="font-weight:500">{n.upper()}</span>'
                    f'<div class="track"><div style="width:{p * 100:.0f}%;background:{col}"></div></div>'
                    f'<span class="tag {"t-fake" if fake else "t-real"}">{"Fake" if fake else "Real"}</span>'
                    f'<span style="color:var(--muted)">{max(p, 1 - p) * 100:.0f}%</span></div>')
    return "".join(rows)


def source_card(s):
    kind = '<span class="tag t-fc">Fact-check</span>' if s["type"] == "fact-check" else '<span class="tag t-unclear">News</span>'
    stance_cls = {"FAKE": "t-fake", "REAL": "t-real"}.get(s.get("stance"), "t-unclear")
    rating = e(s["rating"] or s.get("stance", "").title() or "—")
    url = s["url"] if s["url"].startswith(("http://", "https://")) else "#"
    return (f'<div class="src"><div class="top"><span class="pub">{e(s["publisher"] or "Unknown source")} {kind}</span>'
            f'<span class="tag {stance_cls}">{rating}</span></div>'
            f'<a href="{e(url)}" target="_blank" rel="noopener">{e(s["text"][:160])}</a>'
            f'<div class="meta">Match {s.get("similarity", 0) * 100:.0f}% · {e(s.get("date", ""))}</div></div>')


def empty(msg):
    return f'<div class="empty">{e(msg)}</div>'
