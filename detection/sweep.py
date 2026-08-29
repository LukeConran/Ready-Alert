import numpy as np
import pandas as pd
from evaluate import load_frames, calibrate
from detector import extract_ear

CALIB_FRAMES  = 300
TEST_SUBJECTS = ['A','B','C','D','E','H','I','J','K','L','M','N','O','P','Q','R','S','U','V','W','X','Y','ZA','ZB','ZC']

THRESHOLDS = np.round(np.arange(0.75, 0.91, 0.01), 2).tolist()  # 0.75 → 0.90, 16 values
ALERT_PCTS = np.round(np.arange(0.20, 0.61, 0.02), 2).tolist()  # 0.20 → 0.60, 21 values
WINDOW_SIZE = 30


def extract_ears(frames):
    return [e for e in (extract_ear(f) for f in frames) if e is not None]


def detect(ears, baseline, threshold, alert_pct):
    window = []
    for e in ears:
        window.append(e < baseline * threshold)
        if len(window) > WINDOW_SIZE:
            window.pop(0)
        if len(window) == WINDOW_SIZE and np.mean(window) >= alert_pct:
            return True
    return False


def load_subject(subject):
    print(f"  [{subject}] extracting EARs...", flush=True)
    alert  = load_frames(subject, 'Non Drowsy')
    drowsy = load_frames(subject, 'Drowsy')
    baseline = calibrate(alert[:CALIB_FRAMES])
    return {
        'baseline':     baseline,
        'alert_ears':   extract_ears(alert[CALIB_FRAMES:]),
        'drowsy_ears':  extract_ears(drowsy),
    }


if __name__ == '__main__':
    print("Loading subjects...")
    data = {s: load_subject(s) for s in TEST_SUBJECTS}

    rows = []
    for thresh in THRESHOLDS:
        for pct in ALERT_PCTS:
            caught, false_alarms = 0, 0
            for s, d in data.items():
                if detect(d['drowsy_ears'], d['baseline'], thresh, pct):
                    caught += 1
                if detect(d['alert_ears'],  d['baseline'], thresh, pct):
                    false_alarms += 1
            rows.append({
                'threshold': thresh,
                'alert_pct': pct,
                'detection_rate':   f"{caught/len(data):.0%}",
                'false_alarm_rate': f"{false_alarms/len(data):.0%}",
            })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = '../data/sweep_results.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
