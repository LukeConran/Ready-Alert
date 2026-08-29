import os
import cv2
import numpy as np
import pandas as pd
from detector import calibrate, run, extract_ear, EAR_THRESHOLD

CALIB_FRAMES = 300  # first 300 non-drowsy frames used as baseline
DATA = 'data/Driver Drowsiness Dataset (DDD)'


def load_frames(subject, cls):
    src = os.path.join(DATA, cls)
    files = sorted(f for f in os.listdir(src) if f.upper().startswith(subject))
    return [cv2.imread(os.path.join(src, f)) for f in files]


def ear_stats(frames, baseline):
    ears = [extract_ear(f) for f in frames]
    ears = [e for e in ears if e is not None]
    if not ears:
        return {}
    return {'mean': round(np.mean(ears), 3), 'min': round(np.min(ears), 3),
            'pct_below': round(np.mean(np.array(ears) < baseline * EAR_THRESHOLD), 2)}


def evaluate_subject(subject):
    print(f"  loading frames...", end=' ', flush=True)
    alert_frames  = load_frames(subject, 'Non Drowsy')
    drowsy_frames = load_frames(subject, 'Drowsy')

    if len(alert_frames) < CALIB_FRAMES:
        print("skipped (too few alert frames)")
        return None

    print(f"calibrating...", end=' ', flush=True)
    baseline = calibrate(alert_frames[:CALIB_FRAMES])

    print(f"detecting...", end=' ', flush=True)
    tp = run(drowsy_frames, baseline)
    fp = run(alert_frames[CALIB_FRAMES:], baseline)
    ds = ear_stats(drowsy_frames, baseline)
    print("done.")

    return {
        'subject':         subject,
        'baseline_ear':    round(baseline, 3),
        'drowsy_mean_ear': ds.get('mean'),
        'drowsy_min_ear':  ds.get('min'),
        'drowsy_pct_below': ds.get('pct_below'),
        'caught_drowsy':   tp,
        'false_alarm':     fp,
    }


if __name__ == '__main__':
    test_subjects = ['K', 'R', 'W', 'Y', 'ZC']
    results = []
    for s in test_subjects:
        print(f"[{s}]")
        results.append(evaluate_subject(s))
    df = pd.DataFrame([r for r in results if r])
    print("\n" + df.to_string(index=False))
    print(f"\nDetection rate:  {df.caught_drowsy.mean():.0%}")
    print(f"False alarm rate: {df.false_alarm.mean():.0%}")
