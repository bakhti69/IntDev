# How to test TrafficWatch (for team members)

Three levels. Level 1 takes 5 minutes; do it at least. Levels 2 and 3 are for a deeper check.

Links:

* Website: https://bakhti69.github.io/IntDev/
* Code: https://github.com/bakhti69/IntDev

---

## Level 1: check the website (5 min, nothing to install)

Open https://bakhti69.github.io/IntDev/ and press **Ctrl+F5** so you see the latest version. Check:

1. **Team.** All three members are listed, the captain is marked, and the GitHub links open.
2. **Results on the sample videos.**
   * Click each video tab (C3896, C3897, C3902, C3905) and press play. The video shows coloured boxes around cars and people and a coloured timeline strip under it.
   * Click an event on the timeline. The video should jump to that moment.
   * Watch 3–4 events and ask yourself whether it really happened. Note the video, the time and what you saw.
3. **EDA and Dashboard.** Pictures (heat map, trajectories) and charts load, with nothing empty.
4. **Live demo.** The "Open the demo in Colab" button opens a Google Colab page.
5. **Report.** It reads well, with no typos or broken text.
6. **Phone.** Open the site on your phone as well. Check that nothing overflows the screen.

---

## Level 2: run the live demo (15 min, only a Google account)

This runs the real model on a free Google GPU in the cloud.

1. On the website, click **Try the live demo**, then **Open the demo in Colab**. Sign in with Google if asked.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU → Save**.
3. Choose **Runtime → Run all**. If it says "This notebook was not authored by Google", click **Run anyway**.
4. Wait about 3 minutes. The last cell prints:
   `Running on public URL: https://xxxxxxxx.gradio.live`
   Click that link.
5. Upload a short **.mp4 from the competition camera**: up to 5 minutes and 1 GB, and 30–60 s is ideal. Leave "Compute the accident-risk curve" ticked and click **Detect events**.
6. You get:
   * an annotated video
   * an event timeline
   * the accident-risk curve (the dashed red line at 0.5 is the alarm level)
   * a table of events you can download as `events.json`
7. The note under the button should say **"GPU inference"** and **"scene registration: ok"**.
8. When done, close the Colab tab. The link stops working when the notebook stops, and that is expected.

If the video is too big, ask Baxtiyorjon for a short clip, or cut one yourself with ffmpeg:

```
ffmpeg -i C3905.mp4 -t 60 -vf scale=1920:-2 -c:v libx264 -crf 23 -an clip60.mp4
```

---

## Level 3: run the official submission on your own computer (1–2 h, Windows/Mac/Linux)

This is exactly what the organisers will run, so it is the most important check. You need Python 3.10 or newer and Git.

On Windows, use **Git Bash**. **Don't press Ctrl+C in Git Bash:** it stops the program. Copy with right-click instead.

```
git clone https://github.com/bakhti69/IntDev
cd IntDev
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
mkdir samples
```

Copy one or more sample videos (.mp4) into the `samples` folder, then run:

```
python run_submission.py --videos samples --out my_predictions.json --team IntDev 2>&1 | tee my_run.log
python evaluate.py --pred my_predictions.json --validate-only
```

**What should happen:**

* For every video, the run prints a line like `C3905.mp4 ... 27 events ... OK`.
* The last command ends with `0 error(s) ... -> VALID`.
* The time per video stays well under 3× the video's length. Without a GPU it is about 1.5–2×.
* Running it twice gives the same `my_predictions.json` (the output is deterministic).

Optional unit tests:

```
pip install pytest
python -m pytest -q
```

Expected: `26 passed`.

---

## What to send back

For any problem, send one message with:

* **Which level and step** (e.g. "Level 2, step 5").
* **A screenshot** of the error or the wrong result.
* **For wrong detections:** the video name, the time (e.g. `C3897 2:46`), what the system said and what really happened.
* **For Level 3:** the `my_run.log` file.
