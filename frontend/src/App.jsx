import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const EXAMPLES = [
  "Honestly the best purchase I've made all year. Arrived early, and the quality is superb.",
  "Stopped working after two days and support never replied. Total waste of money.",
  "Packaging was nice, but the product itself felt cheap and flimsy.",
];

const STEPS = [
  {
    n: "1",
    title: "Paste",
    text: "Drop in any customer review, long or short.",
    tone: "sun",
  },
  {
    n: "2",
    title: "Analyze",
    text: "The model reads it and weighs the wording.",
    tone: "sky",
  },
  {
    n: "3",
    title: "Decide",
    text: "Get a verdict plus how confident it is.",
    tone: "rose",
  },
];

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  async function analyzeSentiment(event) {
    event.preventDefault();

    if (!review.trim()) {
      setError("Please enter a customer review.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review: review.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to analyze the review.");
      }

      setResult(data);

      const s = data?.sentiment?.toLowerCase();
      setHistory((prev) =>
        [
          {
            id: Date.now(),
            text: data.review ?? review.trim(),
            positive: s === "positive" || s === "2",
            confidence:
              typeof data.confidence === "number" ? data.confidence : null,
          },
          ...prev,
        ].slice(0, 5)
      );
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the API. Make sure FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setReview("");
    setResult(null);
    setError("");
  }

  const sentiment = result?.sentiment?.toLowerCase();
  const isPositive = sentiment === "positive" || sentiment === "2";
  const confidence =
    typeof result?.confidence === "number"
      ? Math.min(100, Math.max(0, result.confidence))
      : null;

  const posCount = history.filter((h) => h.positive).length;
  const negCount = history.length - posCount;

  return (
    <div className="page">
      <span className="blob blob-a" aria-hidden="true" />
      <span className="blob blob-b" aria-hidden="true" />
      <span className="blob blob-c" aria-hidden="true" />

      <header className="nav">
        <a className="logo" href="/">
          <span className="logo-mark">f</span>
          feedback<em>ai</em>
        </a>
        <span className="nav-note">
          <span className="pulse" /> model is awake
        </span>
      </header>

      <main className="layout">
        {/* ---------------- left column ---------------- */}
        <aside className="side">
          <p className="kicker">a small tool for reading between the lines</p>
          <h1>
            What are your customers <mark className="hl hl-sun">really</mark>{" "}
            saying?
          </h1>
          <p className="lede">
            Paste in a review and a machine learning model will tell you
            whether it leans <b className="happy-word">happy</b> or{" "}
            <b className="sad-word">unhappy</b>, and how sure it is about that.
          </p>

          <ol className="steps">
            {STEPS.map((s) => (
              <li key={s.n} className={`step tone-${s.tone}`}>
                <span className="step-n">{s.n}</span>
                <div>
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </div>
              </li>
            ))}
          </ol>

          <section className="session">
            <div className="session-head">
              <h3>This session</h3>
              <div className="tally">
                <span className="pill pill-good">👍 {posCount}</span>
                <span className="pill pill-bad">👎 {negCount}</span>
              </div>
            </div>

            {history.length === 0 ? (
              <p className="empty">
                Nothing analyzed yet. Your recent reads will pile up here.
              </p>
            ) : (
              <ul className="history">
                {history.map((h) => (
                  <li key={h.id}>
                    <span
                      className={`dot ${h.positive ? "dot-good" : "dot-bad"}`}
                    />
                    <span className="history-text">{h.text}</span>
                    {h.confidence !== null && (
                      <span className="history-conf">
                        {h.confidence.toFixed(0)}%
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>

        {/* ---------------- right column ---------------- */}
        <section className="notebook">
          <span className="tape" aria-hidden="true" />

          <form onSubmit={analyzeSentiment}>
            <label htmlFor="review">Drop a review here</label>
            <textarea
              id="review"
              value={review}
              onChange={(event) => setReview(event.target.value)}
              placeholder="e.g. The product quality is excellent and delivery was fast..."
              maxLength={10000}
              rows={6}
              disabled={loading}
            />

            <div className="meta-row">
              <span className="count">
                {review.length.toLocaleString()} / 10,000
              </span>
              <button type="button" className="link-btn" onClick={reset}>
                start over
              </button>
            </div>

            <div className="examples">
              <span>No review handy? Try one:</span>
              {EXAMPLES.map((text, i) => (
                <button
                  key={i}
                  type="button"
                  className={`chip chip-${i}`}
                  onClick={() => {
                    setReview(text);
                    setResult(null);
                    setError("");
                  }}
                  disabled={loading}
                >
                  #{i + 1}
                </button>
              ))}
            </div>

            <button
              type="submit"
              className="go"
              disabled={loading || !review.trim()}
            >
              {loading ? "Reading it carefully..." : "Analyze sentiment →"}
            </button>
          </form>

          {error && (
            <div className="oops" role="alert">
              <strong>Hmm.</strong> {error}
            </div>
          )}

          {result && (
            <section
              className={`verdict ${isPositive ? "good" : "bad"}`}
              aria-live="polite"
            >
              <div className="verdict-top">
                <div>
                  <span className="tag">the verdict</span>
                  <h2>{isPositive ? "Positive" : "Negative"}</h2>
                  <p>
                    {isPositive
                      ? "This one sounds like a happy customer."
                      : "This one sounds like an unhappy customer."}
                  </p>
                </div>
                <div className="stamp" aria-hidden="true">
                  <span>{isPositive ? "👍" : "👎"}</span>
                  {isPositive ? "happy" : "unhappy"}
                </div>
              </div>

              {confidence !== null && (
                <div className="gauge">
                  <div className="gauge-label">
                    <span>how sure is the model?</span>
                    <strong>{confidence.toFixed(2)}%</strong>
                  </div>
                  <div className="gauge-track">
                    <div
                      className="gauge-fill"
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>
              )}

              <blockquote>
                <span className="tag">what you wrote</span>
                {result.review}
              </blockquote>
            </section>
          )}
        </section>
      </main>

      <p className="fine">
        Heads up: it's a model, not a mind reader. Sarcasm and mixed feelings
        can trip it up.
      </p>
    </div>
  );
}

export default App;