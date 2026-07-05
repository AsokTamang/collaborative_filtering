# 🎬 Movie Recommender: Collaborative Filtering from Scratch with fastai

A from-first-principles implementation of collaborative filtering for movie recommendations, built on the **MovieLens 100k** dataset. This project walks through the full journey of building a recommender system — starting from a raw dot-product model coded by hand, through bias terms and regularization, to fastai's production-ready `collab_learner`, and finally using the learned embeddings to find similar movies.



---

## 📋 Project Overview

**What problem does this solve?**
Given a history of user ratings for movies (on a 1–5 scale), predict how a given user would rate a movie they haven't seen yet. This is the same core problem behind "Recommended for you" on services like Netflix or Amazon.

**Why it matters**
Collaborative filtering is one of the two foundational approaches to recommendation (the other being content-based filtering). It requires no metadata about movies or users beyond the ratings themselves — the model discovers latent structure (genre affinity, actor preference, tone, etc.) purely from patterns in who rated what highly.

**Who would use this**
- Students/practitioners learning the theory behind recommender systems (this notebook is written in a teaching style, with inline comments explaining *why*, not just *what*)
- Anyone looking for a minimal, readable reference implementation of matrix-factorization-based collaborative filtering in PyTorch/fastai, built up in stages from scratch to library-based

---

## ✨ Features

- 📥 Automatic download and parsing of the MovieLens 100k dataset via fastai's `untar_data`
- 🔨 A `DotProduct` recommender model built from scratch in three progressively more capable versions:
  1. Plain dot product of user/movie embeddings
  2. Dot product + `sigmoid_range` to constrain predictions to valid rating bounds
  3. Dot product + bias terms (captures "generally popular movie" / "generally harsh rater" effects)
- 🧮 A fully manual embedding layer (`create_params`) built with raw `nn.Parameter`, replacing fastai's `Embedding` class, to demystify what an embedding layer actually is
- 🛡️ Weight decay (L2 regularization) applied to fix observed overfitting
- 📊 Model interpretation: extracting the most/least "biased" (i.e. universally loved/disliked) movies from learned bias terms
- 🔁 A second, production-grade implementation using fastai's built-in `collab_learner`
- 🧭 A movie-similarity finder using cosine similarity over learned movie embeddings (e.g. "movies similar to *Titanic*")

---

## 🛠️ Technologies Used

| Category | Details |
|---|---|
| Python Version | 3.12.13 (from notebook kernel metadata) |
| Core Framework | [fastai](https://docs.fast.ai/) (`fastai.collab`, `fastai.tabular.all`) |
| Underlying Framework | PyTorch (`torch`, `torch.nn`) |
| Data Handling | pandas, NumPy |
| Dataset | [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) (via `fastai.data.external.URLs.ML_100k`) |
| Environment | Originally run on Kaggle (kernel metadata indicates a Kaggle notebook; CPU-only — `isGpuEnabled: false`) |

---

## 📁 Project Structure

The notebook is currently a single self-contained file. A suggested structure if this were turned into a repository:

```
collab-filter/
│
├── collab-filter.ipynb     # Main notebook (all code, in order below)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/                   # MovieLens 100k data (auto-downloaded by fastai; not checked into git)
├── models/                 # (Suggested) exported model weights via learn.export()
└── outputs/                # (Suggested) saved prediction results, similarity lookups, etc.
```

> **Assumption:** The `models/` and `outputs/` directories are suggested additions for good repo hygiene — the notebook itself does not currently export/save the trained model or write any files to disk.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AsokTamang/collab-filter.git
cd collab-filter

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

Inferred directly from the notebook's `import` statements:

```txt
fastai
torch
pandas
numpy
```

> **Note:** fastai pulls in a compatible PyTorch version as a dependency automatically. No specific version pins are present in the notebook, so none are asserted here — install the latest compatible versions unless you need to reproduce exact results, in which case pin to whatever fastai/torch versions were active on Kaggle as of the notebook's execution date.

---

## 🗂️ Dataset

**Source:** [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/), fetched automatically via `untar_data(URLs.ML_100k)` — no manual download needed.

**Files used:**
| File | Purpose | Format |
|---|---|---|
| `u.data` | Raw ratings | Tab-separated: `user`, `movie` (ID), `rating`, `timestamp` |
| `u.item` | Movie metadata | Pipe-separated, Latin-1 encoded; only columns 0 and 1 used (`movie` ID, `title`) |

**Merged dataset shape:** After merging ratings with titles on the `movie` ID column, each row becomes `(user, movie, rating, timestamp, title)`.

**Scale:**
- **944** unique users
- **1,665** unique movies
- Ratings are integers from 1–5

**Target variable:** `rating` (integer, 1–5)

**Preprocessing:** fastai's `CollabDataLoaders.from_df()` handles the train/validation split, batching (`bs=64`), and building integer-index vocabularies for both `user` and `title` (accessible via `dls.classes['user']` / `dls.classes['title']`) — this is what allows raw IDs/titles to be converted into embedding indices.

> **Assumption:** The exact train/validation split ratio and random seed for the split are handled internally by `CollabDataLoaders.from_df` and not explicitly overridden in the notebook (`set_seed(42)` is called globally at the top, which does control reproducibility of shuffling/split and weight initialization).

---

## 🔄 Workflow

1. **Data Loading** — Download MovieLens 100k via `untar_data`; read `u.data` and `u.item` into pandas DataFrames.
2. **Data Merging** — Join ratings with movie titles on the `movie` ID.
3. **DataLoader Creation** — Build a `CollabDataLoaders` from the merged DataFrame, keyed on `user` and `title`, batch size 64.
4. **Manual Embedding Exploration** — Create raw random `user_factors`/`movie_factors` tensors (5 factors) to build intuition before formalizing them into a model.
5. **Model v1 — Plain Dot Product** — Build `DotProduct` using `Embedding` layers; predict rating as the dot product of user and movie latent vectors. Train for 5 epochs.
6. **Model v2 — Range-Constrained Output** — Add `y_range=(0,5)` and pass the dot product through `sigmoid_range` so predictions can't fall outside plausible rating bounds. Retrain.
7. **Model v3 — Bias Terms** — Add per-user and per-movie bias embeddings (size 1) to capture systematic effects (e.g., a movie that's universally well-liked regardless of user taste). Retrain.
8. **Diagnosing Overfitting** — Observe validation loss increasing while training loss decreases; identify this as overfitting.
9. **Regularization** — Retrain with weight decay (`wd=0.1`) to shrink weights and reduce overfitting; observe both losses now decreasing together.
10. **Building Embeddings from Scratch** — Demonstrate `nn.Parameter` directly (toy `T` class examples), then implement `create_params()` — a manual replacement for `nn.Embedding` using raw parameter tensors and indexing (`self.user_factors[x[:,0]]` instead of a callable `Embedding` module).
11. **Retraining with Custom Embeddings** — Rebuild `DotProduct` using `create_params` instead of `Embedding`, confirming the from-scratch version trains comparably.
12. **Model Interpretation (Manual Model)** — Extract and rank movie biases to find the most/least "universally rated" movies.
13. **Evaluation** — Run `learn.get_preds()` and compare predicted vs. actual ratings on sample data.
14. **Production Model — `collab_learner`** — Replace the from-scratch model with fastai's built-in `collab_learner` (which internally uses `EmbeddingDotBias`), train with the same hyperparameters.
15. **Model Interpretation (fastai Model)** — Repeat the bias-based popularity analysis on the fastai-native model.
16. **Movie Similarity via Embeddings** — Use cosine similarity between learned movie embedding vectors to find the movie most similar to a given title (*Titanic (1997)*).

---

## 🧠 Model Architecture

Three architectures appear in the notebook, all variations on the same core idea: **matrix factorization via learned embeddings**.

### Final "from scratch" model (Cell 35)

| Component | Detail |
|---|---|
| User embedding | `create_params([n_users, n_factors])` → shape `(944, 50)` |
| Movie embedding | `create_params([n_movies, n_factors])` → shape `(1665, 50)` |
| User bias | `create_params([n_users, 1])` |
| Movie bias | `create_params([n_movies, 1])` |
| Prediction | `sigmoid_range((users * movies).sum(dim=1, keepdim=True) + user_bias + movie_bias, 0, 5)` |
| Initialization | All parameters ~ `N(0, 0.01²)` |

### Production model — `collab_learner`

fastai's built-in `EmbeddingDotBias` module (inspected directly in Cell 46):

```
EmbeddingDotBias(
  (u_weight): Embedding(944, 50)
  (i_weight): Embedding(1665, 50)
  (u_bias): Embedding(944, 1)
  (i_bias): Embedding(1665, 1)
)
```

### Training configuration (consistent across all trained variants)

| Hyperparameter | Value |
|---|---|
| Number of latent factors | 50 |
| Loss function | `MSELossFlat()` (Mean Squared Error) |
| Optimizer/Schedule | `fit_one_cycle` (fastai's 1cycle policy, built on Adam-family optimizer) |
| Learning rate | `5e-3` |
| Epochs | 5 (per training run; several separate training runs are shown) |
| Weight decay | `0.1` (applied in overfitting-fix and later runs) |
| Batch size | 64 |
| Output range | `y_range = (0, 5)` via `sigmoid_range` |

---

## 🔍 Code Walkthrough

### `DotProduct(Module)` — plain version (Cell 16)
**Purpose:** Simplest possible latent-factor recommender.
**Inputs:** Batch `x` of shape `(batch_size, 2)` — columns are `[user_id, movie_id]`.
**Outputs:** Raw predicted rating (unconstrained real number) per row.
**How it fits in:** Baseline to demonstrate that a dot product of two learned vectors can capture user–movie affinity, before adding refinements.

### `DotProduct(Module)` — with `y_range` (Cell 23)
**Purpose:** Same as above, but passes the dot product through `sigmoid_range(x, low, high)` so outputs are guaranteed to fall in `[0, 5]` — since raw dot products are unbounded but ratings are not.
**Inputs/Outputs:** Same shape as above; output is now bounded.

### `DotProduct(Module)` — with bias terms (Cell 26)
**Purpose:** Adds a scalar bias per user and per movie to the dot product, letting the model separately capture "this user rates everything highly/low" and "this movie is broadly loved/hated," rather than forcing all of that signal into the interaction term alone.
**Inputs/Outputs:** Same interface; `rating` is computed with `keepdim=True` so the bias terms (shape `(batch_size, 1)`) broadcast correctly before summing.

### `create_params(size)` (Cell 34)
**Purpose:** Manually constructs a learnable embedding-like parameter tensor without using `nn.Embedding`.
**Inputs:** `size` — a list/tuple, e.g. `[n_users, n_factors]`.
**Outputs:** An `nn.Parameter` of that shape, initialized from `N(0, 0.01²)`.
**How it fits in:** Used in the final from-scratch `DotProduct` (Cell 35) to demonstrate that `Embedding` is really just a parameter matrix plus indexing — replacing `self.user_factors(x[:,0])` with `self.user_factors[x[:,0]]`.

### `DotProduct(Module)` — from-scratch embeddings (Cell 35)
**Purpose:** Final and most complete "from scratch" model — combines bias terms with manually-built embeddings (`create_params`) instead of `nn.Embedding`.
**Inputs/Outputs:** Same interface as previous versions.
**How it fits in:** Bridges the gap between understanding embeddings conceptually and using fastai's/PyTorch's built-in, production-ready `Embedding` and `EmbeddingDotBias` classes.

### Movie similarity lookup (Cell 50)
**Purpose:** Given a trained `collab_learner`, find the movie most similar to a target movie by comparing latent embedding vectors.
**Logic:**
1. Extract the full movie embedding matrix: `learn.model.i_weight.weight` → shape `(1665, 50)`.
2. Look up the target movie's row index via `dls.classes['title'].o2i[...]`.
3. Compute cosine similarity between that one movie's vector and every other movie's vector (`nn.CosineSimilarity(dim=1)`).
4. Sort similarity scores descending; skip index `0` (the movie itself, similarity = 1.0) and take index `1` — the closest *other* movie.

---

## 📈 Results

### Training loss progression (from-scratch model with bias + weight decay, Cell 29)

| Epoch | Train Loss | Valid Loss |
|---|---|---|
| 0 | 0.3818 | 0.9117 |
| 1 | 0.4034 | 0.9117 |
| 2 | 0.3812 | 0.8961 |
| 3 | 0.3401 | 0.8834 |
| 4 | 0.3299 | 0.8799 |

*(This is a continuation of an earlier 5-epoch run without weight decay, shown in Cell 27, where validation loss was increasing — the overfitting signal that motivated adding `wd=0.1`.)*

### `collab_learner` training (Cell 44)

| Epoch | Train Loss | Valid Loss |
|---|---|---|
| 0 | 0.9111 | 0.9498 |
| 1 | 0.7535 | 0.8956 |
| 2 | 0.6201 | 0.8625 |
| 3 | 0.5273 | 0.8416 |
| 4 | 0.5181 | 0.8376 |

*(Loss is MSE on a 0–5 rating scale — a validation loss around 0.84 corresponds to a typical prediction error of roughly √0.84 ≈ 0.9 rating points.)*

### Sample predictions vs. actual ratings (from-scratch model, Cell 42)

| Predicted | Actual |
|---|---|
| 3.46 | 4 |
| 2.56 | 4 |
| 2.89 | 4 |
| 4.19 | 5 |
| 4.08 | 5 |

### Movies with lowest learned bias (manual model, Cell 39)
Grease 2 (1982), Showgirls (1995), Lawnmower Man 2: Beyond Cyberspace (1996), Theodore Rex (1995), Robocop 3 (1993)

### Movies with highest learned bias (manual model, Cell 40)
Titanic (1997), Good Will Hunting (1997), Schindler's List (1993), Shawshank Redemption, The (1994), L.A. Confidential (1997)

### Movies with lowest/highest bias (fastai `collab_learner`, Cells 48–49)
- **Lowest:** Amityville 1992: It's About Time, Lawnmower Man 2: Beyond Cyberspace, Spice World, Children of the Corn: The Gathering, Second Jungle Book: Mowgli & Baloo
- **Highest:** Titanic (1997), L.A. Confidential (1997), Schindler's List (1993), Good Will Hunting (1997), Rear Window (1954)

### Example Output — Movie Similarity

```
The movie closest to Titanic is: Crimson Tide (1995)
```

---

## ▶️ How to Run

1. Complete the [Installation](#️-installation) steps above.
2. Launch Jupyter: `jupyter notebook collab-filter.ipynb`
3. Run all cells top to bottom — the notebook downloads MovieLens 100k automatically on first run (no manual dataset setup needed).
4. Training each model variant takes ~30–35 seconds on CPU for 5 epochs (based on the ~6–7s/epoch timing shown in the notebook's own output).
5. To try the similarity lookup on a different movie, change the title string passed to `dls.classes['title'].o2i[...]` in the final cell to any exact title present in `u.item` (title strings must match exactly, including the year suffix, e.g. `'Titanic (1997)'`).

---

## 🚀 Future Improvements

- **Persist the trained model** with `learn.export()` so it doesn't need retraining on every run.
- **Hyperparameter search** over `n_factors`, learning rate, and weight decay rather than the fixed values used here.
- **Proper held-out test set** — currently only a train/validation split is used; a separate test set would give a less optimistic final evaluation.
- **Quantitative similarity evaluation** — the movie-similarity feature is currently only spot-checked qualitatively (one example movie); could be validated against genre metadata if `u.item`'s genre columns are incorporated.
- **Hybrid model** — incorporate movie genre/user demographic metadata (available in `u.item`/`u.user` but unused here) for a hybrid content + collaborative approach.
- **Deployment** — wrap the trained `collab_learner` in a simple API (e.g. FastAPI) for serving live rating predictions.

---



- [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) — GroupLens Research, University of Minnesota
- [fastai](https://www.fast.ai/) and its accompanying book/course, *Practical Deep Learning for Coders* — the `DotProduct`-from-scratch approach and progression toward `collab_learner` closely follows this course's collaborative filtering chapter
- [PyTorch](https://pytorch.org/) — underlying tensor/autograd engine
