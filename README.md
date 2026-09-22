# Detecting Cyber-Attacks on UAVs

Group project: a machine learning system trained on a real UAV testbed dataset to classify normal flight traffic versus network-layer attacks.

## Attack classes

- Deauthentication DoS
- Replay
- False data injection
- Evil twin

## Team

| Name | Responsibility |
|---|---|
| Tracey Parks | Project Overview, Problem, Motivation slides |
| Savion Parks | TBD |
| Robert Kosie-Williams | TBD |
| Malachi Bullock | TBD |

## Repo structure

```
uav-cyberattack-detection/
├── data/
│   ├── raw/            # original downloaded dataset(s) — not committed, see .gitignore
│   └── processed/      # cleaned/feature-engineered data ready for modeling
├── notebooks/           # exploratory analysis, one notebook per person/topic
├── src/                 # reusable scripts (data loading, preprocessing, model training)
├── docs/                 # slide deck exports, write-ups, requirement docs
└── README.md
```

## Getting started

1. Clone the repo: `git clone <repo-url>`
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Drop raw dataset files into `data/raw/` (they're gitignored, so download them locally rather than expecting them from the repo)
4. Work in your own branch: `git checkout -b <yourname>-<feature>`
5. Commit small, focused changes: `git add . && git commit -m "clear message"`
6. Push and open a pull request into `main`: `git push origin <yourname>-<feature>`

## Dataset

Using a real UAV cyber-physical testbed dataset (network + telemetry) covering normal operation and the four attack classes above. Add the exact source/link here once the team finalizes which dataset version you're using.

## Notes

- Keep large data files out of git (see `.gitignore`) — use a shared Drive link or note the download source in this README instead
- One notebook or script per contribution where possible, so changes are easy to review in pull requests
