# Weekly Handoff — How to Update the Squad/Starters

The weekly run fires automatically every **Wednesday 6pm AEST**. For it to use the latest team, drop one file into Google Drive each week.

## What to do each week

1. NRL releases confirmed team lists **Tuesday ~6:30pm AEST**.
2. Open Google Drive → `NRL-Supercoach/inputs/`.
3. Find the most recent file, e.g. `round_9.yaml`. Right-click → **Make a copy**.
4. Rename the copy to `round_{N+1}.yaml` (e.g. `round_10.yaml`).
5. Open it (Drive opens YAML as plain text) and edit two sections:
   - **`squad:`** — the 26 players currently in the team. Only change if a trade was made the previous week.
   - **`starters:`** — the players named to play this round (from the NRL confirmed team lists).
6. Save. That's it — leave both the old and new files in the folder.

The Wednesday 6pm AEST run will detect the highest-numbered file automatically and push results into `NRL-Supercoach/round_{N+1}/`.

## Where do the results go?

After each Wednesday run, look in Google Drive at:

```
NRL-Supercoach/round_{N}/
  ├── team_round_{N}.csv      <- upload to Supercoach site
  ├── trade_advice_r{N}.csv   <- recommended trades
  ├── trade_map_r{N}.png      <- visual trade analysis
  ├── bye_analysis.png
  ├── model_performance.png
  ├── player_analysis.png
  ├── round_summary.csv
  ├── season_plan.csv
  └── trade_plan.csv
```

## What if I miss a week?

The workflow uses the **highest-numbered** `round_*.yaml`. If you don't upload a new one, it re-runs on last week's squad/starters — not catastrophic, but trades won't reflect.

## What if it fails?

GitHub will email Michael automatically when a scheduled run fails.
