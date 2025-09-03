A solver for https://berghain.challenges.listenlabs.ai

### setup

```bash
pip install -r requirements.txt
```

### usage

```sh
python src/berghain.py --solver lp --scenario 1
python src/berghain.py --solver lp --scenario 2
python src/berghain.py --solver lp --scenario 3
```

example output:

```sh
                                           Statistics                                            
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Category                                                ┃ Admitted ┃ Rejected ┃ Adm % ┃ Rej % ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ none                                                    │        0 │       29 │  0.0% │ 90.6% │
│ berlin_local                                            │       58 │        0 │  5.8% │  0.0% │
│ creative                                                │        1 │        0 │  0.1% │  0.0% │
│ creative + berlin_local                                 │        0 │        0 │  0.0% │  0.0% │
│ well_connected                                          │       24 │        2 │  2.4% │  6.2% │
│ well_connected + berlin_local                           │      274 │        0 │ 27.4% │  0.0% │
│ well_connected + creative                               │        5 │        0 │  0.5% │  0.0% │
│ well_connected + creative + berlin_local                │        6 │        0 │  0.6% │  0.0% │
│ techno_lover                                            │      420 │        1 │ 42.0% │  3.1% │
│ techno_lover + berlin_local                             │       11 │        0 │  1.1% │  0.0% │
│ techno_lover + creative                                 │        6 │        0 │  0.6% │  0.0% │
│ techno_lover + creative + berlin_local                  │        6 │        0 │  0.6% │  0.0% │
│ techno_lover + well_connected                           │       94 │        0 │  9.4% │  0.0% │
│ techno_lover + well_connected + berlin_local            │       48 │        0 │  4.8% │  0.0% │
│ techno_lover + well_connected + creative                │       13 │        0 │  1.3% │  0.0% │
│ techno_lover + well_connected + creative + berlin_local │       33 │        0 │  3.3% │  0.0% │
└─────────────────────────────────────────────────────────┴──────────┴──────────┴───────┴───────┘
  Admitted: 999, Rejected: 31, Remaining Capacity: 1
  Constraints: techno_lover: 631/650 (97.1%) | well_connected: 497/450 (110.4%) | creative: 70/300 (23.3%) | berlin_local: 436/750 (58.1%)
Game ended with status failed: https://berghain.challenges.listenlabs.ai/game/e039817a-3431-4955-bd97-ec255c625492
```