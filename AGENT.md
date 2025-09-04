# Berghain Challenge
You're the bouncer at a night club. Your goal is to fill the venue with N=1000 people while satisfying constraints like "at least 40% Berlin locals", or "at least 80% wearing all black". People arrive one by one, and you must immediately decide whether to let them in or turn them away. Your challenge is to fill the venue with as few rejections as possible while meeting all minimum requirements.


## How it works
- People arrive sequentially with binary attributes (e.g., female/male, young/old, regular/new)
- You must make immediate accept/reject decisions
- The game ends when either:
(a) venue is full (1000 people)
(b) you rejected 20,000 people


## Scenarios & Scoring
There are 3 different scenarios. For each, you are given a list of constraints and statistics on the attribute distribution. You can assume, participants are sampled i.i.d., meaning the attribute distribution will not change as the night goes on. You know the overall relative frequency of each attribute and the correlation between attributes. You don't know the exact distribution.
You score is the number of people you rejected before filling the venue (the less the better).

## Additional info

I also have additional info about the full distribution of people that are being sampled.

```python
constraints = [
    Constraint(attribute="techno_lover", min_count=650),
    Constraint(attribute="well_connected", min_count=450),
    Constraint(attribute="creative", min_count=300),
    Constraint(attribute="berlin_local", min_count=750)
]

distribution = [
    ({}, 63/2341),  
    ({'berlin_local': True}, 136/2341),  
    ({'creative': True}, 3/2341),  
    ({'creative': True, 'berlin_local': True}, 3/2341),  
    ({'well_connected': True}, 72/2341),  
    ({'well_connected': True, 'berlin_local': True}, 614/2341),  
    ({'well_connected': True, 'creative': True}, 4/2341),  
    ({'well_connected': True, 'creative': True, 'berlin_local': True}, 16/2341),  
    ({'techno_lover': True}, 970/2341),  
    ({'techno_lover': True, 'berlin_local': True}, 35/2341),  
    ({'techno_lover': True, 'creative': True}, 13/2341),  
    ({'techno_lover': True, 'creative': True, 'berlin_local': True}, 11/2341),  
    ({'techno_lover': True, 'well_connected': True}, 231/2341),  
    ({'techno_lover': True, 'well_connected': True, 'berlin_local': True}, 85/2341),  
    ({'techno_lover': True, 'well_connected': True, 'creative': True}, 18/2341),  
    ({'techno_lover': True, 'well_connected': True, 'creative': True, 'berlin_local': True}, 67/2341),  
]
```

### Solver

Your solver should be in this form:

```python
@dataclass
class Constraint:
    attribute: str
    min_count: int


class YourSolver:
    def __init__(
        self,
        N: int = 1000,
        ...
    ):
        # ...

    def initialize_policy(
        self,
        constraints: List[Constraint],
        distribution: List[Tuple[Dict[str, bool], float]],
    ):
        # ...

    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        # ...
```

### Your Task

Please propose an algorithm for solving, and give the minimum number of possible rejections. Here's some ideas:

- We can do multiple runs and we only care care about the *minimum of all runs*, so we can have a policy which may admit too many people, in the hope that at least 1 run is lucky.
- We have a simulator, so we can fit hyperparameters if needed.