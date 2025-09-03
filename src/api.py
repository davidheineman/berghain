import requests
import time
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Constraint:
    attribute: str
    min_count: int

class BerghainAPIClient:
    """Handles all API interactions with the Berghain challenge server."""
    
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        
    def create_game(self, scenario: int, player_id: str, max_retries: int = 3) -> Dict:
        """Create a new game with retry logic for rate limits."""
        url = f"{self.base_url}/new-game"
        params = {"scenario": scenario, "playerId": player_id}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = (attempt + 1) * 5
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            except requests.exceptions.Timeout:
                raise Exception("Game creation timed out")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Game creation failed: {e}")
                time.sleep(2)
        
        raise Exception("Max retries exceeded")
    
    def make_decision(self, game_id: str, person_index: int, accept: bool) -> Dict:
        """Make a decision and get the next person."""
        url = f"{self.base_url}/decide-and-next?gameId={game_id}&personIndex={person_index}&accept={str(accept).lower()}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise Exception("Decision timed out")
        except Exception as e:
            raise Exception(f"Decision failed: {e}")
    
    def get_first_person(self, game_id: str) -> Dict:
        """Get the first person to start the game."""
        url = f"{self.base_url}/decide-and-next"
        params = {"gameId": game_id, "personIndex": 0}
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def parse_constraints(self, game_data: Dict) -> List[Constraint]:
        """Parse constraints from game data."""
        return [Constraint(c["attribute"], c["minCount"]) for c in game_data["constraints"]]
