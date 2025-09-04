from api import Constraint

def get_constraints(scenario):
    if scenario == 2:
        return [
            Constraint(attribute="techno_lover", min_count=650),
            Constraint(attribute="well_connected", min_count=450),
            Constraint(attribute="creative", min_count=300),
            Constraint(attribute="berlin_local", min_count=750)
        ]
    elif scenario == 3:
        return [
            Constraint(attribute="underground_veteran", min_count=500),
            Constraint(attribute="international", min_count=650),
            Constraint(attribute="fashion_forward", min_count=550),
            Constraint(attribute="queer_friendly", min_count=250),
            Constraint(attribute="vinyl_collector", min_count=200),
            Constraint(attribute="german_speaker", min_count=800)
        ]


def get_corr(scenario: int):
    if scenario == 2:
        return {
            'techno_lover': {
                'techno_lover': 1,
                'well_connected': -0.4696169332674324,
                'creative': 0.09463317039891586,
                'berlin_local': -0.6549403815606182
            },
            'well_connected': {
                'techno_lover': -0.4696169332674324,
                'well_connected': 1,
                'creative': 0.14197259140471485,
                'berlin_local': 0.5724067808436452
            },
            'creative': {
                'techno_lover': 0.09463317039891586,
                'well_connected': 0.14197259140471485,
                'creative': 1,
                'berlin_local': 0.14446459505650772
            },
            'berlin_local': {
                'techno_lover': -0.6549403815606182,
                'well_connected': 0.5724067808436452,
                'creative': 0.14446459505650772,
                'berlin_local': 1
            }
        }
    elif scenario == 3:
        return {
            'underground_veteran': {
                'underground_veteran': 1,
                'international': -0.08110175777152992,
                'fashion_forward': -0.1696563475505309,
                'queer_friendly': 0.03719928376753885,
                'vinyl_collector': 0.07223521156389842,
                'german_speaker': 0.11188766703422799
            },
            'international': {
                'underground_veteran': -0.08110175777152992,
                'international': 1,
                'fashion_forward': 0.375711059360155,
                'queer_friendly': 0.0036693314388711686,
                'vinyl_collector': -0.03083247098181075,
                'german_speaker': -0.7172529382519395
            },
            'fashion_forward': {
                'underground_veteran': -0.1696563475505309,
                'international': 0.375711059360155,
                'fashion_forward': 1,
                'queer_friendly': -0.0034530926793377476,
                'vinyl_collector': -0.11024719606358546,
                'german_speaker': -0.3521024461597403
            },
            'queer_friendly': {
                'underground_veteran': 0.03719928376753885,
                'international': 0.0036693314388711686,
                'fashion_forward': -0.0034530926793377476,
                'queer_friendly': 1,
                'vinyl_collector': 0.47990640803167306,
                'german_speaker': 0.04797381132680503
            },
            'vinyl_collector': {
                'underground_veteran': 0.07223521156389842,
                'international': -0.03083247098181075,
                'fashion_forward': -0.11024719606358546,
                'queer_friendly': 0.47990640803167306,
                'vinyl_collector': 1,
                'german_speaker': 0.09984452286269897
            },
            'german_speaker': {
                'underground_veteran': 0.11188766703422799,
                'international': -0.7172529382519395,
                'fashion_forward': -0.3521024461597403,
                'queer_friendly': 0.04797381132680503,
                'vinyl_collector': 0.09984452286269897,
                'german_speaker': 1
            }
        }

def get_frequencies(scenario: int):
    if scenario == 2:
        return {
            'techno_lover': 0.6265000000000001,
            'well_connected': 0.4700000000000001,
            'creative': 0.06227,
            'berlin_local': 0.398
        }
    elif scenario == 3:
        return {
            'underground_veteran': 0.6794999999999999,
            'international': 0.5735,
            'fashion_forward': 0.6910000000000002,
            'queer_friendly': 0.04614,
            'vinyl_collector': 0.044539999999999996,
            'german_speaker': 0.4565000000000001
        }

def get_distribution(scenario: int):
    if scenario == 1:
        return [
            ({}, 1.0),  
        ]
    elif scenario == 2:
        return [
            ({}, 437/14000),  
            ({'berlin_local': True}, 655/14000),  
            ({'creative': True}, 13/14000),  
            ({'creative': True, 'berlin_local': True}, 25/14000),  
            ({'well_connected': True}, 430/14000),  
            ({'well_connected': True, 'berlin_local': True}, 3518/14000),  
            ({'well_connected': True, 'creative': True}, 39/14000),  
            ({'well_connected': True, 'creative': True, 'berlin_local': True}, 81/14000),  
            ({'techno_lover': True}, 5858/14000),  
            ({'techno_lover': True, 'berlin_local': True}, 212/14000),  
            ({'techno_lover': True, 'creative': True}, 71/14000),  
            ({'techno_lover': True, 'creative': True, 'berlin_local': True}, 102/14000),  
            ({'techno_lover': True, 'well_connected': True}, 1379/14000),  
            ({'techno_lover': True, 'well_connected': True, 'berlin_local': True}, 638/14000),  
            ({'techno_lover': True, 'well_connected': True, 'creative': True}, 154/14000),  
            ({'techno_lover': True, 'well_connected': True, 'creative': True, 'berlin_local': True}, 388/14000),  
        ]
    elif scenario == 3:
        return [
            ({}, 262/20000),  
            ({'german_speaker': True}, 469/20000),  
            ({'vinyl_collector': True}, 3/20000),  
            ({'vinyl_collector': True, 'german_speaker': True}, 16/20000),  
            ({'queer_friendly': True}, 3/20000),  
            ({'queer_friendly': True, 'german_speaker': True}, 5/20000),  
            ({'queer_friendly': True, 'vinyl_collector': True}, 6/20000),  
            ({'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 5/20000),  
            ({'fashion_forward': True}, 259/20000),  
            ({'fashion_forward': True, 'german_speaker': True}, 1261/20000),  
            ({'fashion_forward': True, 'vinyl_collector': True}, 7/20000),  
            ({'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 7/20000),  
            ({'fashion_forward': True, 'queer_friendly': True}, 7/20000),  
            ({'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 26/20000),  
            ({'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 8/20000),  
            ({'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 13/20000),  
            ({'international': True}, 332/20000),  
            ({'international': True, 'german_speaker': True}, 108/20000),  
            ({'international': True, 'vinyl_collector': True}, 3/20000),  
            ({'international': True, 'vinyl_collector': True, 'german_speaker': True}, 7/20000),  
            ({'international': True, 'queer_friendly': True}, 6/20000),  
            ({'international': True, 'queer_friendly': True, 'german_speaker': True}, 4/20000),  
            ({'international': True, 'queer_friendly': True, 'vinyl_collector': True}, 7/20000),  
            ({'international': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 7/20000),  
            ({'international': True, 'fashion_forward': True}, 3105/20000),  
            ({'international': True, 'fashion_forward': True, 'german_speaker': True}, 423/20000),  
            ({'international': True, 'fashion_forward': True, 'vinyl_collector': True}, 17/20000),  
            ({'international': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 10/20000),  
            ({'international': True, 'fashion_forward': True, 'queer_friendly': True}, 39/20000),  
            ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 34/20000),  
            ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 18/20000),  
            ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 23/20000),  
            ({'underground_veteran': True}, 132/20000),  
            ({'underground_veteran': True, 'german_speaker': True}, 3096/20000),  
            ({'underground_veteran': True, 'vinyl_collector': True}, 40/20000),  
            ({'underground_veteran': True, 'vinyl_collector': True, 'german_speaker': True}, 157/20000),  
            ({'underground_veteran': True, 'queer_friendly': True}, 5/20000),  
            ({'underground_veteran': True, 'queer_friendly': True, 'german_speaker': True}, 40/20000),  
            ({'underground_veteran': True, 'queer_friendly': True, 'vinyl_collector': True}, 19/20000),  
            ({'underground_veteran': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 62/20000),  
            ({'underground_veteran': True, 'fashion_forward': True}, 247/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'german_speaker': True}, 2095/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'vinyl_collector': True}, 11/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 44/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True}, 37/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 49/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 32/20000),  
            ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 46/20000),  
            ({'underground_veteran': True, 'international': True}, 887/20000),  
            ({'underground_veteran': True, 'international': True, 'german_speaker': True}, 281/20000),  
            ({'underground_veteran': True, 'international': True, 'vinyl_collector': True}, 21/20000),  
            ({'underground_veteran': True, 'international': True, 'vinyl_collector': True, 'german_speaker': True}, 66/20000),  
            ({'underground_veteran': True, 'international': True, 'queer_friendly': True}, 24/20000),  
            ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'german_speaker': True}, 15/20000),  
            ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'vinyl_collector': True}, 20/20000),  
            ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 36/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True}, 5223/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'german_speaker': True}, 479/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'vinyl_collector': True}, 27/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 35/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True}, 89/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 59/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 50/20000),  
            ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 76/20000),  
        ]
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Supported scenarios: 1, 2, 3")
    