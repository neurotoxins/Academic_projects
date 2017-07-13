import numpy as np
import pickle, os

heroesTotal = 110
featuresTotal = heroesTotal * 2

class LogisticRegression:
    def __init__(self, model_root='logisticRegression'):
        model_path = os.path.join(model_root, 'lr_model.pkl')
        with open(model_path, 'r') as input_file:
            self.model = pickle.load(input_file)

    def transform(self, my_team, their_team):
        X = np.zeros(featuresTotal, dtype=np.int8)
        for hero_id in my_team:
            X[hero_id - 1] = 1
        for hero_id in their_team:
            X[hero_id - 1 + heroesTotal] = 1
        return X

    def recommend(self, my_team, their_team, hero_candidates):
        '''Returns a list of (hero, probablility of winning with hero added) recommended to complete my_team.'''
        team_possibilities = [(candidate, my_team + [candidate]) for candidate in hero_candidates]

        prob_candidate_pairs = []
        for candidate, team in team_possibilities:
            query = self.transform(team, their_team)
            prob = self.score(query) #self.model.predict_proba(query)[0][1]
            prob_candidate_pairs.append((prob, candidate))
        prob_candidate_pairs = sorted(prob_candidate_pairs, reverse=True)[0:5 - len(my_team)]
        return prob_candidate_pairs

    def score(self, query):
        '''Score the query using the model, considering both radiant and dire teams.'''
        radiant_query = query
        dire_query = np.concatenate((radiant_query[heroesTotal:featuresTotal], radiant_query[0:heroesTotal]))
        rad_prob = self.model.predict_proba(radiant_query)[0][1]
        dire_prob = self.model.predict_proba(dire_query)[0][0]
        return (rad_prob + dire_prob) / 2

    def predict(self, dream_team, their_team):
        '''Returns the probability of the dream_team winning against their_team.'''
        dream_team_query = self.transform(dream_team, their_team)
        return self.score(dream_team_query)
        #return self.model.predict_proba(dream_team_query)[0][1]