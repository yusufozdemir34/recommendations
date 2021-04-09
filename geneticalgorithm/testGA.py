from geneticalgorithm.ApplyGeneticAlgorithm import GeneticAlgorithm
from service.ModelService import load_model_as_np

utility, test, user, item, user_user_pearson = load_model_as_np()
n_users = len(user)

ga = GeneticAlgorithm.cluster_by_genetic_algoritm(n_users, user_user_pearson)

print("test ediyoruz. ")
