from locust import HttpUser, between, task

# Ajuste: importar EXAMPLE_INPUT do service_v2
from service_v2 import EXAMPLE_INPUT

class RAGRecrutamentoTestUser(HttpUser):
    @task
    def responder(self):
        url = "/responder"
        self.client.post(url, json={"perguntas": EXAMPLE_INPUT})

    wait_time = between(0.05, 2)
