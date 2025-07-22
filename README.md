Datathon_Recrutamento
==============================

Este projeto explora o desenvolvimento de Agentes de inteligência artficial para Recrutamento, focando em como utilizar os dados históricos de contratações para treinar um agente e simular o papel de um entrevistador.Este projeto busca otimizar e padronizar o estagio inicial do processo de recrutamento, oferecendo uma ferramenta inovadora que pode agilizar a triagem de candidatos e oferecer insights baseados em daddos complementando o trabalho dos recrutadores humanos.

Estrutura do projeto
------------

 - <b>model.py</b>: Arquivo cria duas classes RAGAgent e RAGRunnable. RAGAgent define métodos com configurações iniciais e o modelo utilizado para realizar embeddings no prompt do usuário (entrada da API) <b>'sentence-transformers/all-MiniLM-L6-v2'</b>, a base de conhecimento criada utiliza o mesmo modelo para realização de embeddings <i>(ingestion.py)</i>, o prompt busca uma correspondência na base de conhecimento por busca semântica. A classe RAGRunnable é responsável pelo instanciamento da classe RAGAgent e implementação do método predict que executa a inferência, a LLM escolhida foi <b> TinyLlama/TinyLlama-1.1B-Chat-v1.0 </b>, foi escolhido pelo desempenho em computadores sem aceleradores de GPU (execução apenas de CPU).

- <b>02_tratamento_arquivos.ipynb</b>: Desestrutura os arquvivos applicants.json e prospects.json (Decision), filtra situações de prospeção com candidatos aprovados (Contratado pela Decision, Aprovado,Contratado como Hunting',
'Proposta Aceita')

- <b>ingestion.py</b>: Cria embeddings a partir do tratamento de dados utilizando mesmo modelo de embeddings do prompt <b>'sentence-transformers/all-MiniLM-L6-v2'</b>.
  
- <b>experiment.py</b>: Instanciamento da classe RAGAgent para avaliar o desempenho do modelo seleciona, registro de logs de execução e métricas no MLFlow.

- <b>register.py</b>: Realiza teste de performance com modelo selecionado, realiza log de métrics e registra o modelo no Model Registry (com controle de versão) e no BentoML (possibilita o funcionamento dos endpoints).

- <b>service_v2.py</b>: Expõe endpoints de inferência e registro de métricas utilizando BentoML.

## Endpoints

- <b> POST / info </b>
    - <b> Descrição:</b> Retorna metadados do modelo registro e métricas de avaliação (Rouge e Bleu)
    - <b> Respota:</b> JSON
    - <b> Argumentos: </b> Nenhum
- <b> POST / responder </b>
    - <b> Descrição:</b> Executa RAG e retorna resposta (inferência)
    - <b> Respota:</b> JSON
    - <b> Argumentos: </b> Pergunta: String com o prompt
