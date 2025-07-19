function loadData() {
    // Altere a URL para o endpoint /info do seu serviço BentoML
    fetch('http://localhost:3000/info') // Use a URL do seu serviço BentoML
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = '';

            // Verifica se há dados de métricas e parâmetros
            if (data && data.metrics && data.parameters) {
                const metrics = data.metrics;
                const parameters = data.parameters;

                // Cria uma única linha com os metadados e métricas do modelo registrado
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${new Date().toLocaleString()}</td>
                    <td>${parameters.pergunta || 'N/A'}</td>
                    <td>${metrics.response_length ? 'Resposta gerada (veja logs do MLflow)' : 'N/A'}</td>
                    <td>${(metrics.latency_ms !== undefined && metrics.latency_ms !== null) ? metrics.latency_ms.toFixed(2) : 'N/A'}</td>
                    <td>${(metrics.manual_rouge1 !== undefined && metrics.manual_rouge1 !== null) ? metrics.manual_rouge1.toFixed(4) : 'N/A'}</td>
                    <td>${(metrics.manual_bleu !== undefined && metrics.manual_bleu !== null) ? metrics.manual_bleu.toFixed(4) : 'N/A'}</td>
                `;
                tbody.appendChild(row);

                // Opcional: Adicionar uma linha para cada 'log_prediction_request' se desejar
                // Isso exigiria buscar os runs de log de predição do MLflow Tracking Server
                // e extrair as informações, o que é mais complexo e pode ser feito se realmente necessário.
                // Por enquanto, focamos em exibir as métricas do *modelo registrado*.

            } else {
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="6">Nenhum dado de métrica ou metadado do modelo encontrado. Verifique o serviço BentoML e o MLflow.</td>`;
                tbody.appendChild(row);
            }
        })
        .catch(error => {
            console.error('Erro ao carregar dados:', error);
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = `<tr><td colspan="6">Erro ao carregar dados: ${error.message}</td></tr>`;
        });
}

// Carrega os dados automaticamente quando a página é carregada
document.addEventListener('DOMContentLoaded', loadData);