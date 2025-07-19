let previousExecution = null;
// Carrega execução anterior do localStorage ao iniciar
try {
    const prev = localStorage.getItem('previousExecution');
    if (prev) previousExecution = JSON.parse(prev);
} catch (e) { previousExecution = null; }

function loadData() {
    // Agora, use o caminho relativo, e o proxy no package.json cuidará do resto
    fetch('/info', { // APENAS '/info', sem 'http://localhost:3000'
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = '';

            // Exibe a execução anterior, se houver
            if (previousExecution) {
                const prev = previousExecution;
                const prevRow = document.createElement('tr');
                prevRow.style.backgroundColor = '#f0f0f0';
                prevRow.innerHTML = `
                    <td>${prev.timestamp || 'Execução anterior'}</td>
                    <td>${prev.parameters?.pergunta || 'N/A'}</td>
                    <td>${prev.metrics?.response_length !== undefined ? prev.metrics.response_length : 'N/A'}</td>
                    <td>${prev.metrics?.response_word_count !== undefined ? prev.metrics.response_word_count : 'N/A'}</td>
                    <td>${prev.metrics?.response_sentence_count !== undefined ? prev.metrics.response_sentence_count : 'N/A'}</td>
                    <td>${prev.metrics?.response_alpha_ratio !== undefined ? prev.metrics.response_alpha_ratio.toFixed(4) : 'N/A'}</td>
                    <td>${prev.metrics?.response_avg_word_length !== undefined ? prev.metrics.response_avg_word_length.toFixed(4) : 'N/A'}</td>
                    <td>${(prev.metrics?.latency_ms !== undefined && prev.metrics?.latency_ms !== null) ? prev.metrics.latency_ms.toFixed(2) : 'N/A'}</td>
                    <td>${(prev.metrics?.manual_rouge1 !== undefined && prev.metrics?.manual_rouge1 !== null) ? prev.metrics.manual_rouge1.toFixed(4) : 'N/A'}</td>
                    <td>${(prev.metrics?.manual_bleu !== undefined && prev.metrics?.manual_bleu !== null) ? prev.metrics.manual_bleu.toFixed(4) : 'N/A'}</td>
                `;
                tbody.appendChild(prevRow);
            }

            if (data && data.metrics && data.parameters) {
                const metrics = data.metrics;
                const parameters = data.parameters;

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${new Date().toLocaleString()}</td>
                    <td>${parameters.pergunta || 'N/A'}</td>
                    <td>${metrics.response_length !== undefined ? metrics.response_length : 'N/A'}</td>
                    <td>${metrics.response_word_count !== undefined ? metrics.response_word_count : 'N/A'}</td>
                    <td>${metrics.response_sentence_count !== undefined ? metrics.response_sentence_count : 'N/A'}</td>
                    <td>${metrics.response_alpha_ratio !== undefined ? metrics.response_alpha_ratio.toFixed(4) : 'N/A'}</td>
                    <td>${metrics.response_avg_word_length !== undefined ? metrics.response_avg_word_length.toFixed(4) : 'N/A'}</td>
                    <td>${(metrics.latency_ms !== undefined && metrics.latency_ms !== null) ? metrics.latency_ms.toFixed(2) : 'N/A'}</td>
                    <td>${(metrics.manual_rouge1 !== undefined && metrics.manual_rouge1 !== null) ? metrics.manual_rouge1.toFixed(4) : 'N/A'}</td>
                    <td>${(metrics.manual_bleu !== undefined && metrics.manual_bleu !== null) ? metrics.manual_bleu.toFixed(4) : 'N/A'}</td>
                `;
                tbody.appendChild(row);

                // Salva a execução atual como anterior para a próxima chamada e no localStorage
                previousExecution = {
                    timestamp: new Date().toLocaleString(),
                    metrics: { ...metrics },
                    parameters: { ...parameters }
                };
                try {
                    localStorage.setItem('previousExecution', JSON.stringify(previousExecution));
                } catch (e) {}
            } else {
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="10">Nenhum dado de métrica ou metadado do modelo encontrado. Verifique o serviço BentoML e o MLflow.</td>`;
                tbody.appendChild(row);
            }
        })
        .catch(error => {
            console.error('Erro ao carregar dados:', error);
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = `<tr><td colspan="6">Erro ao carregar dados: ${error.message}</td></tr>`;
        });
}

document.addEventListener('DOMContentLoaded', loadData);