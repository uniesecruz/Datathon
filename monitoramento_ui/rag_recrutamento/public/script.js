function loadData() {
    fetch('../data/data.1.log')
        .then(response => response.text())
        .then(text => {
            const lines = text.trim().split('\n');
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = '';
            lines.forEach(line => {
                try {
                    const log = JSON.parse(line);
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${log.timestamp}</td>
                        <td>${log.input}</td>
                        <td>${log.output}</td>
                        <td>${log.latency_ms}</td>
                        <td>${log.rouge1}</td>
                        <td>${log.bleu}</td>
                    `;
                    tbody.appendChild(row);
                } catch (e) {
                    // linha inválida
                }
            });
        });
}
