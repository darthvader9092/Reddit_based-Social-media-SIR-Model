document.addEventListener('DOMContentLoaded', () => {
    const subredditName = SUBREDDIT_NAME; // Injected from template
    const analyzeBtn = document.getElementById('analyze-trends-btn');
    const trendResultsDiv = document.getElementById('trend-results');
    const trendSelector = document.getElementById('trend-selector');
    const daysInput = document.getElementById('days-input');
    const predictBtn = document.getElementById('predict-btn');
    const chartCanvas = document.getElementById('trend-chart');
    let trendChart;

    analyzeBtn.addEventListener('click', async () => {
        trendResultsDiv.innerHTML = '<p class="placeholder">Analyzing... this may take a moment.</p>';
        analyzeBtn.disabled = true;
        const response = await fetch(`/api/trends/${subredditName}`);
        const data = await response.json();
        if (data.error) {
            trendResultsDiv.innerHTML = `<p class="placeholder" style="color: #f87171;">Error: ${data.error}</p>`;
            analyzeBtn.disabled = false;
            return;
        }
        displayTrendResults(data);
        populateTrendSelector(data.topics);
        analyzeBtn.disabled = false;
    });

    function displayTrendResults(data) {
        let html = `<strong>Topics Discovered:</strong><ul class="topic-list">`;
        data.topics.forEach(topic => {
            html += `<li><strong>Topic ${topic.id}:</strong> ${topic.keywords}</li>`;
        });
        html += '</ul>';
        trendResultsDiv.innerHTML = html;
    }

    function populateTrendSelector(topics) {
        trendSelector.innerHTML = '';
        topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic.id;
            option.textContent = `Topic ${topic.id}: ${topic.keywords}`;
            trendSelector.appendChild(option);
        });
        trendSelector.disabled = false;
        daysInput.disabled = false;
        predictBtn.disabled = false;
    }

    predictBtn.addEventListener('click', async () => {
        const topicId = parseInt(trendSelector.value);
        const days = parseInt(daysInput.value);
        if (isNaN(topicId) || isNaN(days)) return;

        predictBtn.textContent = 'Predicting...';
        predictBtn.disabled = true;

        const response = await fetch(`/api/predict_trend/${subredditName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic_id: topicId, days: days })
        });
        const data = await response.json();

        if (data.error) {
            alert(`Prediction Error: ${data.error}`);
        } else {
            updateChart(data.predictions);
        }

        predictBtn.textContent = 'Predict';
        predictBtn.disabled = false;
    });

    function initializeChart() {
        const ctx = chartCanvas.getContext('2d');
        trendChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Susceptible', data: [], borderColor: '#38BDF8', tension: 0.3, pointBackgroundColor: '#38BDF8' },
                { label: 'Infected', data: [], borderColor: '#F471B5', tension: 0.3, pointBackgroundColor: '#F471B5' },
                { label: 'Recovered', data: [], borderColor: '#6B7280', tension: 0.3, pointBackgroundColor: '#6B7280' }
            ]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: 'white' }}}, scales: { x: { ticks: { color: 'white' }}, y: { ticks: { color: 'white' }} }}
        });
    }

    function updateChart(predictions) {
        trendChart.data.labels = [];
        trendChart.data.datasets.forEach((dataset) => { dataset.data = []; });
        trendChart.data.labels = Array.from({ length: predictions.length }, (_, i) => `Day ${i + 1}`);
        trendChart.data.datasets[0].data = predictions.map(p => p[0]);
        trendChart.data.datasets[1].data = predictions.map(p => p[1]);
        trendChart.data.datasets[2].data = predictions.map(p => p[2]);
        trendChart.update();
    }

    initializeChart();
});