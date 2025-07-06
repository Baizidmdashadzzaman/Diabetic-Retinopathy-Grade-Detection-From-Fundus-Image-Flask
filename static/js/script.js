document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const predictionResult = document.getElementById('predictionResult');
    const predictionImage = document.getElementById('predictionImage');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(event) {
            event.preventDefault(); // Prevent default form submission

            predictionResult.innerHTML = '<h2>Prediction:</h2><p>Processing...</p>';
            predictionImage.style.display = 'none';
            loadingSpinner.style.display = 'block'; // Show spinner

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            // Display image preview
            if (fileInput.files && fileInput.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    predictionImage.src = e.target.result;
                    predictionImage.style.display = 'block';
                };
                reader.readAsDataURL(fileInput.files[0]);
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                loadingSpinner.style.display = 'none'; // Hide spinner

                if (response.ok) {
                    let resultHtml = `<h2>Prediction:</h2>`;
                    resultHtml += `<p><strong>Predicted Class:</strong> ${data.prediction.class_name} (Level ${data.prediction.class_index})</p>`;
                    resultHtml += `<p><strong>Confidence:</strong> ${(data.prediction.probability * 100).toFixed(2)}%</p>`;
                    resultHtml += `<p><strong>All Probabilities:</strong></p><ul>`;
                    for (const className in data.all_probabilities) {
                        resultHtml += `<li>${className}: ${(data.all_probabilities[className] * 100).toFixed(2)}%</li>`;
                    }
                    resultHtml += `</ul>`;
                    predictionResult.innerHTML = resultHtml;
                } else {
                    predictionResult.innerHTML = `<h2 class="error">Error:</h2><p>${data.error}</p>`;
                }

            } catch (error) {
                loadingSpinner.style.display = 'none'; // Hide spinner
                predictionResult.innerHTML = `<h2 class="error">Error:</h2><p>Could not connect to the server or an unexpected error occurred: ${error.message}</p>`;
            }
        });
    }
});