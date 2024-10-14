document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = parseFloat(value);  // Convertendo valores para número
    });

    try {
        const response = await fetch('http://localhost:5000/predict_heart_disease', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("prediction-form").style.display = "none";

        document.getElementById('result').style.display = "block";
        document.getElementById('result').innerText = `Probability of Heart Disease: ${result.probability}`;
        document.getElementById('warning').innerText = `${result.warning}`;
        
        document.getElementById('repeatButton').style.display = "block";
    } catch (error) {
        console.error('Error submitting data:', error);
        document.getElementById('result').innerText = 'An error occurred while processing the data';
    }
});


document.getElementById('repeatButton').addEventListener('click', function(e) {
    e.preventDefault();
    document.getElementById("prediction-form").style.display = "block";
    document.getElementById('result').style.display = "none";
    document.getElementById('repeatButton').style.display = "none";

});

