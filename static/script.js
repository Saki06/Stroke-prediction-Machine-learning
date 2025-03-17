fetch('https://your-app.onrender.com/predict', {  // Flask API URL
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(userInput)
})
.then(response => response.json())
.then(data => console.log(data));
