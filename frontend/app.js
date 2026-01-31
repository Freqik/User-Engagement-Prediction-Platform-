const API_URL = "http://localhost:8000";

document.getElementById('churnForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // UI Elements
    const submitBtn = document.querySelector('.btn-submit');
    const resultCard = document.getElementById('resultCard');
    const riskBadge = document.getElementById('riskBadge');
    const probText = document.getElementById('probText');
    const meterFill = document.getElementById('meterFill');
    const resultExplanation = document.getElementById('resultExplanation');

    // Basic Validation
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    if (!data.tenure || data.tenure < 0) {
        alert("Please enter a valid number for Time with Company.");
        return;
    }
    if (!data.MonthlyCharges || data.MonthlyCharges < 0) {
        alert("Please enter a valid Monthly Bill amount.");
        return;
    }

    // Prepare Data for API
    // We send defaults for hidden fields, and convert numbers
    const payload = {
        ...data,
        SeniorCitizen: parseInt(data.SeniorCitizen),
        tenure: parseInt(data.tenure),
        MonthlyCharges: parseFloat(data.MonthlyCharges),
        TotalCharges: data.TotalCharges ? parseFloat(data.TotalCharges) : 0,

        // Ensure required hidden fields are passed if they were missing from formData for any reason
        // Note: They are present in HTML, so FormData catches them.
    };

    // UI Loading State
    const originalText = submitBtn.innerText;
    submitBtn.innerText = "Analyzing Risk...";
    submitBtn.disabled = true;
    submitBtn.style.opacity = "0.7";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }

        const result = await response.json();

        // Display Logic
        resultCard.style.display = 'block';

        // Probability Formatting
        const probPercent = (result.churn_probability * 100).toFixed(1) + "%";
        probText.innerText = probPercent;
        meterFill.style.width = probPercent;

        // Risk Logic & Styling
        resultCard.classList.remove('low', 'medium', 'high');
        riskBadge.classList.remove('badge-low', 'badge-medium', 'badge-high');

        let riskText = "";
        let explanation = "";

        if (result.risk_category === 'HIGH') {
            riskBadge.innerText = "HIGH RISK";
            riskBadge.classList.add('badge-high');
            resultCard.classList.add('high');
            meterFill.style.backgroundColor = 'var(--danger)';
            explanation = "This customer shows patterns strongly associated with cancellation. Immediate action (e.g., offering a discount or contract upgrade) is recommended.";
        } else if (result.risk_category === 'MEDIUM') {
            riskBadge.innerText = "MEDIUM RISK";
            riskBadge.classList.add('badge-medium');
            resultCard.classList.add('medium');
            meterFill.style.backgroundColor = 'var(--warning)';
            explanation = "There are some warning signs. A proactive check-in or customer service call might help improve loyalty.";
        } else {
            riskBadge.innerText = "LOW RISK";
            riskBadge.classList.add('badge-low');
            resultCard.classList.add('low');
            meterFill.style.backgroundColor = 'var(--success)';
            explanation = "This customer appears stable and satisfied based on their profile. No immediate churn risk detected.";
        }

        resultExplanation.innerText = explanation;

        // Smooth scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error("Prediction Error:", error);
        alert("Unable to analyze risk right now.\n\nPlease check if the backend server is running and try again.");
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
        submitBtn.style.opacity = "1";
    }
});
