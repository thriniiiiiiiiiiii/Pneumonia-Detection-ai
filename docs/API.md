# API Reference — PneumoDetect AI

> **Base URL (Production):** `https://pneumodetect-api.onrender.com`
> **Base URL (Local):** `http://localhost:8000`
> **Spec Version:** 1.0 · **OpenAPI:** Available at `/docs` (Swagger UI) and `/redoc`

---

## Authentication

The public API is currently **open** (no authentication required) for research and demo use.

For enterprise deployments, add `Authorization: Bearer <token>` header. Contact the maintainer to receive an API key.

---

## Endpoints

### `POST /predict`

Analyze a chest X-ray image and return a pneumonia diagnosis.

**Request**

```http
POST /predict
Content-Type: multipart/form-data

file: <image binary>  (JPEG, PNG, DCM supported)
```

**Constraints**

| Parameter | Constraint |
|---|---|
| File formats | `image/jpeg`, `image/png`, `application/dicom` |
| Max file size | 10 MB |
| Min resolution | 64 × 64 px |

**Response: `200 OK`**

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 92.54,
  "confidence_level": "High",
  "recommendation": "Strong indication of pneumonia. Recommend immediate medical attention.",
  "raw_score": 0.9253779053688049,
  "timestamp": "2025-08-18T15:18:33.827996Z",
  "filename": "person34_virus_76.jpeg",
  "image_size": "1648x1400",
  "cross_operator_validation_performance": {
    "accuracy": "86.0%",
    "sensitivity": "96.4%",
    "specificity": "74.8%",
    "roc_auc": "96.4%",
    "validated_on": "485 independent samples"
  },
  "disclaimer": "AI screening tool only. Always consult a qualified healthcare professional."
}
```

**Confidence Level Mapping**

| `raw_score` Range | `confidence_level` | Recommendation |
|---|---|---|
| 0.80 – 1.00 | `"High"` | Strong indication — immediate review recommended |
| 0.60 – 0.79 | `"Moderate"` | Possible indication — clinical correlation advised |
| 0.50 – 0.59 | `"Low"` | Borderline result — radiologist review required |
| 0.00 – 0.49 | `"Normal"` | No significant finding — routine follow-up |

**Error Responses**

| Code | Reason |
|---|---|
| `400` | Invalid file type or corrupted image |
| `413` | File exceeds maximum size (10MB) |
| `422` | Unprocessable entity — malformed request |
| `500` | Internal inference error |

---

### `GET /health`

Returns the current health status of the API and model.

**Response: `200 OK`**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_source": "ayushirathour/chest-xray-pneumonia-detection",
  "timestamp": "2025-08-18T15:20:00Z",
  "version": "1.0.0"
}
```

---

### `GET /stats`

Returns pre-computed cross-operator validation performance statistics.

**Response: `200 OK`**

```json
{
  "cross_operator_validation": {
    "accuracy": 0.860,
    "sensitivity": 0.964,
    "specificity": 0.748,
    "roc_auc": 0.964,
    "sample_size": 485,
    "dataset": "Pneumonia Radiography Dataset (Kaggle)",
    "validation_date": "2025"
  },
  "internal_validation": {
    "accuracy": 0.948,
    "sensitivity": 0.896,
    "specificity": 1.000,
    "roc_auc": 0.988
  },
  "statistical_test": {
    "method": "Bootstrap AUC Comparison",
    "n_bootstraps": 1000,
    "mean_delta_auc": -0.0001,
    "ci_95": [-0.0115, 0.0099],
    "p_value": 0.978,
    "conclusion": "No significant difference — model generalizes robustly"
  }
}
```

---

### `GET /docs`

Interactive Swagger UI for exploring and testing all endpoints directly in your browser.

**URL:** `https://pneumodetect-api.onrender.com/docs`

---

## SDK / Integration Examples

### Python

```python
import requests

def predict_pneumonia(image_path: str, api_url: str = "https://pneumodetect-api.onrender.com") -> dict:
    """Submit a chest X-ray image for pneumonia analysis."""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{api_url}/predict",
            files={"file": f},
            timeout=30
        )
    response.raise_for_status()
    return response.json()

# Example usage
result = predict_pneumonia("chest_xray.jpg")
print(f"Diagnosis: {result['diagnosis']} ({result['confidence']:.1f}% confidence)")
```

### cURL

```bash
curl -X POST "https://pneumodetect-api.onrender.com/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/xray.jpg"
```

### JavaScript / Node.js

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function predictPneumonia(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));

  const response = await axios.post(
    'https://pneumodetect-api.onrender.com/predict',
    form,
    { headers: form.getHeaders() }
  );
  return response.data;
}
```

---

## Rate Limits

| Plan | Requests/minute | Notes |
|---|---|---|
| **Public (Default)** | 100 | Suitable for research |
| **Enterprise** | Custom | Contact maintainer |

---

## Versioning

The API is versioned via the URL prefix pattern (`/v1/`, `/v2/`, etc.) for future breaking changes. The current release (`v1`) has no prefix for simplicity.

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| `1.0.0` | Aug 2025 | Initial public release |
