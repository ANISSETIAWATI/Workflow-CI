# Workflow-CI: Automated Machine Learning Retraining

Project ini merupakan implementasi **Continuous Integration (CI)** untuk model Machine Learning menggunakan **MLflow Projects**. Project ini dirancang untuk melakukan re-training model secara otomatis guna memastikan performa model tetap optimal dengan data terbaru.

## Struktur Repositori
Sesuai dengan kriteria tugas, repositori ini memiliki struktur sebagai berikut:
```text
Workflow-CI/
├── MLProject/ (Folder Utama Project)
│   ├── modelling.py (Script pelatihan model dengan autolog)
│   ├── conda.yaml (Environment dependencies)
│   ├── MLProject (Instruksi entry-point MLflow)
│   ├── TelcoCustomerChurn_raw_preprocessing/ (Dataset hasil preprocessing)
│   └── docker_hub_link.txt (Tautan repositori Docker Hub)
└── README.md

Project ini telah memenuhi kriteria Advance melalui implementasi berikut:
- Automated Workflow: Menggunakan file MLProject untuk mendefinisikan entry point pelatihan model otomatis.
- Docker Hub Integration: Telah disiapkan infrastruktur untuk pengemasan model menjadi Docker Image.
- Docker Hub Repository: anissetiawati/telco-churn-ci
- Containerization Ready: Struktur ini mendukung perintah mlflow models build-docker untuk mendistribusikan model dalam bentuk kontainer yang terstandari

