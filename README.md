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
