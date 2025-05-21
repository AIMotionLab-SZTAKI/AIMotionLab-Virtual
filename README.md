# Windflow Augmentation

This branch introduces **Windflow simulation integration** into the AIMotionLab-Virtual environment. It adds support for defining static scenes via MuJoCo-style `.xml` files and integrates wind simulation using OpenFOAM. The simulated airflow data can then be used in real-time drone physics.

---

## Key Features

- Parse custom `.xml` scene files into geometry objects.
- Preprocess and convert scenes into OpenFOAM-compatible STL files.
- Run wind simulations with OpenFOAM.
- Export wind vector fields to `.csv`.
- Sample wind data in real time during drone flight.

---

## Docker Setup (Recommended)

A prebuilt Docker image with **all dependencies** (OpenFOAM, ParaView, Python libs, etc.) is available:

```bash
docker pull szabokrisztian/thesis_environment:latest
```

All the relevant shell scripts for launching and managing simulations are provided in the ```scripts/windflow``` folder.

To learn how to start the Docker container and run the system, use the scripts in that folder.
