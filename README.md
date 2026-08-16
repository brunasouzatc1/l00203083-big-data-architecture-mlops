# l00203083 - Big Data Architecture and MLOps Iris

## Project Overview

This project implements a machine learning pipeline for Iris species classification using MLOps practices and containerised deployment.

## Machine Learning

The project uses the Iris dataset in data/l00203083_Iris_dataset.csv.

A Random Forest classifier is used with sepal length, sepal width, petal length and petal width as features.

The model achieved 90% test accuracy.

## API

The project provides a Flask REST API.

GET / returns information about the API.

POST /predict accepts Iris measurements and returns the predicted species.

Example prediction:

{"prediction": "Iris-setosa"}

## Docker

The application is containerised using Docker and Python 3.11.

## MLOps

GitHub Actions provides Continuous Integration, Continuous Training and Continuous Deployment workflows.

The workflows are stored in .github/workflows/.

## Git Branching

The repository uses main for the stable version and develop for development and testing.

## Kubernetes

The Flask API was deployed to a local Kind Kubernetes cluster.

The deployment uses two replicas and a Kubernetes Service.

The final Kubernetes API test successfully returned Iris-setosa.

## Project Structure

.github/workflows/
app/
data/
k8s/
src/
tests/
Dockerfile
requirements.txt
README.md
.gitignore

## Technologies

Python 3.11, pandas, scikit-learn, Flask, pytest, Docker, Git, GitHub Actions, Kubernetes and Kind.

## Student

Student ID: l00203083

Module: Big Data Architecture

Academic Year: 2025/26
