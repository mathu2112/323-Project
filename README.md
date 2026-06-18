### CSCI323 Modern Artificial Intelligence — Spring 2026
# Project Overview

This project develops a machine learning solution for Middle East Infra Contracting LLC, 
a UAE-based road infrastructure and marking company. The system predicts whether a construction project will be completed on time or delayed, using historical project data extracted from the company's ERP system.

The model enables project managers to identify high-risk projects at the planning stage — 
before execution begins — allowing early intervention and better resource allocation.

---
## Problem Statement

Middle East Infra Contracting LLC currently has no systematic mechanism to identify 
which projects are likely to be delayed before or during execution. Project performance 
is assessed reactively, meaning delays are only confirmed after they have already occurred. 
This project addresses that gap using a supervised machine learning classification model 
trained on historical project records.

## Solution

A Random Forest Classifier is trained on historical ERP project data to predict 
project delay outcomes. The model outputs:

- A binary prediction: **On Time** or **Delayed**
- A **delay probability score** (0–100%)
- A **risk level**: Low / Medium / High

---

## Dataset

- Source: Middle East Infra Contracting LLC (ERP system)
- Key features: Emirate, Type of Work, Contractor, Allocated Budget, 
  Labor Hours, Planned Duration, Start Month
