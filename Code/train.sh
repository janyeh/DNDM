#!/bin/bash
git pull origin main
python train.py
python notify.py