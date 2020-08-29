#!/bin/bash
isort ./lightsheet/
autopep8 --recursive --in-place ./lightsheet/
flake8 ./lightsheet/
