# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir requests boto3 numpy
COPY run.py .
CMD ["python", "-u", "run.py"]
