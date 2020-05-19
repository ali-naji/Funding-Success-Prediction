FROM python:3.7.0

ARG PIP_REMOTE_PACKAGE
ARG TRUSTED_HOST
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV FLASK_APP run.py

RUN adduser --disabled-password --gecos '' ml-api-user
ADD packages/ml_api /opt/ml_api
WORKDIR /opt/ml_api

RUN pip install --upgrade pip awscli
RUN pip install -r requirements.txt
RUN aws s3 cp s3://mytrainedmodels/funding_model/ $(python -c "import funding_model; import os; print(os.path.dirname(funding_model.__file__)+'/trained_models')")  --recursive

RUN chown -R ml-api-user ./
RUN chmod +rwx ml-api-user run.sh

EXPOSE 5000
CMD ["bash", "./run.sh"]



