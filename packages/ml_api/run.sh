export IS_DEBUG=${DEBUG:-false}
exec gunicorn --bind 0.0.0.0:5000 --log-level debug --access-logfile - --error-logfile - run:app --timeout 1200