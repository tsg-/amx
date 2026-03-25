#!/bin/sh


echo -e "Running a quick test of each vLLM Instance:\n (Testing 1st Instance - may take up to 30 seconds to generate output)"

curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "ibm-granite/granite-3.3-8b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": true
      }'


      echo -e "\n(Testing 2nd Instance - may take up to 30 seconds to generate output)"

curl -N -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "ibm-granite/granite-3.3-8b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": true
      }'

