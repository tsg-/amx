from flask import Flask, render_template_string, request, Response, stream_with_context
import requests
import time
import json

app = Flask(__name__)

# Two vLLM API endpoints
VLLM_SERVICES = {
    "vLLM with AMX (Port 8000)": "http://localhost:8000/v1/chat/completions",
    "vLLM No AMX (Port 8001)": "http://localhost:8001/v1/chat/completions"
}

# Model name
VLLM_MODEL = "ibm-granite/granite-3.3-8b-instruct"

# Predefined questions
QUESTIONS = [
    "What types of computations are most predominate in AI/ML?",
    "Tell me a joke that only a Technologist would get.",
    "Summarize the concept of Machine Learning.",
    "What is the difference between TCP and UDP?",
    "Why does a high CPU tCase complication air cooling?"
]

# HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AMX on vLLM</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        select, button { padding: 8px; margin-top: 10px; }
        td { border: 0px solid black; text-align: center; vertical-align: middle; }
         /* Define the blinking animation */
     @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0; }
          }
      .blinking {
        color: white;               /* Text color */
        font-weight: bold;        /* Make it bold */
        animation: blink 1s infinite; /* Apply animation */
          }
      .circle { border-radius: 50%; }
      .rounded { border-radius: 15px; }
        #answer { margin-top: 20px; padding: 10px; background: #f0f0f0; white-space: pre-wrap; }
        #metrics { margin-top: 10px; font-size: 0.9em; color: white; }
    </style>
</head>
<body style="background-color: #0071C5;">
    <h1 style="color: white"><B>DEMO:</B> Intel Advanced Matrix eXtensions (AMX) on vLLM</h1>

    Enables the user to see AMX performance impact for vLLM workloads using the exact same hardware.

    <P>

    <table cellspacing="5" >

    <TH style="background-color: #33ff56;"><label for="service"><B>1.</B> Select vLLM Service:</label></TD>
    <TH style="background-color: #33ff58; width: 300px"><label for="question"><B>2.</B> Select a Question:</label></TD>
    <TH class="blinking" style="width: 140px">Click to Start:</TD>
    <TH rowspan="2" style="width: 300px"><img src="/static/intel-logo.jpg" alt="Intel Corporation Logo" width="240" height="100" class="rounded"></TD>

    <TR>

    <TD>
    <select id="service">
        {% for name in services %}
            <option value="{{ name }}">{{ name }}</option>
        {% endfor %}
    </select>
    </TD>

    <TD>
    <select id="question">
        <option value="">-- Please choose --</option>
        {% for q in questions %}
            <option value="{{ q }}">{{ q }}</option>
        {% endfor %}
    </select>
    </TD>

    <TD>
    <button id="askBtn"><B>ASK</B></button>
    </TD>

    </table>

    <br>

    <B><UL>vLLM Response:</UL></B>

    <div id="answer" style="width:1020px; max-height:150px; background-color:lightblue; overflow: auto; border:1px solid black;"></div>
    <div id="metrics"></div>

    <script>
        document.getElementById("askBtn").addEventListener("click", function() {
            document.getElementById("answer").textContent = "";
            document.getElementById("metrics").textContent = "";

            const service = document.getElementById("service").value;
            const question = document.getElementById("question").value;

            if (!service || !question) {
                document.getElementById("metrics").textContent = "Please select both a service and a question.";
                return;
            }

            const eventSource = new EventSource(`/stream?service=${encodeURIComponent(service)}&question=${encodeURIComponent(question)}`);

            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.token) {
                    document.getElementById("answer").textContent += data.token;
                }
                if (data.metrics) {
                    document.getElementById("metrics").textContent =
                        `Time to first token: ${data.metrics.time_to_first_token.toFixed(2)}s | ` +
                        `Total time: ${data.metrics.total_time.toFixed(2)}s | ` +
                        `Tokens/sec: ${data.metrics.tokens_per_second.toFixed(2)}`;
                }
                if (data.done) {
                    eventSource.close();
                }
            };

            eventSource.onerror = function() {
                document.getElementById("metrics").textContent = "Error streaming from server.";
                eventSource.close();
            };
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE, questions=QUESTIONS, services=VLLM_SERVICES.keys())

@app.route("/stream", methods=["GET"])
def stream():
    service_name = request.args.get("service")
    question = request.args.get("question", "").strip()

    if not service_name or service_name not in VLLM_SERVICES:
        return "Invalid service", 400
    if not question:
        return "Invalid question", 400

    api_url = VLLM_SERVICES[service_name]

    def generate():
        start_time = time.time()
        first_token_time = None
        token_count = 0

        payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 300,
            "temperature": 0,
            "seed": 12345,
            "stream": True
        }

        try:
            with requests.post(api_url, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        total_time = time.time() - start_time
                        tokens_per_second = (token_count / total_time) if total_time > 0 else 0
                        metrics = {
                            "time_to_first_token": (first_token_time - start_time) if first_token_time else 0,
                            "total_time": total_time,
                            "tokens_per_second": tokens_per_second
                        }
                        yield f"data: {json.dumps({'done': True, 'metrics': metrics})}\n\n"
                        break
                    try:
                        data_json = json.loads(data_str)
                        token = data_json["choices"][0]["delta"].get("content", "")
                        if token:
                            token_count += 1
                            if first_token_time is None:
                                first_token_time = time.time()
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except Exception:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

