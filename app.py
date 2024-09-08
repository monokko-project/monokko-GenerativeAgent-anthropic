from flask import Flask, request, jsonify
import dotenv
import os
from agent.anthropic_model import Monokko
import json
import pathlib
import warnings

warnings.simplefilter('ignore')

app = Flask(__name__)
app.config['TIMEOUT'] = 300 
app.json.ensure_ascii = False

def _set_up():
    dotenv.load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY") is None:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

def load_monokko_configs(dir_path="monokkoConfigs"):
    config_list = []
    config_path = pathlib.Path(__file__).parent / dir_path
    for config_file in config_path.iterdir():
        if config_file.suffix == ".json":
            with open(config_file, "r") as f:
                config_list.append(json.load(f))
    return config_list

mono = Monokko()
agent_names = []

@app.route('/interact', methods=['POST'])
def user_interaction():
    data = request.json
    agent_name = data.get('agent_name')
    question = data.get('question') + "(lang:ja)"

    if agent_name not in agent_names:
        return jsonify({"error": "Invalid agent name"}), 400

    if not question:
        return jsonify({"error": "Question is required"}), 400

    stay_in_dialogue, mono_observation = mono.agents[agent_name].generate_dialogue_response(question)

    mono_observation = mono_observation.replace("\n", " ")
    mono_observation = mono_observation.replace("\\", "")
    mono_observation = mono_observation.replace("\"", "")
    mono_observation = mono_observation.split("said")[1].strip()
   
    return jsonify({
        "agent_name": agent_name,
        "response": mono_observation,
        "stay_in_dialogue": stay_in_dialogue
    })

@app.route('/agents', methods=['GET'])
def list_agents():
    return jsonify(agent_names)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is working"}), 200

def main():
    _set_up()
    monokko_configs = load_monokko_configs()

    global agent_names
    for config in monokko_configs:
        agent_names.append(config["name"])
        mono.create_agent(config["name"], agent_traits=config["traits"], agent_status=config["status"])
        mono.agents[config["name"]].add_observations(config["observations"])

    app.run(debug=True, host="0.0.0.0", port=11030)

if __name__ == "__main__":
    main()