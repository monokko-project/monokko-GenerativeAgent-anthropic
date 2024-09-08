import dotenv
import os
# from agenet import anthropic_model
import warnings
from agent.anthropic_model import Monokko
import json
import pathlib
warnings.simplefilter('ignore')
import pandas as pd

from agent.utils import get_emotion_parameter

def _set_up():
    dotenv.load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY") is None:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

def load_monokko_configs(dir_path="monokkoConfigs"):
    config_list = []
    
    # another way without __file__
    file_path = pathlib.Path.cwd()
    config_path = file_path / dir_path

    for config_file in config_path.iterdir():
        if config_file.suffix == ".json":
            with open(config_file, "r") as f:
                config_list.append(json.load(f))

    return config_list


def run_conversation(agents: dict, agent_keys: list,  initial_observation: str) -> None:
    """Runs a conversation between agents."""
    # _, observation = agents[1].generate_reaction(initial_observation)
    _, observation = agents[agent_keys[0]].generate_dialogue_response(initial_observation)

    # print(observation)
    turns = 0

    ## dataframe dict
    agnet_1  = {
        "happiness": [],
        "sadness": [],
        "fear": [],
        "anger": [],
        "surprise": []
    }

    agnet_2  = {
        "happiness": [],
        "sadness": [],
        "fear": [],
        "anger": [],
        "surprise": []
    }

    # return
    while True:
        break_dialogue = False
        for agent_name, agent_inst in agents.items():
            stay_in_dialogue, observation = agent_inst.generate_dialogue_response(
                observation
            )
            
            obs, emotion, action = get_emotion_parameter(observation)
            print(obs)
            print(emotion)
            if action:
                print(f"行動: {action}")

            if agent_name == agent_keys[0]:
                for key, value in emotion.items():
                    agnet_1[key].append(value)
            else:
                for key, value in emotion.items():
                    agnet_2[key].append(value)
            

            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                # None
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1


    df_agent_1 = pd.DataFrame(agnet_1)
    df_agent_2 = pd.DataFrame(agnet_2)

    # to_csv
    df_agent_1.to_csv("agent_1.csv", index=False)
    df_agent_2.to_csv("agent_2.csv", index=False)

def run(mono: Monokko, agent_names) -> None:
    ### Interact with agents

    names_text = ", ".join(agent_names)
    observation = f"他のエージェントたち（{names_text}）に心配事があれば相談してみて！思いついたことは適当にしゃべってみよう"

    run_conversation(mono.agents, agent_names, observation)


def create_merged_agent(mono: Monokko, agent_names, merged_agent_name: str = "きゅーぶん") -> None:
    
    mono.merge_agents(agent_names, merged_agent_name)
    print(mono.agent_merged)

    return mono


def user_interaction(mono: Monokko, agent_names) -> None:

    while True:
        print("Which agent would you like to interact with?")
        for i, agent_name in enumerate(agent_names):
            print(f"{i+1}. {agent_name}")


        agent_index = int(input("Enter the agent number: ")) - 1

        if agent_index < 0 or agent_index >= len(agent_names):
            if agent_index == -1:
                break

            print("Invalid agent number. Please try again.")
            continue

        agent_name = agent_names[agent_index]

        print(f"Interacting with {agent_name}...")
        observation = input("Enter your observation: ")

        stay_in_dialogue, mono_observation = mono.agents[agent_name].generate_dialogue_response(
            observation
        )

        print(mono_observation)


_set_up()

monokko_configs = load_monokko_configs()
mono = Monokko()

### Create agents

agent_names = []

for config in monokko_configs:
    agent_names.append(config["name"])
    mono.create_agent(config["name"], agent_traits=config["traits"], agent_status=config["status"])
    # print(mono.agents[config["name"]].get_summary())
    mono.agents[config["name"]].add_observations(config["observations"])
    print(mono.agents[config["name"]].get_summary())


# user_interaction(mono, agent_names)
# mono.merge_agents(agent_names, merged_agent_name="きゅーぶん")
# print(mono.agent_merged.get_summary())

run(mono, agent_names)



None 
